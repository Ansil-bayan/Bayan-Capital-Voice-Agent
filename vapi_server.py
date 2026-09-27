import json

import chromadb
import httpx
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from session_manager import SessionStore
from signal_engine import SignalEngine

app = FastAPI()

# --- RAG setup (unchanged) ---------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef,
)

# --- Live-signal setup (new) ---------------------------------------------
signal_engine = SignalEngine()
session_store = SessionStore()


async def push_control_message(control_url: str, content: str, role: str = "system") -> None:
    """Fire-and-forget: inject a silent guidance message into a live call
    via Vapi's Live Call Control endpoint. Requires monitorPlan.controlEnabled
    on the assistant (see setup_vapi.py)."""
    if not control_url:
        print("[WARN] No controlUrl for this call - skipping urgent nudge push")
        return
    payload = {"type": "add-message", "message": {"role": role, "content": content}}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(control_url, json=payload)
    except Exception as e:
        print(f"[WARN] Failed to push control message: {e}")


def extract_call_id_and_control_url(body: dict) -> tuple[str, str]:
    """Vapi nests the call object slightly differently across message types
    (top-level `call` on transcript/status events, `message.call` on tool
    calls in some SDK versions) - check both."""
    message = body.get("message", {})
    call = message.get("call") or body.get("call") or {}
    call_id = call.get("id", "")
    control_url = (call.get("monitor") or {}).get("controlUrl", "")
    return call_id, control_url


@app.post("/vapi-events")
async def vapi_events(request: Request):
    """Single Server URL handling transcript, tool-call, and status-update
    events. Point the assistant's serverUrl (and serverMessages, see
    setup_vapi.py) here."""
    body = await request.json()
    message = body.get("message", {})
    msg_type = message.get("type")
    call_id, control_url = extract_call_id_and_control_url(body)

    # --- Live transcript -> signal detection -----------------------------
    if msg_type == "transcript":
        if message.get("role") == "user" and message.get("transcriptType") == "final":
            session = session_store.get_or_create(call_id, control_url)
            text = message.get("transcript", "")
            if text:
                nudges = session.ingest_transcript(text, signal_engine)
                for n in nudges:
                    if n.get("priority", 99) <= 2:
                        await push_control_message(
                            session.control_url,
                            f"[LIVE SIGNAL - {n['signal_id']}] {n['nudge_text']} "
                            "Adjust your very next response to account for this.",
                        )
        return JSONResponse(content={"received": True})

    # --- Call ended -> clean up session -----------------------------------
    if msg_type == "status-update" and message.get("status") == "ended":
        session_store.drop(call_id)
        return JSONResponse(content={"received": True})

    # --- Tool call (knowledge-base search) --------------------------------
    if msg_type == "tool-calls" or "toolCall" in body or (
        "message" in body and "toolCalls" in body.get("message", {})
    ):
        return await handle_kb_tool_call(body, call_id)

    return JSONResponse(content={"received": True})


async def handle_kb_tool_call(body: dict, call_id: str):
    print("\n--- [DEBUG] Raw Payload Received from Vapi ---")
    print(json.dumps(body, indent=2))

    tool_call_id = None
    query_text = ""

    # Schema Style 1: Standard Vapi Tool Call Webhook
    if "message" in body and "toolCalls" in body["message"]:
        tool_call = body["message"]["toolCalls"][0]
        tool_call_id = tool_call.get("id")
        args = tool_call.get("function", {}).get("arguments", {})
        query_text = args.get("query", "") if isinstance(args, dict) else json.loads(args).get("query", "")

    # Schema Style 2: Direct Server Tool Webhook
    elif "toolCall" in body:
        tool_call = body["toolCall"]
        tool_call_id = tool_call.get("id")
        args = tool_call.get("function", {}).get("arguments", {})
        query_text = args.get("query", "") if isinstance(args, dict) else json.loads(args).get("query", "")

    print(f"--- [DEBUG] Extracted Query: '{query_text}' | Call ID: '{call_id}' ---")

    if not query_text:
        return JSONResponse(content={
            "results": [{"toolCallId": tool_call_id, "result": "No query was provided by the agent."}]
        })

    # Retrieve from ChromaDB
    results = collection.query(query_texts=[query_text], n_results=10)
    retrieved_docs = results["documents"][0] if results.get("documents") else []
    print(f"--- [DEBUG] Found {len(retrieved_docs)} matching documents ---")

    context_str = "\n---\n".join(retrieved_docs) if retrieved_docs else "No relevant information found in knowledge base."

    # --- Merge queued (non-urgent) nudges into the KB response -----------
    guidance_block = ""
    if call_id:
        session = session_store.get_or_create(call_id)
        pending = session.pop_pending_nudges()
        if pending:
            lines = "\n".join(f"- {n['nudge_text']}" for n in pending)
            guidance_block = f"\n\n[AGENT GUIDANCE - apply while phrasing this answer]\n{lines}"

    return JSONResponse(content={
        "results": [
            {"toolCallId": tool_call_id, "result": context_str + guidance_block}
        ]
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)