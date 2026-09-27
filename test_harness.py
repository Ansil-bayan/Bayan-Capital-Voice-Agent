"""
Local, offline test harness - run this on YOUR machine (not the sandbox),
where your OPENAI_API_KEY and network already work. No Vapi call needed.

What it does:
  1. Transcribes a raw .wav file locally with faster-whisper (same "tiny.en"
     model your pipeline_runner.py already uses).
  2. Walks the transcript segment by segment, running your existing
     SignalEngine + NudgeController exactly as pipeline_runner.py does.
  3. For every segment, generates TWO agent replies with gpt-4o-mini:
       - baseline_transcript.txt    -> KB context only, nudges ignored
       - nudge_aware_transcript.txt -> KB context + whatever guidance would
                                        have fired/queued at that point
  4. Writes both files so you can diff them side by side.

Install:
    pip install faster-whisper openai chromadb

Place this file in the same directory as:
    signal_engine.py       (yours, already have it)
    nudge_controller.py    (yours - referenced by pipeline_runner.py)
    ./chroma_db/           (optional - if present, used for real KB retrieval;
                             if missing, the script falls back to a placeholder
                             context string so it still runs)

Uses Groq (OpenAI-compatible endpoint, generous free tier) instead of OpenAI directly.
Get a key at https://console.groq.com/keys - no separate SDK needed, the `openai`
package works as-is pointed at Groq's base URL.

Run:
    export GROQ_API_KEY=gsk_...
    python test_harness.py path/to/your_audio.wav
"""

import os
import sys
from typing import Dict, List

from faster_whisper import WhisperModel
from openai import OpenAI

from signal_engine import SignalEngine
from nudge_controller import NudgeController

# Try to load a real KB; fall back gracefully if you don't have one set up here.
try:
    import chromadb
    from chromadb.utils import embedding_functions

    _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    _ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    _collection = _chroma_client.get_collection(name="knowledge_base", embedding_function=_ef)
    KB_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Knowledge base not available locally ({e}); using placeholder context.")
    _collection = None
    KB_AVAILABLE = False

import time



SYSTEM_PROMPT_BASE = """
You are an AI Voice Agent representing Apex Capital for Business Loans.
Answer the customer's message using ONLY the provided knowledge base context.
Keep the reply to 2-3 sentences, in a natural spoken tone.
If the context says no relevant information was found, say so briefly.
"""
GROQ_MODEL = "openai/gpt-oss-20b"

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    sys.exit("[ERROR] GROQ_API_KEY environment variable is not set. Run: $env:GROQ_API_KEY='your_key'")

client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1",
)


def retrieve_kb_context(query: str) -> str:
    if not KB_AVAILABLE or not query.strip():
        return "No relevant information found in knowledge base."
    results = _collection.query(query_texts=[query], n_results=2)
    docs = results["documents"][0] if results.get("documents") else []
    return "\n---\n".join(docs) if docs else "No relevant information found in knowledge base."


def generate_reply(customer_text: str, kb_context: str, guidance_lines: List[str]) -> str:
    guidance_block = ""
    if guidance_lines:
        guidance_block = (
            "\n\n[AGENT GUIDANCE - apply silently while phrasing your reply, never read aloud]\n"
            + "\n".join(f"- {g}" for g in guidance_lines)
        )

    user_content = (
        f'Customer said: "{customer_text}"\n\n'
        f"Knowledge base context:\n{kb_context}{guidance_block}"
    )

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_harness.py path/to/audio.wav")
        sys.exit(1)
    wav_path = sys.argv[1]

    print(f"[Harness] Loading Whisper (tiny.en) and transcribing {wav_path} ...")
    whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(wav_path, beam_size=1, language="en")
    segments = list(segments)  # materialize - faster-whisper returns a generator
    print(f"[Harness] {len(segments)} segments transcribed.")

    signal_engine = SignalEngine()
    nudge_controller = NudgeController(confidence_threshold=0.80, cooldown_seconds=25.0)

    full_transcript = ""
    pending_nudges: List[Dict] = []  # mirrors "queued for next KB tool call" in production

    baseline_lines = []
    nudge_lines = []

    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue

        full_transcript += " " + text
        call_duration_ms = seg.end * 1000  # faster-whisper timestamps are in seconds

        raw_signals = signal_engine.analyze_transcript(full_transcript, call_duration_ms)
        fired_nudges = nudge_controller.process_signals(raw_signals)

        urgent = [n for n in fired_nudges if n.get("priority", 99) <= 2]
        queued = [n for n in fired_nudges if n.get("priority", 99) > 2]
        pending_nudges.extend(queued)

        kb_context = retrieve_kb_context(text)

        # Baseline never sees any nudge, ever.
        baseline_reply = generate_reply(text, kb_context, guidance_lines=[])

        # Nudge-aware: urgent nudges apply this turn (mirrors live call-control
        # injection); queued ones accumulate and apply as soon as this turn's
        # "KB tool call" happens (mirrors the merge into the tool response).
        applied_guidance = [n["nudge_text"] for n in urgent] + [n["nudge_text"] for n in pending_nudges]
        nudge_reply = generate_reply(text, kb_context, guidance_lines=applied_guidance)
        if applied_guidance:
            pending_nudges.clear()  # "delivered" this turn

        stamp = f"[{seg.start:6.1f}s-{seg.end:6.1f}s]"
        print(f"{stamp} Customer: {text}")
        for n in fired_nudges:
            print(f"         -> nudge fired: {n['signal_id']} (priority {n['priority']})")

        baseline_lines.append(f"{stamp} Customer: {text}\nAgent (baseline): {baseline_reply}\n")
        entry = f"{stamp} Customer: {text}\nAgent (nudge-aware): {nudge_reply}\n"
        if applied_guidance:
            entry += "  [guidance applied: " + "; ".join(applied_guidance) + "]\n"
        nudge_lines.append(entry)

    with open("baseline_transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(baseline_lines))
    with open("nudge_aware_transcript.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(nudge_lines))

    print("\n[Harness] Done.")
    print("  baseline_transcript.txt     - no nudge awareness")
    print("  nudge_aware_transcript.txt  - with live + queued nudge guidance applied")


if __name__ == "__main__":
    time.sleep(1)
    main()