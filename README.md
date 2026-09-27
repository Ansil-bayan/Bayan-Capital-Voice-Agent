# Bayan Company Voice Agent — RAG + Live Signal Nudging

A Vapi-powered voice agent that answers business-loan questions from a
ChromaDB knowledge base, with a live signal-detection layer that nudges
the agent's behavior mid-call (compliance gaps, frustration, cross-sell
opportunities, payment difficulty) based on what the customer is saying.

## Architecture

Two subsystems, wired together through one webhook server:

**1. RAG voice agent (Vapi)**
- `setup_vapi.py` — registers the `search_knowledge_base`,
  `submit_lead_qualification`, and `transferCall` tools with Vapi, then
  creates the assistant (OpenAI gpt-4o-mini + Cartesia voice + Deepgram
  transcription). Also enables `monitorPlan.controlEnabled` (for live
  nudge injection) and `serverMessages` (so Vapi actually sends
  transcript/tool-call/status events to our server).
- `vapi_server.py` — the single webhook server Vapi talks to. Handles
  knowledge-base tool calls (queries ChromaDB) and live transcript
  events (feeds the signal engine below).

**2. Live signal & nudge engine**
- `signal_engine.py` — regex-based rule matching over the running call
  transcript; detects signals like `COMPLIANCE_GAP`, `RISING_FRUSTRATION`,
  `PAYMENT_DIFFICULTY`, `CROSS_SELL_OPPORTUNITY`.
- `nudge_controller.py` — dedups/cools down raw signals into actual
  nudges (confidence threshold + cooldown window).
- `session_manager.py` — per-call session state (transcript so far,
  a dedicated `NudgeController` instance, queued nudges), keyed by
  Vapi's `call.id`.

**How a nudge reaches the model:**
- *Urgent* nudges (priority ≤ 2, e.g. compliance/frustration) are pushed
  immediately into the live call via Vapi's Live Call Control API
  (`monitor.controlUrl` + `add-message`), so the agent's very next reply
  reflects it.
- *Lower-priority* nudges (e.g. cross-sell) are queued and merged into
  the next `search_knowledge_base` tool response as an
  `[AGENT GUIDANCE]` block alongside the retrieved KB context.

**Offline testing (no Vapi/OpenAI usage required):**
- `pipeline_runner.py` — legacy local-audio test rig using Whisper for
  ASR; superseded by Vapi's own Deepgram transcripts in production, kept
  only for standalone signal-engine testing against a `.wav` file.
- `test_harness.py` — transcribes a `.wav` file locally, replays it
  through `signal_engine.py` + `nudge_controller.py`, and generates two
  side-by-side transcripts (`baseline_transcript.txt` vs
  `nudge_aware_transcript.txt`) using Groq's OpenAI-compatible API, so
  you can sanity-check nudge behavior before spending real Vapi minutes.

## Demo

# Creating Knowledge Base

<img width="1020" height="467" alt="image" src="https://jam.dev/c/92a6db1e-d2a1-48b6-9f0c-d8ff84b30f16" />


# Creating Voice Agent


https://jam.dev/c/7acdfca7-78c4-4480-92e9-31022b2549c4


## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file:



Populate `./chroma_db` with your knowledge base (business loan FAQs,
rates, policies) using the `all-MiniLM-L6-v2` sentence-transformer
embedding function — `vapi_server.py` expects a collection named
`knowledge_base`.

## Running the live agent

```bash
# 1. Expose your webhook publicly (for local dev)
ngrok http 8000

# 2. Start the server (keep this running)
python vapi_server.py

# 3. Register tools + create the assistant (run once)
python setup_vapi.py
```

Then test from the Vapi dashboard ("Talk to Assistant") or place a real
call via the Vapi API.

## Testing offline first

```bash
export GROQ_API_KEY=gsk_...
python test_harness.py path/to/your_audio.wav
```

Produces `baseline_transcript.txt` and `nudge_aware_transcript.txt` for
comparison, without touching Vapi or OpenAI quota.

## Notes / limitations

- `SessionStore` is in-memory — fine for a single-process deployment;
  scale to multiple workers requires backing it with Redis.
- `test_harness.py` assumes the input audio is entirely the customer's
  side of the call; diarize first if it's a two-speaker recording.
- Vapi's webhook payload shape has varied across SDK versions;
  `extract_call_id_and_control_url()` in `vapi_server.py` checks both
  known locations for `call.id` / `call.monitor.controlUrl` — verify
  against your account if nudges aren't reaching the model.
