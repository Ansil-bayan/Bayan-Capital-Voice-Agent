"""
Per-call session state for the Vapi <-> live-signal integration.

One CallSession is created per active Vapi call (keyed by Vapi's call.id).
It owns:
  - the running transcript for that call
  - a dedicated NudgeController instance (so cooldown/dedup state never
    leaks between concurrent calls)
  - a queue of "pending" nudges waiting to be merged into the next
    knowledge-base tool response

Urgent nudges are NOT queued here - the caller (vapi_server.py) pushes
those out immediately via Live Call Control and only queues the rest.
"""

import time
from typing import Dict, List

from signal_engine import SignalEngine
from nudge_controller import NudgeController

# Priorities <= this threshold are pushed live via Call Control instead
# of waiting for the next KB tool-call response.
URGENT_PRIORITY_THRESHOLD = 2


class CallSession:
    def __init__(self, call_id: str, control_url: str = ""):
        self.call_id = call_id
        self.control_url = control_url
        self.start_time = time.time()
        self.full_transcript = ""
        self.pending_nudges: List[Dict] = []
        self.nudge_controller = NudgeController(
            confidence_threshold=0.80, cooldown_seconds=25.0
        )

    def call_duration_ms(self) -> float:
        return (time.time() - self.start_time) * 1000

    def ingest_transcript(self, text: str, signal_engine: SignalEngine) -> List[Dict]:
        """Append a final user transcript chunk, run detection, and return
        the newly fired nudges (already filtered by the controller's
        cooldown/dedup logic)."""
        self.full_transcript += " " + text
        raw_signals = signal_engine.analyze_transcript(
            self.full_transcript, self.call_duration_ms()
        )
        nudges = self.nudge_controller.process_signals(raw_signals)

        for n in nudges:
            if n.get("priority", 99) > URGENT_PRIORITY_THRESHOLD:
                self.pending_nudges.append(n)

        return nudges

    def pop_pending_nudges(self) -> List[Dict]:
        """Drain and return everything queued for the next KB tool response."""
        nudges, self.pending_nudges = self.pending_nudges, []
        return nudges


class SessionStore:
    """In-memory registry of active call sessions.

    Fine for a single-process FastAPI deployment. If you run multiple
    workers/replicas, back this with Redis instead (same interface,
    JSON-serialize CallSession's mutable state).
    """

    def __init__(self):
        self._sessions: Dict[str, CallSession] = {}

    def get_or_create(self, call_id: str, control_url: str = "") -> CallSession:
        session = self._sessions.get(call_id)
        if session is None:
            session = CallSession(call_id, control_url)
            self._sessions[call_id] = session
        elif control_url and not session.control_url:
            session.control_url = control_url
        return session

    def drop(self, call_id: str) -> None:
        self._sessions.pop(call_id, None)