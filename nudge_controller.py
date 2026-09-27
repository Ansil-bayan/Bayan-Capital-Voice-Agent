import time
from typing import Dict, List, Optional


class NudgeController:

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        cooldown_seconds: float = 30.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds

        # Memory store to track active and historical nudges
        self.nudge_history = {}  # {signal_id: last_triggered_timestamp}
        self.active_nudges = {}  # {nudge_id: nudge_object}

    def process_signals(self, raw_signals: List[Dict]) -> List[Dict]:
        """Applies suppression rules and returns actionable nudges."""
        valid_nudges = []
        now = time.time()

        for signal in raw_signals:
            signal_id = signal["signal_id"]

            # 1. Confidence Threshold Guardrail
            if signal["confidence"] < self.confidence_threshold:
                continue

            # 2. Cooldown & Duplicate Suppression
            last_triggered = self.nudge_history.get(signal_id, 0)
            if (now - last_triggered) < self.cooldown_seconds:
                continue  # Suppress duplicate alert within cooldown period

            # Signal accepted -> Build actionable Nudge
            nudge = {
                "nudge_id": f"{signal_id}_{int(now)}",
                "signal_id": signal_id,
                "nudge_text": signal["nudge_text"],
                "priority": signal["priority"],
                "confidence": signal["confidence"],
                "created_at": now,
                "expires_at": now + 20.0,  # 20-second auto-expiry
            }

            self.nudge_history[signal_id] = now
            valid_nudges.append(nudge)

        # 3. Priority Order (Priority 1 = Highest)
        valid_nudges.sort(key=lambda x: x["priority"])
        return valid_nudges