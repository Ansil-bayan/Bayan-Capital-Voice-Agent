import re
import time
from typing import Dict, List, Optional


class SignalEngine:

    def __init__(self):
        # Operational signals mapped to rules and default priorities (1 = Highest)
        self.rules = [
            {
                "signal_id": "COMPLIANCE_GAP",
                "patterns": [
                    r"record(ed)? for quality",
                    r"terms and conditions",
                    r"privacy policy",
                ],
                "negation_required": True,  # Alert if missing when critical phase starts
                "trigger_keywords": [
                    "payment",
                    "credit card",
                    "ssn",
                    "policy change",
                    "social security",
                ],
                "nudge_text": "COMPLIANCE: Mandatory recording disclosure missing before handling sensitive data.",
                "priority": 1,
                "confidence": 0.95,
            },
            {
                "signal_id": "CROSS_SELL_OPPORTUNITY",
                "patterns": [
                    r"second (car|vehicle|auto)",
                    r"another house",
                    r"buying a new",
                    r"my spouse",
                    r"additional driver",
                ],
                "negation_required": False,
                "nudge_text": "CROSS-SELL: Customer mentioned additional asset/driver. Offer Multi-Policy Discount.",
                "priority": 3,
                "confidence": 0.85,
            },
            {
                "signal_id": "RISING_FRUSTRATION",
                "patterns": [
                    r"unacceptable",
                    r"speak (to|with) a manager",
                    r"ridiculous",
                    r"waste of time",
                    r"canceling my account",
                ],
                "negation_required": False,
                "nudge_text": "DE-ESCALATION: Acknowledge frustration and offer immediate resolution options.",
                "priority": 2,
                "confidence": 0.90,
            },
            {
                "signal_id": "PAYMENT_DIFFICULTY",
                "patterns": [
                    r"can't afford",
                    r"card declined",
                    r"payment plan",
                    r"late fee",
                    r"due date extension",
                ],
                "negation_required": False,
                "nudge_text": "PAYMENT SUPPORT: Present flexible payment plan or hardship assistance options.",
                "priority": 2,
                "confidence": 0.88,
            },
        ]

    def analyze_transcript(
        self, transcript: str, call_duration_ms: int
    ) -> List[Dict]:
        """Scans transcript for signals and returns raw detected events."""
        start_time = time.time()
        detected_signals = []
        text_lower = transcript.lower()

        for rule in self.rules:
            # Standard pattern match
            if not rule["negation_required"]:
                for pattern in rule["patterns"]:
                    if re.search(pattern, text_lower):
                        detected_signals.append(
                            {
                                "signal_id": rule["signal_id"],
                                "nudge_text": rule["nudge_text"],
                                "priority": rule["priority"],
                                "confidence": rule["confidence"],
                                "trigger_phrase": pattern,
                                "extraction_latency_ms": round(
                                    (time.time() - start_time) * 1000, 2
                                ),
                            }
                        )
                        break

            # Compliance Gap check (Triggered if workflow enters sensitive stage without prior disclosure)
            else:
                has_trigger = any(
                    kw in text_lower for kw in rule["trigger_keywords"]
                )
                has_disclosure = any(
                    re.search(p, text_lower) for p in rule["patterns"]
                )

                if (
                    has_trigger
                    and not has_disclosure
                    and call_duration_ms > 10000
                ):
                    detected_signals.append(
                        {
                            "signal_id": rule["signal_id"],
                            "nudge_text": rule["nudge_text"],
                            "priority": rule["priority"],
                            "confidence": rule["confidence"],
                            "trigger_phrase": "missing_mandatory_disclosure",
                            "extraction_latency_ms": round(
                                (time.time() - start_time) * 1000, 2
                            ),
                        }
                    )

        return detected_signals