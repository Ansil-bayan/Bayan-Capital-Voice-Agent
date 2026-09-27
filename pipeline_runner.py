import asyncio
import json
import time
from numpy import float32, frombuffer, int16
import numpy as np
from faster_whisper import WhisperModel
import websockets

from nudge_controller import NudgeController
from signal_engine import SignalEngine

# Configuration
STREAM_URI = "ws://localhost:8765"
WAV_FILE = "test_call.wav"


class PipelineRunner:

    def __init__(self):
        print("[Pipeline] Initializing Local Whisper Engine (tiny.en)...")
        self.whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.signal_engine = SignalEngine()
        self.nudge_controller = NudgeController(
            confidence_threshold=0.80, cooldown_seconds=25.0
        )

        self.audio_buffer = bytearray()
        self.latencies = []
        self.full_transcript = ""

    async def run(self):
        async with websockets.connect(STREAM_URI) as ws:
            # Request audio stream
            await ws.send(
                json.dumps({"action": "start_stream", "file_path": WAV_FILE})
            )
            print(f"[Pipeline] Connected to streamer. Processing real-time...")

            last_transcribe_time = time.time()
            chunk_emission_map = {}  # {chunk_index: emission_timestamp_ms}

            while True:
                msg = await ws.recv()

                # Handle metadata
                if not isinstance(msg, bytes):
                    data = json.loads(msg)
                    if data.get("event") == "chunk_meta":
                        chunk_emission_map[data["chunk_index"]] = data[
                            "emission_timestamp_ms"
                        ]
                    elif data.get("event") == "stop":
                        print("\n[Pipeline] Stream finished.")
                        break
                    continue

                # Handle raw binary audio chunk
                self.audio_buffer.extend(msg)
                now = time.time()

                # Execute ASR & Signal Pipeline every 1.0s window
                if now - last_transcribe_time >= 1.0:
                    t_start_pipeline = time.time()

                    # 1. Convert PCM Bytes to Float32 NumPy array
                    pcm_data = (
                        frombuffer(self.audio_buffer, dtype=int16).astype(
                            float32
                        )
                        / 32768.0
                    )

                    # 2. Local ASR Transcription
                    segments, _ = self.whisper.transcribe(
                        pcm_data, beam_size=1, language="en"
                    )
                    recent_text = " ".join([s.text for s in segments]).strip()
                    t_asr_done = time.time()

                    if recent_text:
                        self.full_transcript += " " + recent_text
                        call_duration_ms = len(pcm_data) / 16.0

                        # 3. Signal Extraction
                        raw_signals = self.signal_engine.analyze_transcript(
                            self.full_transcript, call_duration_ms
                        )
                        t_signal_done = time.time()

                        # 4. Nudge Control & Suppression
                        nudges = self.nudge_controller.process_signals(
                            raw_signals
                        )
                        t_nudge_done = time.time()

                        # Metrics Calculation
                        total_latency_ms = int(
                            (t_nudge_done - t_start_pipeline) * 1000
                        )
                        asr_latency_ms = int(
                            (t_asr_done - t_start_pipeline) * 1000
                        )
                        self.latencies.append(total_latency_ms)

                        # Display Dashboard Output
                        print(
                            f"\n[Transcript] ...{recent_text[-60:]}"
                        )
                        print(
                            f"├─ Latency: {total_latency_ms}ms (ASR: {asr_latency_ms}ms | Signals: {int((t_signal_done - t_asr_done)*1000)}ms)"
                        )

                        if nudges:
                            for n in nudges:
                                print(
                                    f"└─ 🚨 [NUDGE DISPATCHED] [{n['signal_id']}] (Priority {n['priority']}) -> {n['nudge_text']}"
                                )

                    last_transcribe_time = now

        self.generate_latency_report()

    def generate_latency_report(self):
        """Prints required P50/P95 latency analysis."""
        if not self.latencies:
            print("No latency metrics collected.")
            return

        p50 = np.percentile(self.latencies, 50)
        p95 = np.percentile(self.latencies, 95)

        print("\n==================================================")
        print("          REAL-TIME LATENCY REPORT (P50/P95)      ")
        print("==================================================")
        print(f" Total Processed Windows : {len(self.latencies)}")
        print(f" P50 Latency (Median)     : {round(p50, 2)} ms")
        print(f" P95 Latency (Tail)       : {round(p95, 2)} ms")
        print(f" Minimum / Maximum        : {min(self.latencies)}ms / {max(self.latencies)}ms")
        print("==================================================\n")


if __name__ == "__main__":
    runner = PipelineRunner()
    asyncio.run(runner.main() if hasattr(runner, "main") else runner.run())