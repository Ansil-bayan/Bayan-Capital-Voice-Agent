import asyncio
import json
import time
from faster_whisper import WhisperModel
import numpy as np
import websockets

# Load fast, lightweight local Whisper model (CPU or CUDA)
print("[ASR Engine] Loading local Whisper model...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")


async def run_local_asr_client():
    uri = "ws://localhost:8765"
    audio_buffer = bytearray()

    async with websockets.connect(uri) as websocket:
        # Request stream start
        await websocket.send(
            json.dumps({"action": "start_stream", "file_path": "test_call.wav"})
        )

        last_transcribe_time = time.time()

        while True:
            message = await websocket.recv()

            if isinstance(message, bytes):
                # Append raw binary audio chunk to rolling buffer
                audio_buffer.extend(message)

                # Process buffer every 1.0 second of accumulated audio
                now = time.time()
                if now - last_transcribe_time >= 1.0:
                    t_start_asr = time.time()

                    # Convert 16-bit PCM bytes to float32 NumPy array normalized to [-1, 1]
                    pcm_data = (
                        np.frombuffer(audio_buffer, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32768.0
                    )

                    # Run inference on local model
                    segments, _ = model.transcribe(
                        pcm_data, beam_size=1, language="en"
                    )
                    transcript = " ".join([seg.text for seg in segments]).strip()

                    asr_latency_ms = int((time.time() - t_start_asr) * 1000)

                    if transcript:
                        print(
                            f"[Transcript Stream] Latency: {asr_latency_ms}ms | Text: \"{transcript}\""
                        )

                    last_transcribe_time = now

            else:
                data = json.loads(message)
                if data.get("event") == "stop":
                    print("[ASR Engine] Stream finished.")
                    break


if __name__ == "__main__":
    asyncio.run(run_local_asr_client())