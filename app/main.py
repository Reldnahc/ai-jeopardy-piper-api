import io
import os
import wave
import asyncio
import time
import onnxruntime as ort

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from piper import PiperVoice
from piper import SynthesisConfig

PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-amy-low.onnx")
PIPER_MODEL = os.getenv("PIPER_MODEL", f"/models/{PIPER_VOICE}.onnx")

MAX_INFLIGHT_PER_WORKER = int(os.getenv("MAX_INFLIGHT_PER_WORKER", "1"))
_sem = asyncio.Semaphore(MAX_INFLIGHT_PER_WORKER)

DEFAULT_VOLUME = float(os.getenv("PIPER_VOLUME", "1.0"))
DEFAULT_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
DEFAULT_NOISE_SCALE = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
DEFAULT_NOISE_W = float(os.getenv("PIPER_NOISE_W", "0.8"))

USE_CUDA = os.getenv("PIPER_USE_CUDA", "0") == "1"

app = FastAPI(title="Piper Python API", version="1.0")

voice: PiperVoice | None = None

# ---- simple in-process metrics (per worker) ----
_metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "tts_synth_seconds_sum": 0.0,
    "tts_total_seconds_sum": 0.0,
    "tts_queue_wait_seconds_sum": 0.0,
    "bytes_total": 0,
}


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    volume: float | None = None
    length_scale: float | None = None
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    normalize_audio: bool | None = None


@app.on_event("startup")
def _load_voice():
    global voice
    print("onnxruntime providers:", ort.get_available_providers(), flush=True)
    voice = PiperVoice.load(PIPER_MODEL, use_cuda=USE_CUDA)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": PIPER_MODEL,
        "use_cuda": USE_CUDA,
        "max_inflight_per_worker": MAX_INFLIGHT_PER_WORKER,
    }


def synth_to_wav_bytes_timed(text: str, syn_config: SynthesisConfig | None) -> tuple[bytes, float]:
    if voice is None:
        raise RuntimeError("voice not loaded")

    start = time.perf_counter()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)
    synth_s = time.perf_counter() - start
    return buf.getvalue(), synth_s


@app.get("/metrics")
def metrics():
    lines = []
    lines.append("# TYPE piper_requests_total counter")
    lines.append(f"piper_requests_total {_metrics['requests_total']}")
    lines.append("# TYPE piper_errors_total counter")
    lines.append(f"piper_errors_total {_metrics['errors_total']}")
    lines.append("# TYPE piper_tts_synth_seconds_sum counter")
    lines.append(f"piper_tts_synth_seconds_sum {_metrics['tts_synth_seconds_sum']}")
    lines.append("# TYPE piper_tts_total_seconds_sum counter")
    lines.append(f"piper_tts_total_seconds_sum {_metrics['tts_total_seconds_sum']}")
    lines.append("# TYPE piper_tts_queue_wait_seconds_sum counter")
    lines.append(f"piper_tts_queue_wait_seconds_sum {_metrics['tts_queue_wait_seconds_sum']}")
    lines.append("# TYPE piper_bytes_total counter")
    lines.append(f"piper_bytes_total {_metrics['bytes_total']}")
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/tts")
async def tts(req: TtsRequest):
    if voice is None:
        raise HTTPException(status_code=500, detail="Voice not loaded")

    _metrics["requests_total"] += 1

    syn_config = SynthesisConfig(
        volume=req.volume if req.volume is not None else DEFAULT_VOLUME,
        length_scale=req.length_scale if req.length_scale is not None else DEFAULT_LENGTH_SCALE,
        noise_scale=req.noise_scale if req.noise_scale is not None else DEFAULT_NOISE_SCALE,
        noise_w_scale=req.noise_w_scale if req.noise_w_scale is not None else DEFAULT_NOISE_W,
        normalize_audio=req.normalize_audio if req.normalize_audio is not None else True,
    )

    total_start = time.perf_counter()

    # Measure how long we wait for a concurrency slot
    wait_start = time.perf_counter()
    async with _sem:
        queue_wait_s = time.perf_counter() - wait_start

        try:
            audio, synth_s = await asyncio.to_thread(synth_to_wav_bytes_timed, req.text, syn_config)
        except Exception as e:
            _metrics["errors_total"] += 1
            raise HTTPException(status_code=500, detail=str(e))

    total_s = time.perf_counter() - total_start

    # Update metrics
    _metrics["tts_synth_seconds_sum"] += synth_s
    _metrics["tts_total_seconds_sum"] += total_s
    _metrics["tts_queue_wait_seconds_sum"] += queue_wait_s
    _metrics["bytes_total"] += len(audio)

    # Put the timings in headers (clear units)
    headers = {
        "X-TTS-Total-ms": f"{total_s * 1000:.2f}",
        "X-TTS-QueueWait-ms": f"{queue_wait_s * 1000:.2f}",
        "X-TTS-Synth-ms": f"{synth_s * 1000:.2f}",
        "X-TTS-Bytes": str(len(audio)),
    }

    return Response(content=audio, media_type="audio/wav", headers=headers)
