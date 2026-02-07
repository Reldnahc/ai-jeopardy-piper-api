import io
import os
import wave
import asyncio
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


def synth_to_wav_bytes(text: str, syn_config: SynthesisConfig | None) -> bytes:
    if voice is None:
        raise RuntimeError("voice not loaded")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    return buf.getvalue()


@app.post("/tts")
async def tts(req: TtsRequest):
    if voice is None:
        raise HTTPException(status_code=500, detail="Voice not loaded")

    syn_config = SynthesisConfig(
        volume=req.volume if req.volume is not None else DEFAULT_VOLUME,
        length_scale=req.length_scale if req.length_scale is not None else DEFAULT_LENGTH_SCALE,
        noise_scale=req.noise_scale if req.noise_scale is not None else DEFAULT_NOISE_SCALE,
        noise_w_scale=req.noise_w_scale if req.noise_w_scale is not None else DEFAULT_NOISE_W,
        normalize_audio=req.normalize_audio if req.normalize_audio is not None else True,
    )

    async with _sem:
        try:
            audio = await asyncio.to_thread(
                synth_to_wav_bytes,
                req.text,
                syn_config
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=audio,
        media_type="audio/wav",
    )
