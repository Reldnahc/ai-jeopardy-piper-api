FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
WORKDIR /app

# Python + basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Make "python" exist
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Better wheel/install behavior in CI
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install deps first (keeps download layer cacheable)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ---- Download voice at build time ----
ARG PIPER_VOICE=en_US-amy-low
RUN mkdir -p /models \
 && python -m piper.download_voices "${PIPER_VOICE}" --download-dir /models \
 && ls -lah /models

# App last (so code changes don't invalidate model download layer)
COPY app ./app

ENV PIPER_VOICE=${PIPER_VOICE}
ENV PIPER_MODEL=/models/${PIPER_VOICE}.onnx
ENV PIPER_USE_CUDA=1
ENV MAX_INFLIGHT_PER_WORKER=16

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
