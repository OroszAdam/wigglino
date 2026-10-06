# CPU image (x86_64 or arm64), port 7860, runs as uid 1000.
# GPU host: build with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
FROM python:3.12-slim

ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The HF cache dir exists in the image so a volume mounted there inherits the user's ownership.
RUN useradd --create-home --uid 1000 user \
 && mkdir -p /home/user/.cache/huggingface \
 && chown -R user:user /home/user/.cache
WORKDIR /app

COPY requirements.txt requirements-da3.txt ./
RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}" \
 && pip install -r requirements.txt \
 && pip install --no-deps -r requirements-da3.txt

COPY src ./src

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONPATH=/app/src \
    DATA_DIR=/tmp/wigglegram \
    PRELOAD_MODELS=mono,metric \
    PORT=7860

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn wigglegram.main:app --host 0.0.0.0 --port ${PORT} --no-server-header"]
