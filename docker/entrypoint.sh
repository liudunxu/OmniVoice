#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MODEL_ID="${MODEL_ID:-${OMNIVOICE_MODEL_ID:-k2-fsa/OmniVoice}}"
PRELOADED_MODEL="${OMNIVOICE_PRELOADED_MODEL:-/opt/omnivoice/models/k2-fsa/OmniVoice}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
HF_HOME="${HF_HOME:-${MODEL_DIR}/huggingface}"
AUDIO_SEPARATOR_MODEL_DIR="${AUDIO_SEPARATOR_MODEL_DIR:-${MODEL_DIR}/audio-separator}"
WHISPER_MODEL_DIR="${WHISPER_MODEL_DIR:-${MODEL_DIR}/whisper}"

if [[ -d "${PRELOADED_MODEL}" ]]; then
  MODEL_ID="${PRELOADED_MODEL}"
fi

export HF_HOME MODEL_DIR AUDIO_SEPARATOR_MODEL_DIR WHISPER_MODEL_DIR PORT HOST
mkdir -p "${HF_HOME}" "${MODEL_DIR}" "${AUDIO_SEPARATOR_MODEL_DIR}" "${WHISPER_MODEL_DIR}" /app/work

NVIDIA_LIB_DIRS="$(
  find /app/.venv/lib -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd ':' -
)"
if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec python /app/api.py \
  --model "${MODEL_ID}" \
  --ip "${HOST}" \
  --port "${PORT}" \
  --asr-backend "${OMNIVOICE_ASR_BACKEND:-qwen3}"
