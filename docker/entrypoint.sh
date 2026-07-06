#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MODEL_ID="${MODEL_ID:-${OMNIVOICE_MODEL_ID:-k2-fsa/OmniVoice}}"
PRELOADED_MODEL="${OMNIVOICE_PRELOADED_MODEL:-/opt/omnivoice/models/k2-fsa/OmniVoice}"
HF_HOME="${HF_HOME:-/opt/omnivoice/hf}"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
AUDIO_SEPARATOR_MODEL_DIR="${AUDIO_SEPARATOR_MODEL_DIR:-${MODEL_DIR}/audio-separator}"

if [[ -d "${PRELOADED_MODEL}" ]]; then
  MODEL_ID="${PRELOADED_MODEL}"
fi

export HF_HOME MODEL_DIR AUDIO_SEPARATOR_MODEL_DIR PORT HOST
mkdir -p "${HF_HOME}" "${MODEL_DIR}" "${AUDIO_SEPARATOR_MODEL_DIR}" /app/work

exec python /app/api.py \
  --model "${MODEL_ID}" \
  --ip "${HOST}" \
  --port "${PORT}"
