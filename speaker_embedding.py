#!/usr/bin/env python3
"""Lazy WavLM speaker-embedding backend for speaker comparison.

Used by ``api._compare_speaker_waveforms`` as the default similarity backend
(``backend="wavlm_base_sv"``), replacing the MFCC heuristic. Nothing is
downloaded or loaded until the first embedding request — the default
checkpoint (~370MB) is pulled from the Hugging Face hub on first use.

Environment:
  SPEAKER_COMPARE_BACKEND   "wavlm" (default) or "mfcc" (disables this
                            backend entirely).
  SPEAKER_EMBED_MODEL       HF checkpoint, default "microsoft/wavlm-base-plus-sv".
  SPEAKER_EMBED_DEVICE      "cuda"/"cpu"; default cuda when available else cpu.
  SPEAKER_EMBED_MAX_SECONDS Center-crop longer clips (default 30).
  SPEAKER_EMBED_CACHE_SIZE  LRU entries keyed by sha256 of the 16kHz mono
                            float32 bytes (default 256; 0 disables caching).
"""

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 16000
_MIN_SECONDS = 0.5

_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_model = None
_feature_extractor = None
_uses_xvector = False
_load_failed = False
_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _backend_env() -> str:
    return _env("SPEAKER_COMPARE_BACKEND", "wavlm").lower()


def _device() -> str:
    override = os.environ.get("SPEAKER_EMBED_DEVICE")
    if override and override.strip():
        return override.strip()
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _max_seconds() -> float:
    try:
        return float(_env("SPEAKER_EMBED_MAX_SECONDS", "30"))
    except ValueError:
        return 30.0


def _cache_size() -> int:
    try:
        return max(0, int(_env("SPEAKER_EMBED_CACHE_SIZE", "256")))
    except ValueError:
        return 256


def backend_name() -> str:
    return "wavlm_base_sv"


def available() -> bool:
    """True when the backend is enabled and the model loaded successfully."""
    if _backend_env() == "mfcc":
        return False
    return _ensure_model()


def _ensure_model() -> bool:
    global _model, _feature_extractor, _uses_xvector, _load_failed
    if _model is not None:
        return True
    if _load_failed:
        return False
    with _load_lock:
        if _model is not None:
            return True
        if _load_failed:
            return False
        try:
            from transformers import AutoFeatureExtractor

            name = _env("SPEAKER_EMBED_MODEL", "microsoft/wavlm-base-plus-sv")
            device = _device()
            _feature_extractor = AutoFeatureExtractor.from_pretrained(name)
            try:
                # Preferred: x-vector head emits a speaker embedding directly.
                from transformers import WavLMForXVector

                model = WavLMForXVector.from_pretrained(name)
                _uses_xvector = True
            except (ImportError, AttributeError):
                # Fallback for transformers versions without WavLMForXVector:
                # base model + last_hidden_state mean pooling.
                from transformers import WavLMModel

                model = WavLMModel.from_pretrained(name)
                _uses_xvector = False
            model.eval()
            model.to(device)
            _model = model
            logger.info(
                "speaker embedding model loaded: %s (%s) on %s",
                name,
                "xvector" if _uses_xvector else "meanpool",
                device,
            )
            return True
        except Exception as exc:
            _load_failed = True
            logger.warning("speaker embedding model load failed: %s", exc)
            return False


def _prepare(waveform: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """Mono float32 at 16kHz, center-cropped to SPEAKER_EMBED_MAX_SECONDS."""
    import torch
    import torchaudio

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=0)
    arr = np.nan_to_num(arr.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    rate = int(sample_rate)
    if rate != _TARGET_SAMPLE_RATE and arr.size:
        arr = torchaudio.functional.resample(
            torch.from_numpy(arr), orig_freq=rate, new_freq=_TARGET_SAMPLE_RATE
        ).numpy()
    if arr.size < int(_MIN_SECONDS * _TARGET_SAMPLE_RATE):
        return None
    max_samples = int(_max_seconds() * _TARGET_SAMPLE_RATE)
    if 0 < max_samples < arr.size:
        start = (arr.size - max_samples) // 2
        arr = arr[start : start + max_samples]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _compute_embedding(arr: np.ndarray) -> np.ndarray:
    import torch

    device = _device()
    inputs = _feature_extractor(
        arr, sampling_rate=_TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        if _uses_xvector:
            emb = _model(**inputs).embeddings
        else:
            emb = _model(**inputs).last_hidden_state.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def embedding(waveform: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
    """L2-normalized speaker embedding, or None when unavailable/too short."""
    if not available():
        return None
    try:
        arr = _prepare(waveform, sample_rate)
    except Exception as exc:
        logger.warning("speaker embedding preprocess failed: %s", exc)
        return None
    if arr is None:
        return None
    key = hashlib.sha256(arr.tobytes()).hexdigest()
    with _infer_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached
        try:
            emb = _compute_embedding(arr)
        except Exception as exc:
            logger.warning("speaker embedding inference failed: %s", exc)
            return None
        size = _cache_size()
        if size > 0:
            _cache[key] = emb
            _cache.move_to_end(key)
            while len(_cache) > size:
                _cache.popitem(last=False)
        return emb


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0.0 when either vector is degenerate."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)
