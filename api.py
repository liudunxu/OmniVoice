#!/usr/bin/env python3
"""Standalone OmniVoice HTTP API server compatible with voxcpm_api.py.

Endpoints:
  POST /api/voxcpm/synthesize   Synthesize audio with OmniVoice
  GET  /api/health              Health check
  GET  /api/voxcpm/status       Model cache status
  POST /api/voxcpm/unload       Unload model from memory
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from aiohttp import web
from tqdm.auto import tqdm

# Fix torchaudio 2.11+ torchcodec fallback issues on machines without FFmpeg DLLs.
import torchaudio

_orig_torchaudio_load = torchaudio.load


def _patched_torchaudio_load(uri, *args, **kwargs):
    try:
        return _orig_torchaudio_load(uri, *args, **kwargs)
    except (ImportError, OSError, RuntimeError):
        data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
        if not kwargs.get("channels_first", True):
            waveform = waveform.T
        return waveform, sr


torchaudio.load = _patched_torchaudio_load

os.environ["TQDM_BAR_FORMAT"] = (
    "{desc:25} {percentage:3.0f}% "
    "|{bar:40}| "
    "{n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}]"
)

_tqdm_defaults = {
    "bar_format": os.environ["TQDM_BAR_FORMAT"],
    "ascii": "█▓▒░ ",
}
_original_tqdm_init = tqdm.__init__


def _patched_tqdm_init(self, *args, **kwargs):
    for key, value in _tqdm_defaults.items():
        kwargs.setdefault(key, value)
    _original_tqdm_init(self, *args, **kwargs)


tqdm.__init__ = _patched_tqdm_init

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.lang_map import LANG_NAME_TO_ID

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("omni_voice_api")

ROOT = Path(__file__).resolve().parent
WORK_ROOT = ROOT / "work"
OUTPUT_DIR = WORK_ROOT / "omni_voice_api_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TEXT_LEN = int(os.environ.get("VOXCPM_MAX_TEXT_LEN", "2000"))
MAX_REQUEST_MB = int(os.environ.get("VOXCPM_MAX_REQUEST_MB", os.environ.get("OMNIVOICE_MAX_REQUEST_MB", "64")))
MAX_REQUEST_SIZE = MAX_REQUEST_MB * 1024 * 1024
OMNIVOICE_SEED_MOD = 2**31 - 1

_API_MODEL = None
_API_MODEL_ID = "k2-fsa/OmniVoice"
_API_DEVICE = None
_API_LOAD_ASR = False
_MODEL_LOAD_LOCK = asyncio.Lock()
# Inference concurrency is gated by a semaphore (not a lock) so multi-GPU or
# high-VRAM GPUs can serve requests in parallel. Default 1 preserves the
# previous serialized behaviour. Set OMNIVOICE_MAX_CONCURRENCY > 1 to enable.
_API_INFER_SEM = asyncio.Semaphore(
    int(os.environ.get("OMNIVOICE_MAX_CONCURRENCY", "1"))
)

_DURATION_ESTIMATOR = RuleDurationEstimator()
_VOICE_PROMPT_CACHE: OrderedDict[str, Any] = OrderedDict()
_MAX_VOICE_PROMPT_CACHE_SIZE = int(
    os.environ.get("OMNIVOICE_VOICE_PROMPT_CACHE_SIZE", "100")
)
# Whether max_duration_ms should hard-reject requests whose natural duration
# exceeds the limit. Default warn-only to avoid breaking existing callers.
_ENFORCE_MAX_DURATION = str(
    os.environ.get("OMNIVOICE_ENFORCE_MAX_DURATION", "0")
).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Adaptive quality optimization: parameter profiles based on reference audio
# ---------------------------------------------------------------------------

# Quality profiles: optimized parameters for different reference audio lengths
# Short ref audio (<2s): need more steps and conservative guidance for stability
# Medium ref audio (2-4s): balanced parameters
# Longer ref audio (4-5s): can use slightly more aggressive guidance
_QUALITY_PROFILES = {
    "short": {  # ref audio < 2s
        "num_step": 48,
        "guidance_scale": 1.8,
        "t_shift": 0.05,
        "layer_penalty_factor": 4.0,
        "position_temperature": 3.0,
        "class_temperature": 0.0,
    },
    "medium": {  # ref audio 2-4s
        "num_step": 40,
        "guidance_scale": 2.0,
        "t_shift": 0.08,
        "layer_penalty_factor": 5.0,
        "position_temperature": 4.0,
        "class_temperature": 0.0,
    },
    "optimal": {  # ref audio 4-5s (sweet spot)
        "num_step": 36,
        "guidance_scale": 2.0,
        "t_shift": 0.1,
        "layer_penalty_factor": 5.0,
        "position_temperature": 5.0,
        "class_temperature": 0.0,
    },
    "long": {  # ref audio >5s
        "num_step": 32,
        "guidance_scale": 2.2,
        "t_shift": 0.1,
        "layer_penalty_factor": 5.0,
        "position_temperature": 5.0,
        "class_temperature": 0.0,
    },
}


def _get_ref_audio_duration(audio_path: str) -> Optional[float]:
    """Get duration of reference audio in seconds."""
    try:
        info = sf.info(str(audio_path))
        if info.samplerate > 0:
            return round(info.frames / info.samplerate, 3)
    except Exception:
        pass
    return None


def _bytes_audio_duration(audio_bytes: bytes) -> Optional[float]:
    """Read duration from raw audio bytes without touching disk."""
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        if info.samplerate > 0:
            return round(info.frames / info.samplerate, 3)
    except Exception:
        return None
    return None


def _select_quality_profile(ref_duration: Optional[float]) -> str:
    """Select quality profile based on reference audio duration."""
    if ref_duration is None:
        return "medium"
    if ref_duration < 2.0:
        return "short"
    elif ref_duration < 4.0:
        return "medium"
    elif ref_duration <= 5.0:
        return "optimal"
    else:
        return "long"


def _get_adaptive_params(
    ref_duration: Optional[float],
    user_cfg: Optional[float] = None,
    user_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Get adaptive parameters based on reference audio duration.

    User-specified values take precedence over adaptive defaults.
    """
    profile_name = _select_quality_profile(ref_duration)
    profile = _QUALITY_PROFILES[profile_name].copy()

    # User values override adaptive defaults
    if user_cfg is not None:
        profile["guidance_scale"] = user_cfg
    if user_steps is not None:
        profile["num_step"] = user_steps

    logger.info(
        f"Adaptive profile: {profile_name} (ref_duration={ref_duration}s), "
        f"num_step={profile['num_step']}, guidance_scale={profile['guidance_scale']}, "
        f"t_shift={profile['t_shift']}"
    )
    return profile


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _json_response(data, status=200):
    return web.json_response(data, status=status)


def _error(message, status=400):
    return _json_response({"ok": False, "error": message}, status=status)


def _audio_duration_seconds(path):
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return round(info.frames / info.samplerate, 3)
    except Exception:
        return None
    return None


def _decode_base64_audio_bytes(b64_data):
    """Decode base64 audio data to bytes. Supports data URI prefix."""
    b64_data = str(b64_data or "").strip()
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
    return base64.b64decode(b64_data)


def _write_base64_audio(b64_data, out_path):
    """Decode base64 audio data and write to file. Supports data URI prefix.

    Returns the written path.
    """
    audio_bytes = _decode_base64_audio_bytes(b64_data)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    return out_path


def _make_voice_prompt_cache_key(audio_bytes: bytes, prompt_text: str, preprocess_prompt: bool) -> str:
    """Hash reference audio bytes + prompt metadata for prompt caching."""
    payload = {
        "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "prompt_text": prompt_text,
        "preprocess_prompt": preprocess_prompt,
        "model_id": _API_MODEL_ID,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _get_cached_voice_clone_prompt(
    model,
    audio_path: Optional[str],
    audio_bytes: Optional[bytes],
    audio_wav: Optional[Tuple[np.ndarray, int]],
    prompt_text: str,
    preprocess_prompt: bool,
):
    """Create a voice-clone prompt, caching by audio content hash.

    Either ``audio_path`` (on-disk file) or ``audio_wav`` ((waveform, sr)
    tuple, e.g. decoded from ``audio_bytes``) must be supplied; the latter
    avoids disk I/O when the cache will hit.
    """
    if audio_bytes is None and audio_path is None:
        raise ValueError("audio_bytes or audio_path required for voice clone prompt")
    if audio_bytes is None:
        audio_bytes = Path(audio_path).read_bytes()

    cache_key = _make_voice_prompt_cache_key(audio_bytes, prompt_text, preprocess_prompt)
    cached = _VOICE_PROMPT_CACHE.get(cache_key)
    if cached is not None:
        logger.debug("Voice clone prompt cache hit: %s", cache_key[:16])
        return cached

    ref_input = audio_wav if audio_wav is not None else audio_path
    prompt = _create_voice_clone_prompt(
        model,
        ref_input,
        prompt_audio=None,
        prompt_text=prompt_text,
        preprocess_prompt=preprocess_prompt,
    )

    # Simple LRU: pop oldest if cache is full.
    if len(_VOICE_PROMPT_CACHE) >= _MAX_VOICE_PROMPT_CACHE_SIZE:
        _VOICE_PROMPT_CACHE.popitem(last=False)
    _VOICE_PROMPT_CACHE[cache_key] = prompt
    return prompt


def _estimate_natural_duration(
    text: str,
    ref_text: Optional[str],
    ref_duration: Optional[float],
) -> float:
    """Estimate natural duration for the target text using the rule estimator.

    Falls back to a default speaking rate when the provided reference text/audio
    imply an unrealistic speed (e.g. prompt text much shorter than the reference
    audio), which would otherwise wildly over/under-estimate the duration.
    """
    if ref_duration and ref_text and ref_duration > 0:
        ref_weight = _DURATION_ESTIMATOR.calculate_total_weight(ref_text)
        if ref_weight > 0:
            speed = ref_weight / ref_duration
            # Normal speech roughly spans 1-50 weighted chars/sec. Outside this
            # range we treat the reference as inconsistent and use a default rate.
            if 1.0 <= speed <= 50.0:
                return _DURATION_ESTIMATOR.estimate_duration(
                    text, ref_text, ref_duration, low_threshold=2.0
                )
    # Fallback: assume a neutral reference when no prompt text/audio is available
    # or when the reference is inconsistent.
    return _DURATION_ESTIMATOR.estimate_duration(
        text, "Nice to meet you.", 1.5, low_threshold=2.0
    )


def _waveform_to_wav_bytes(waveform, sampling_rate: int) -> bytes:
    """Encode a waveform to WAV bytes without touching disk."""
    arr = waveform
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.squeeze(np.asarray(arr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, arr, int(sampling_rate), subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path))


def _set_api_model(model, model_id, device, load_asr):
    global _API_MODEL, _API_MODEL_ID, _API_DEVICE, _API_LOAD_ASR
    _API_MODEL = model
    _API_MODEL_ID = model_id
    _API_DEVICE = device
    _API_LOAD_ASR = load_asr


def _load_api_model_sync():
    logger.info(f"加载模型: {_API_MODEL_ID}, 设备: {_API_DEVICE} ...")
    model = OmniVoice.from_pretrained(
        _API_MODEL_ID,
        device_map=_API_DEVICE,
        dtype=torch.float16,
        load_asr=_API_LOAD_ASR,
    )
    logger.info("模型加载完成！")
    return model


async def _ensure_api_model():
    global _API_MODEL
    if _API_MODEL is not None:
        return _API_MODEL
    async with _MODEL_LOAD_LOCK:
        # Double-check after acquiring lock.
        if _API_MODEL is not None:
            return _API_MODEL
        _API_MODEL = await asyncio.to_thread(_load_api_model_sync)
    return _API_MODEL


def _bool_option(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_seed(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value)) % OMNIVOICE_SEED_MOD
    except (TypeError, ValueError):
        return None


def _sha256_text(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _stable_seed_from_request(data, text, effective_prompt_text, reference_audio_base64, prompt_wav_base64):
    explicit = _normalize_seed(data.get("seed") or data.get("omnivoice_seed"))
    if explicit is not None:
        return explicit
    if str(os.environ.get("OMNIVOICE_DETERMINISTIC", "1")).lower() in {"0", "false", "no", "off"}:
        return None
    payload = {
        "text": text,
        "prompt_text": effective_prompt_text,
        "reference_audio_sha256": _sha256_text(reference_audio_base64),
        "prompt_audio_sha256": _sha256_text(prompt_wav_base64),
        "model_id": data.get("model_id") or _API_MODEL_ID,
        "cfg_value": data.get("cfg_value", 2.0),
        "inference_timesteps": data.get("inference_timesteps", 32),
        "denoise": _bool_option(data.get("denoise"), True),
        "speed": data.get("speed", 1.0),
        "duration": data.get("duration"),
        "language": data.get("language")
        or data.get("target_lang")
        or data.get("target_language")
        or data.get("output_language_code"),
        "instruct": data.get("instruct"),
        "preprocess_prompt": _bool_option(data.get("preprocess_prompt"), True),
        "postprocess_output": _bool_option(data.get("postprocess_output"), True),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16) % OMNIVOICE_SEED_MOD


def _apply_seed(seed):
    seed = _normalize_seed(seed)
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    return seed


_LANG_CODE_ALIASES = {
    "tl": "fil",
    "filipino": "fil",
}


def _resolve_language(value):
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"auto", "自动检测"}:
        return None
    if raw in _LANG_CODE_ALIASES:
        return _LANG_CODE_ALIASES[raw]
    if raw in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[raw]
    lower_map = {name.lower(): code for name, code in LANG_NAME_TO_ID.items()}
    resolved = lower_map.get(raw.lower())
    if resolved:
        return resolved
    if raw.lower() in _LANG_CODE_ALIASES:
        return _LANG_CODE_ALIASES[raw.lower()]
    return raw


def _create_voice_clone_prompt(
    model,
    reference_audio,
    prompt_audio=None,
    prompt_text="",
    preprocess_prompt=True,
):
    ref_text_clean = prompt_text.strip() if prompt_text else None
    if prompt_audio and prompt_audio != reference_audio:
        candidates = [
            {"ref_audio": reference_audio, "prompt_audio": prompt_audio, "ref_text": ref_text_clean},
            {"ref_audio": reference_audio, "prompt_wav": prompt_audio, "ref_text": ref_text_clean},
            {"ref_audio": reference_audio, "prompt_wav_path": prompt_audio, "ref_text": ref_text_clean},
        ]
        for kwargs in candidates:
            try:
                return model.create_voice_clone_prompt(
                    **kwargs,
                    preprocess_prompt=preprocess_prompt,
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
    return model.create_voice_clone_prompt(
        ref_audio=reference_audio,
        ref_text=ref_text_clean,
        preprocess_prompt=preprocess_prompt,
    )


def _is_empty_reference_after_preprocess(exc):
    return "Reference audio is empty after silence removal" in str(exc)


def _cleanup_temp_paths(*paths):
    for path in paths:
        if path is not None:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


def _make_generation_config(
    cfg_value=2.0,
    inference_timesteps=32,
    denoise=True,
    preprocess_prompt=True,
    postprocess_output=True,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
    audio_chunk_duration=15.0,
    audio_chunk_threshold=30.0,
):
    return OmniVoiceGenerationConfig(
        num_step=int(inference_timesteps),
        guidance_scale=float(cfg_value),
        t_shift=float(t_shift),
        layer_penalty_factor=float(layer_penalty_factor),
        position_temperature=float(position_temperature),
        class_temperature=float(class_temperature),
        denoise=bool(denoise),
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
        audio_chunk_duration=float(audio_chunk_duration),
        audio_chunk_threshold=float(audio_chunk_threshold),
    )


def _generate_omnivoice_audio(
    model,
    text,
    voice_clone_prompt=None,
    cfg_value=2.0,
    inference_timesteps=32,
    denoise=True,
    speed=1.0,
    duration=None,
    language=None,
    instruct=None,
    preprocess_prompt=True,
    postprocess_output=True,
    seed=None,
    t_shift=0.1,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
    audio_chunk_duration=15.0,
    audio_chunk_threshold=30.0,
):
    """Run model.generate and return the first audio waveform (np.ndarray)."""
    gen_config = _make_generation_config(
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        denoise=denoise,
        preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
        t_shift=t_shift,
        layer_penalty_factor=layer_penalty_factor,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
        audio_chunk_duration=audio_chunk_duration,
        audio_chunk_threshold=audio_chunk_threshold,
    )
    kw: Dict[str, Any] = {
        "text": text.strip(),
        "language": _resolve_language(language),
        "generation_config": gen_config,
    }
    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)
    if voice_clone_prompt is not None:
        kw["voice_clone_prompt"] = voice_clone_prompt
    if instruct and str(instruct).strip() and str(instruct).strip() != "None":
        kw["instruct"] = str(instruct).strip()

    def generate_with_seed():
        _apply_seed(seed)
        return model.generate(**kw)

    try:
        audio = generate_with_seed()
    except ValueError as exc:
        if not preprocess_prompt or not _is_empty_reference_after_preprocess(exc):
            raise
        logger.warning(
            "Reference audio became empty after OmniVoice silence removal; retrying with preprocess_prompt=False"
        )
        kw["generation_config"] = _make_generation_config(
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            denoise=denoise,
            preprocess_prompt=False,
            postprocess_output=postprocess_output,
            t_shift=t_shift,
            layer_penalty_factor=layer_penalty_factor,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
            audio_chunk_duration=audio_chunk_duration,
            audio_chunk_threshold=audio_chunk_threshold,
        )
        audio = generate_with_seed()

    # model.generate returns list[np.ndarray]; take the first (and only) item.
    return audio[0]


def _generate_with_duration_refinement(
    model,
    text,
    target_duration,
    duration_tolerance,
    max_attempts,
    voice_clone_prompt=None,
    max_duration=None,
    ratio_clamp=(0.5, 2.0),
    **gen_kwargs,
):
    """Generate audio, optionally retrying until duration is within tolerance.

    Returns:
        (audio_waveform, attempts_made, attempt_log)
    """
    if target_duration is None or target_duration <= 0:
        audio = _generate_omnivoice_audio(
            model, text, voice_clone_prompt=voice_clone_prompt, **gen_kwargs
        )
        return audio, 1, []

    current_duration = float(target_duration)
    best_audio = None
    best_error = float("inf")
    attempt_log = []

    for attempt in range(max_attempts):
        kwargs = dict(gen_kwargs)
        kwargs["duration"] = current_duration
        audio = _generate_omnivoice_audio(
            model, text, voice_clone_prompt=voice_clone_prompt, **kwargs
        )
        actual_duration = audio.shape[-1] / model.sampling_rate
        error = abs(actual_duration - current_duration)
        attempt_log.append(
            {
                "attempt": attempt + 1,
                "target_duration": current_duration,
                "actual_duration": actual_duration,
                "error": error,
            }
        )

        if duration_tolerance is None or error <= duration_tolerance:
            return audio, attempt + 1, attempt_log

        if error < best_error:
            best_error = error
            best_audio = audio

        if attempt < max_attempts - 1 and actual_duration > 0:
            # Scale target duration by the observed ratio, clamped to avoid
            # divergence when the model output is wildly off (e.g. actual=0.1s
            # for a 10s target would otherwise try 100s next).
            raw_ratio = current_duration / actual_duration
            clamped_ratio = max(ratio_clamp[0], min(ratio_clamp[1], raw_ratio))
            next_duration = current_duration * clamped_ratio
            if max_duration is not None and next_duration > max_duration:
                logger.warning(
                    "Duration refinement ratio %.3f would push target to %.3fs, "
                    "exceeding max_duration=%.3fs; clamping to max_duration.",
                    raw_ratio,
                    next_duration,
                    max_duration,
                )
                next_duration = float(max_duration)
            logger.info(
                "Duration refinement attempt %d: target=%.3fs actual=%.3fs "
                "ratio=%.3f (clamped=%.3f); retrying with duration=%.3fs",
                attempt + 1,
                current_duration,
                actual_duration,
                raw_ratio,
                clamped_ratio,
                next_duration,
            )
            current_duration = next_duration

    logger.warning(
        "Duration refinement did not converge within tolerance %.3fs after %d attempts; "
        "returning closest result (error=%.3fs)",
        duration_tolerance if duration_tolerance is not None else 0.0,
        max_attempts,
        best_error,
    )
    return best_audio, max_attempts, attempt_log


def _audio_duration(waveform, sampling_rate: int) -> float:
    """Return audio duration in seconds from a 1-D waveform."""
    arr = np.asarray(waveform)
    return float(arr.shape[-1]) / sampling_rate


def _measure_silence_ratio(waveform, threshold: float = 0.01) -> float:
    """Return ratio of samples whose absolute amplitude is below threshold."""
    arr = np.asarray(waveform)
    if arr.size == 0:
        return 1.0
    return float(np.mean(np.abs(arr) < threshold))


def _compute_rms(waveform) -> float:
    arr = np.asarray(waveform)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _check_audio_quality(
    waveform,
    sampling_rate: int,
    target_duration: Optional[float] = None,
    duration_tolerance: Optional[float] = None,
    ref_duration: Optional[float] = None,
) -> list[str]:
    """Check generated audio for common badcase patterns.

    Returns a list of issue labels; empty list means no detected issue.
    """
    issues = []
    arr = np.asarray(waveform)
    duration = _audio_duration(arr, sampling_rate)
    peak = float(np.abs(arr).max()) if arr.size > 0 else 0.0
    rms = _compute_rms(arr)
    silence_ratio = _measure_silence_ratio(arr)

    if arr.size == 0 or duration < 0.05:
        issues.append("empty")
    if silence_ratio > 0.5:
        issues.append("too_much_silence")
    if peak > 0.99:
        issues.append("clipping")
    if 0 < rms < 0.005:
        issues.append("too_quiet")

    if target_duration is not None and target_duration > 0:
        tol = duration_tolerance if duration_tolerance is not None else 0.0
        # Flag if deviation is more than 2x tolerance or > 0.5s, whichever is larger.
        if abs(duration - target_duration) > max(tol * 2, 0.5):
            issues.append("duration_off_target")

    if ref_duration is not None and ref_duration > 0:
        ratio = duration / ref_duration
        if ratio > 3.0 or ratio < 0.33:
            issues.append("duration_off_reference")

    return issues


def _apply_fallback_params(gen_kwargs: Dict[str, Any], issues: list[str]) -> Dict[str, Any]:
    """Build a more conservative generation config for badcase retry."""
    fallback = dict(gen_kwargs)

    # More decoding steps + slightly stronger guidance for stability.
    fallback["inference_timesteps"] = min(
        int(fallback.get("inference_timesteps", 32) * 1.5), 64
    )
    fallback["cfg_value"] = min(
        float(fallback.get("cfg_value", 2.0)) + 0.2, 3.0
    )

    if "too_much_silence" in issues or "empty" in issues:
        # Tighter position sampling to reduce random unmasking of silences.
        fallback["position_temperature"] = max(
            float(fallback.get("position_temperature", 5.0)) * 0.6, 1.0
        )

    if "clipping" in issues:
        # Disable post-processing in case aggressive trimming/leveling caused clipping.
        fallback["postprocess_output"] = False

    if "too_quiet" in issues:
        # Keep post-processing so RMS normalization can boost quiet output.
        fallback["postprocess_output"] = True

    if "duration_off_target" in issues or "duration_off_reference" in issues:
        # Reduce t_shift to emphasize earlier (lower-SNR) steps, often yields
        # more stable timing.
        fallback["t_shift"] = max(float(fallback.get("t_shift", 0.1)) * 0.7, 0.03)

    return fallback


def _duration_error(waveform, sampling_rate: int, target_duration: Optional[float]) -> float:
    if target_duration is None or target_duration <= 0:
        return 0.0
    return abs(_audio_duration(waveform, sampling_rate) - target_duration)


def _generate_with_quality_retry(
    model,
    text,
    target_duration,
    duration_tolerance,
    voice_clone_prompt=None,
    ref_duration=None,
    max_duration=None,
    enable_quality_retry=True,
    **gen_kwargs,
):
    """Generate audio with duration refinement and one badcase retry.

    Returns:
        (audio_waveform, duration_attempts, duration_log, quality_issues, quality_retried)
    """
    audio, attempts, log = _generate_with_duration_refinement(
        model,
        text,
        target_duration=target_duration,
        duration_tolerance=duration_tolerance,
        max_attempts=2,
        voice_clone_prompt=voice_clone_prompt,
        max_duration=max_duration,
        **gen_kwargs,
    )
    issues = _check_audio_quality(
        audio,
        model.sampling_rate,
        target_duration=target_duration,
        duration_tolerance=duration_tolerance,
        ref_duration=ref_duration,
    )

    if not issues or not enable_quality_retry:
        return audio, attempts, log, issues, False

    logger.info(
        "Quality issues detected on first attempt: %s; retrying with fallback params",
        issues,
    )
    fallback_kwargs = _apply_fallback_params(gen_kwargs, issues)
    # For retry, use the original target without refinement to keep latency bounded.
    audio2, attempts2, log2 = _generate_with_duration_refinement(
        model,
        text,
        target_duration=target_duration,
        duration_tolerance=None,
        max_attempts=1,
        voice_clone_prompt=voice_clone_prompt,
        **fallback_kwargs,
    )
    issues2 = _check_audio_quality(
        audio2,
        model.sampling_rate,
        target_duration=target_duration,
        duration_tolerance=duration_tolerance,
        ref_duration=ref_duration,
    )

    # Choose the result with fewer issues; tie-break by duration closeness.
    if len(issues2) < len(issues) or (
        len(issues2) == len(issues)
        and _duration_error(audio2, model.sampling_rate, target_duration)
        < _duration_error(audio, model.sampling_rate, target_duration)
    ):
        return audio2, attempts + attempts2, log + log2, issues2, True

    return audio, attempts, log, issues, False


routes = web.RouteTableDef()


@routes.get("/api/health")
async def health(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    return _json_response({"ok": True, "service": "voxcpm2_api"})


@routes.get("/api/voxcpm/status")
@routes.get("/api/status")
async def status(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    cached_models = []
    if _API_MODEL is not None:
        cached_models.append({
            "model_id": _API_MODEL_ID,
            "device": _API_DEVICE,
            "load_denoiser": False,
            "optimize": False,
        })
    return _json_response({
        "ok": True,
        "models_cached": len(cached_models),
        "cached_models": cached_models,
    })


@routes.post("/api/voxcpm/unload")
@routes.post("/api/unload")
async def unload(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    global _API_MODEL
    count = 1 if _API_MODEL is not None else 0
    _API_MODEL = None
    _VOICE_PROMPT_CACHE.clear()
    import gc
    gc.collect()
    if sys.platform != "win32":
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return _json_response({"ok": True, "unloaded": count})


@routes.post("/api/voxcpm/synthesize")
@routes.post("/api/synthesize")
async def synthesize(request):
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")

    try:
        data = await request.json()
    except Exception as exc:
        tb = traceback.format_exc()
        logger.warning(f"[{req_id}] Failed to parse JSON: {exc}\n{tb}")
        return _error(f"Invalid JSON body: {exc}\n{tb}", status=400)

    text = re.sub(r"\s+", " ", (data.get("text") or "").strip())
    if not text:
        logger.warning(f"[{req_id}] Missing text parameter")
        return _error("text is required and cannot be empty.")
    if len(text) > MAX_TEXT_LEN:
        logger.warning(f"[{req_id}] Text too long: {len(text)} > {MAX_TEXT_LEN}")
        return _error(f"text exceeds max length {MAX_TEXT_LEN}.")

    reference_audio_base64 = data.get("reference_audio_base64")
    prompt_wav_base64 = data.get("prompt_wav_base64") or data.get("prompt_audio_base64") or data.get("prompt_wav")
    prompt_text = re.sub(r"\s+", " ", (data.get("prompt_text") or "").strip())
    effective_prompt_text = prompt_text if prompt_wav_base64 else ""

    # Get user-specified values (None means use adaptive defaults)
    user_cfg = data.get("cfg_value")
    user_steps = data.get("inference_timesteps")
    denoise = _bool_option(data.get("denoise"), True)
    optimize = _bool_option(data.get("optimize"), False)
    target_duration_ms = data.get("target_duration_ms")
    max_duration_ms = data.get("max_duration_ms")
    duration_tolerance_ms = data.get("duration_tolerance_ms")
    user_duration = data.get("duration")
    user_speed = float(data.get("speed", 1.0))
    seed = _stable_seed_from_request(data, text, effective_prompt_text, reference_audio_base64, prompt_wav_base64)

    # Parse duration control parameters.
    target_duration_sec = (
        float(target_duration_ms) / 1000.0 if target_duration_ms is not None else None
    )
    max_duration_sec = (
        float(max_duration_ms) / 1000.0 if max_duration_ms is not None else None
    )
    duration_tolerance_sec = (
        float(duration_tolerance_ms) / 1000.0
        if duration_tolerance_ms is not None
        else None
    )

    # Prepare output directory
    out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decode reference audio to measure duration for adaptive params.
    # We deliberately skip writing a temp file here — the voice-clone prompt
    # cache will be probed inside the inference lock and only on cache miss
    # do we need to materialize a file (or pass an in-memory tuple directly).
    ref_temp_path = None
    prompt_temp_path = None
    resolved_ref = None
    resolved_prompt = None
    ref_duration = None
    ref_audio_bytes = None
    prompt_audio_bytes = None

    if reference_audio_base64:
        try:
            ref_audio_bytes = _decode_base64_audio_bytes(reference_audio_base64)
            ref_duration = _bytes_audio_duration(ref_audio_bytes)
            logger.info(
                f"[{req_id}] reference audio decoded in-memory: {len(ref_audio_bytes)} bytes, "
                f"duration={ref_duration}s"
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[{req_id}] Failed to decode reference_audio_base64: {exc}\n{tb}")
            return _error(f"Failed to decode reference_audio_base64: {exc}\n{tb}")

    if prompt_wav_base64:
        try:
            prompt_audio_bytes = _decode_base64_audio_bytes(prompt_wav_base64)
            logger.info(
                f"[{req_id}] prompt wav decoded in-memory: {len(prompt_audio_bytes)} bytes"
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[{req_id}] Failed to decode prompt_wav_base64: {exc}\n{tb}")
            _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
            return _error(f"Failed to decode prompt_wav_base64: {exc}\n{tb}")

    # Get adaptive parameters based on reference audio duration
    adaptive_params = _get_adaptive_params(
        ref_duration=ref_duration,
        user_cfg=float(user_cfg) if user_cfg is not None else None,
        user_steps=int(user_steps) if user_steps is not None else None,
    )

    cfg_value = adaptive_params["guidance_scale"]
    inference_timesteps = adaptive_params["num_step"]
    t_shift = adaptive_params["t_shift"]
    layer_penalty_factor = adaptive_params["layer_penalty_factor"]
    position_temperature = adaptive_params["position_temperature"]
    class_temperature = adaptive_params["class_temperature"]

    # Estimate natural duration for the target text (used for badcase avoidance
    # and max-duration validation).
    estimated_natural_duration = _estimate_natural_duration(
        text,
        ref_text=effective_prompt_text if effective_prompt_text else None,
        ref_duration=ref_duration,
    )

    # Validate against max_duration_ms early.
    if max_duration_sec is not None and estimated_natural_duration > max_duration_sec:
        msg = (
            f"Estimated natural duration ({estimated_natural_duration:.2f}s) exceeds "
            f"max_duration_ms ({max_duration_sec:.2f}s)."
        )
        if _ENFORCE_MAX_DURATION:
            logger.warning(f"[{req_id}] {msg}")
            _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
            return _error(
                f"{msg} Try shorter text or increase max_duration_ms.", status=400
            )
        logger.warning(
            f"[{req_id}] {msg} Allowing request because "
            f"OMNIVOICE_ENFORCE_MAX_DURATION is not set."
        )

    # Adaptive duration: try to match ref audio length when user doesn't specify duration
    # Strategy: use ref audio duration as target, but avoid badcases by checking text length
    effective_duration = user_duration
    effective_speed = user_speed
    if user_duration is None and target_duration_sec is not None:
        # target_duration_ms takes precedence over ref-duration matching.
        effective_duration = target_duration_sec
        logger.info(
            f"[{req_id}] Using target_duration_ms={target_duration_ms} "
            f"as target duration={effective_duration:.3f}s"
        )
    elif user_duration is None and ref_duration is not None and user_speed == 1.0:
        # If ref_duration can accommodate the text without rushing (>=70% of natural duration)
        if ref_duration >= estimated_natural_duration * 0.7:
            effective_duration = ref_duration
            logger.info(f"[{req_id}] Using ref_duration={ref_duration}s as target (text_len={len(text)}, est_natural={estimated_natural_duration:.1f}s)")
        else:
            # Text is too long for ref_duration, use natural estimation to avoid badcase
            logger.info(f"[{req_id}] Skipping ref_duration={ref_duration}s (text too long, est_natural={estimated_natural_duration:.1f}s), using model estimation")
    elif user_duration is None and ref_duration is not None and user_speed != 1.0:
        # User specified speed, respect it but log for debugging
        logger.info(f"[{req_id}] User specified speed={user_speed}, skipping ref_duration matching")

    # Final validation: requested/fallback duration must not exceed max_duration_ms.
    if (
        max_duration_sec is not None
        and effective_duration is not None
        and effective_duration > max_duration_sec
    ):
        logger.warning(
            f"[{req_id}] Requested target duration {effective_duration:.2f}s "
            f"exceeds max_duration_ms={max_duration_ms}"
        )
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(
            f"Target duration ({effective_duration:.2f}s) exceeds max_duration_ms ({max_duration_sec:.2f}s).",
            status=400,
        )

    logger.info(
        f"[{req_id}] params: text_len={len(text)}, has_ref={bool(reference_audio_base64)}, "
        f"ref_duration={ref_duration}s, has_prompt_wav={bool(prompt_wav_base64)}, "
        f"prompt_len={len(effective_prompt_text)}, requested_model={data.get('model_id') or ''}, "
        f"loaded_model={_API_MODEL_ID}, device={_API_DEVICE}, cfg={cfg_value}, "
        f"steps={inference_timesteps}, t_shift={t_shift}, denoise={denoise}, "
        f"layer_penalty={layer_penalty_factor}, pos_temp={position_temperature}, "
        f"class_temp={class_temperature}, duration={effective_duration}, speed={effective_speed}, "
        f"target_ms={target_duration_ms}, max_ms={max_duration_ms}, "
        f"tolerance_ms={duration_tolerance_ms}, seed={seed if seed is not None else '-'}"
    )

    out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = data.get("output_name")
    if not out_name:
        key = hashlib.sha256(
            json.dumps({
                "text": text,
                "prompt": effective_prompt_text,
                "ref_b64_len": len(reference_audio_base64) if reference_audio_base64 else 0,
                "prompt_b64_len": len(prompt_wav_base64) if prompt_wav_base64 else 0,
                "model": _API_MODEL_ID,
                "device": _API_DEVICE,
                "cfg": cfg_value,
                "steps": inference_timesteps,
                "t_shift": t_shift,
                "denoise": denoise,
                "optimize": optimize,
                "target_duration_ms": target_duration_ms,
                "max_duration_ms": max_duration_ms,
                "duration_tolerance_ms": duration_tolerance_ms,
                "seed": seed,
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        out_name = f"voxcpm_{key}.wav"
    out_path = out_dir / out_name

    # Load model (loading is serialized via _MODEL_LOAD_LOCK).
    model = await _ensure_api_model()

    preprocess_prompt = _bool_option(data.get("preprocess_prompt"), True)
    postprocess_output = _bool_option(data.get("postprocess_output"), True)

    if (
        target_duration_sec is not None
        and postprocess_output
        and (duration_tolerance_sec is None or duration_tolerance_sec < 0.05)
    ):
        logger.warning(
            f"[{req_id}] target_duration_ms requested with tight tolerance and "
            f"postprocess_output=True; post-processing may trim trailing silence "
            f"and break exact duration matching."
        )

    language = (
        data.get("language")
        or data.get("target_lang")
        or data.get("target_language")
        or data.get("output_language_code")
    )
    instruct = data.get("instruct")

    gen_kwargs = {
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "denoise": denoise,
        "speed": effective_speed,
        "duration": None,  # overridden by refinement loop
        "language": language,
        "instruct": instruct,
        "preprocess_prompt": preprocess_prompt,
        "postprocess_output": postprocess_output,
        "seed": seed,
        "t_shift": t_shift,
        "layer_penalty_factor": layer_penalty_factor,
        "position_temperature": position_temperature,
        "class_temperature": class_temperature,
        "audio_chunk_duration": float(data.get("audio_chunk_duration", 15.0)),
        "audio_chunk_threshold": float(data.get("audio_chunk_threshold", 30.0)),
    }

    # Cap concurrent inference to keep deterministic seeding safe and avoid
    # ASR/tokenizer races. Audio I/O before/after this block can still overlap
    # across requests. Semaphore (not Lock) so OMNIVOICE_MAX_CONCURRENCY > 1
    # can serve requests in parallel on high-VRAM GPUs.
    start_time = time.time()
    try:
        async with _API_INFER_SEM:
            logger.info(f"[{req_id}] synthesis started -> {out_path}")

            # Build voice-clone prompt (cached) inside the lock to avoid concurrent
            # ASR/tokenizer access. When the cache hits, we pass an in-memory
            # (waveform, sr) tuple directly — no temp file I/O needed.
            voice_clone_prompt = None
            if reference_audio_base64 or prompt_wav_base64:
                clone_audio_bytes = ref_audio_bytes or prompt_audio_bytes
                prompt_for_clone = effective_prompt_text if prompt_wav_base64 else ""
                if clone_audio_bytes:
                    cache_key = _make_voice_prompt_cache_key(
                        clone_audio_bytes, prompt_for_clone, preprocess_prompt
                    )
                    pre_have = cache_key in _VOICE_PROMPT_CACHE
                    if pre_have:
                        ref_path_for_prompt = None
                    else:
                        ref_temp_path = out_dir / f"ref_{uuid.uuid4().hex}.wav"
                        ref_temp_path.parent.mkdir(parents=True, exist_ok=True)
                        ref_temp_path.write_bytes(clone_audio_bytes)
                        ref_path_for_prompt = str(ref_temp_path)

                    try:
                        voice_clone_prompt = _get_cached_voice_clone_prompt(
                            model,
                            audio_path=ref_path_for_prompt,
                            audio_bytes=clone_audio_bytes,
                            audio_wav=None,
                            prompt_text=prompt_for_clone,
                            preprocess_prompt=preprocess_prompt,
                        )
                    except ValueError as exc:
                        if not preprocess_prompt or not _is_empty_reference_after_preprocess(exc):
                            raise
                        logger.warning(
                            f"[{req_id}] Reference audio became empty after silence removal; "
                            f"retrying with preprocess_prompt=False"
                        )
                        preprocess_prompt = False
                        gen_kwargs["preprocess_prompt"] = False
                        voice_clone_prompt = _get_cached_voice_clone_prompt(
                            model,
                            audio_path=ref_path_for_prompt,
                            audio_bytes=clone_audio_bytes,
                            audio_wav=None,
                            prompt_text=prompt_for_clone,
                            preprocess_prompt=False,
                        )

            (
                audio_waveform,
                attempts_made,
                attempt_log,
                quality_issues,
                quality_retried,
            ) = await asyncio.to_thread(
                _generate_with_quality_retry,
                model,
                text,
                target_duration=effective_duration,
                duration_tolerance=duration_tolerance_sec,
                voice_clone_prompt=voice_clone_prompt,
                ref_duration=ref_duration,
                max_duration=max_duration_sec,
                enable_quality_retry=_bool_option(
                    data.get("quality_retry"), True
                ),
                **gen_kwargs,
            )
            if "empty" in quality_issues:
                raise RuntimeError(
                    "Generated audio is empty after quality retry. "
                    "Use a longer, non-silent reference audio or disable voice cloning."
                )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Synthesis failed: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Synthesis failed: {exc}\n{tb}", status=502)

    try:
        wav_bytes = _waveform_to_wav_bytes(audio_waveform, model.sampling_rate)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Failed to encode output audio: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Failed to encode output audio: {exc}\n{tb}", status=502)

    output_path_for_response = str(out_path.resolve())
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.warning(f"[{req_id}] Failed to persist output file (continuing): {exc}")
        output_path_for_response = ""

    elapsed = round(time.time() - start_time, 3)
    audio_duration = round(audio_waveform.shape[-1] / model.sampling_rate, 3)
    logger.info(
        f"[{req_id}] synthesis finished in {elapsed}s, output_size={len(wav_bytes)} bytes, "
        f"audio_duration={audio_duration}s, duration_attempts={attempts_made}, "
        f"quality_issues={quality_issues}, quality_retried={quality_retried}"
    )

    output_base64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")

    _cleanup_temp_paths(ref_temp_path, prompt_temp_path)

    logger.info(f"[{req_id}] response sent, audio_base64_len={len(output_base64)}")
    return _json_response({
        "ok": True,
        "audio_base64": output_base64,
        "output_path": output_path_for_response,
        "relative_path": _relative_path(Path(output_path_for_response)) if output_path_for_response else "",
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": audio_duration,
        "target_duration_ms": target_duration_ms,
        "max_duration_ms": max_duration_ms,
        "duration_tolerance_ms": duration_tolerance_ms,
        "seed": seed,
        "duration_attempts": attempts_made,
        "duration_refinement_log": attempt_log,
        "quality_issues": quality_issues,
        "quality_retried": quality_retried,
        "duration_match": {
            "ref_duration": ref_duration,
            "target_duration": effective_duration,
            "actual_duration": audio_duration,
            "match_ratio": round(audio_duration / ref_duration, 3) if ref_duration and audio_duration else None,
        },
        "adaptive_params": {
            "ref_duration": ref_duration,
            "profile": _select_quality_profile(ref_duration),
            "num_step": inference_timesteps,
            "guidance_scale": cfg_value,
            "t_shift": t_shift,
        },
    })


@routes.get("/")
async def index(request):
    return web.Response(
        content_type="text/html",
        text="""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>OmniVoice API</title></head>
<body>
  <h1>OmniVoice API Server</h1>
  <pre>
GET  /api/health
GET  /api/voxcpm/status  (alias: /api/status)
POST /api/voxcpm/unload  (alias: /api/unload)
POST /api/voxcpm/synthesize  (alias: /api/synthesize)

Request (JSON):
{
  "text": "要合成的文本",
  "reference_audio_base64": "data:audio/wav;base64,xxxx...",  // 可选
  "prompt_wav_base64": "data:audio/wav;base64,yyyy...",       // 可选；不传则只使用 reference
  "prompt_text": "参考音频对应的文本",                         // 可选；仅随 prompt_wav 使用
  "model_id": "k2-fsa/OmniVoice",
  "device": "auto",
  "cfg_value": 2.0,                    // 可选；不传则自适应调节
  "inference_timesteps": 32,           // 可选；不传则自适应调节
  "t_shift": 0.1,                      // 可选；不传则自适应调节
  "layer_penalty_factor": 5.0,         // 可选
  "position_temperature": 5.0,         // 可选
  "class_temperature": 0.0,            // 可选；0=贪心解码
  "denoise": true,
  "optimize": false,
  "target_duration_ms": 2200,          // 可选；目标时长（毫秒），优先于 ref_duration 匹配
  "max_duration_ms": 3000,             // 可选；允许的最大时长（毫秒），超出时提前拒绝
  "duration_tolerance_ms": 176,        // 可选；目标时长容差，超容差会自动重试一次
  "quality_retry": true,               // 可选；检测空/静音/削顶等 badcase 后自动重试
  "audio_chunk_duration": 15.0,        // 可选；长文本分块时长
  "audio_chunk_threshold": 30.0,       // 可选；触发分块阈值
  "seed": 123456789                    // 可选；不传时默认派生稳定 seed
}

Adaptive Quality Profiles (based on reference audio length):
  < 2s:  num_step=48, guidance_scale=1.8, t_shift=0.05 (conservative)
  2-4s: num_step=40, guidance_scale=2.0, t_shift=0.08 (balanced)
  4-5s: num_step=36, guidance_scale=2.0, t_shift=0.10 (optimal)
  > 5s: num_step=32, guidance_scale=2.2, t_shift=0.10 (standard)

Duration control priority: target_duration_ms > duration > ref_duration heuristic > speed.
Note: exact duration matching works best with postprocess_output=false.

Response (JSON):
{
  "ok": true,
  "audio_base64": "data:audio/wav;base64,xxxx...",
  "output_path": "/abs/path/to/output.wav",
  "relative_path": "work/omni_voice_api_outputs/voxcpm_xxx.wav",
  "elapsed_seconds": 12.345,
  "audio_duration_seconds": 2.431,
  "target_duration_ms": 2200,
  "max_duration_ms": 3000,
  "duration_tolerance_ms": 176,
  "duration_attempts": 1,
  "duration_refinement_log": [
    {"attempt": 1, "target_duration": 2.2, "actual_duration": 2.431, "error": 0.231}
  ],
  "quality_issues": [],
  "quality_retried": false,
  "seed": 123456789,
  "adaptive_params": {
    "ref_duration": 3.5,
    "profile": "medium",
    "num_step": 40,
    "guidance_scale": 2.0,
    "t_shift": 0.08
  }
}
  </pre>
</body>
</html>""",
    )


async def on_startup(app):
    print(f"[OmniVoice API] listening on http://{app['host']}:{app['port']} (max request {MAX_REQUEST_MB} MB)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="OmniVoice API")
    parser.add_argument("--model", default="k2-fsa/OmniVoice", help="模型路径或 HuggingFace 仓库 ID")
    parser.add_argument("--device", default=None, help="运行设备 (cuda/mps/cpu)")
    parser.add_argument("--ip", default="0.0.0.0", help="服务器 IP")
    parser.add_argument("--port", type=int, default=6006, help="服务器端口")
    parser.add_argument("--load-asr", action="store_true", help="启动时加载 ASR（默认不加载）")
    args = parser.parse_args(argv)

    device = args.device or get_best_device()
    _set_api_model(None, args.model, device, args.load_asr)

    app = web.Application(client_max_size=MAX_REQUEST_SIZE)
    app["host"] = args.ip
    app["port"] = args.port
    app.add_routes(routes)
    app.on_startup.append(on_startup)
    web.run_app(app, host=args.ip, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
