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
import array
import base64
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import wave
from collections import OrderedDict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from aiohttp import web
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
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
DEFAULT_SEPARATOR_MODEL = os.environ.get("SEPARATION_MODEL", "vocals_mel_band_roformer.ckpt")
SEPARATION_MODEL_DIR = Path(
    os.environ.get(
        "AUDIO_SEPARATOR_MODEL_DIR",
        str(Path(os.environ.get("MODEL_DIR", WORK_ROOT / "models")) / "audio-separator"),
    )
)
DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_MODEL_DIR = Path(
    os.environ.get(
        "WHISPER_MODEL_DIR",
        str(Path(os.environ.get("MODEL_DIR", WORK_ROOT / "models")) / "whisper"),
    )
)
WHISPER_MAX_MODELS = int(os.environ.get("WHISPER_MAX_MODELS", "2"))
MIN_REFERENCE_DURATION_FOR_DURATION_RATIO = float(
    os.environ.get("OMNIVOICE_MIN_REFERENCE_DURATION_FOR_DURATION_RATIO", "1.5")
)
OUTPUT_TEXT_QC_LANGS = {
    code.strip().lower()
    for code in os.environ.get("OMNIVOICE_OUTPUT_TEXT_QC_LANGS", "fil,tl").split(",")
    if code.strip()
}
OUTPUT_TEXT_QC_MODEL = os.environ.get("OMNIVOICE_OUTPUT_TEXT_QC_MODEL", "large-v3")
OUTPUT_TEXT_QC_MIN_TOKENS = int(os.environ.get("OMNIVOICE_OUTPUT_TEXT_QC_MIN_TOKENS", "3"))
OUTPUT_TEXT_QC_MIN_COVERAGE = float(os.environ.get("OMNIVOICE_OUTPUT_TEXT_QC_MIN_COVERAGE", "0.62"))
OUTPUT_PEAK_CEILING = float(os.environ.get("OMNIVOICE_OUTPUT_PEAK_CEILING", "0.94"))

# Quality-issue labels considered severe enough to drive retry / report.
# Shared by the OmniVoice and VoxCPM synth paths so the set stays in sync.
_SEVERE_ISSUE_LABELS = frozenset({
    "empty", "clipping", "near_clipping", "harsh_high_freq",
    "impulsive_spike", "plosive", "periodic_pulse", "text_incomplete",
    "source_script_residue", "duration_off_target", "duration_off_reference",
})
VOXCPM_MODEL_ID = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
VOXCPM_LOAD_DENOISER = str(
    os.environ.get("VOXCPM_LOAD_DENOISER", "0")
).strip().lower() in {"1", "true", "yes", "on"}
VOXCPM_OPTIMIZE = str(
    os.environ.get("VOXCPM_OPTIMIZE", "0")
).strip().lower() in {"1", "true", "yes", "on"}
VOXCPM_VOICES_CACHE_SIZE = int(os.environ.get("VOXCPM_VOICES_CACHE_SIZE", "64"))
# When true, loading one TTS engine unloads the other to free VRAM (single 24GB
# GPU coexistence). Default 0: both may stay resident lazily.
_EXCLUSIVE_MODE = str(
    os.environ.get("OMNIVOICE_EXCLUSIVE_MODE", "0")
).strip().lower() in {"1", "true", "yes", "on"}

_API_MODEL = None
_API_MODEL_ID = "k2-fsa/OmniVoice"
_API_DEVICE = None
_API_LOAD_ASR = False
_MODEL_LOAD_LOCK = asyncio.Lock()
_WHISPER_MODELS: OrderedDict[Tuple[str, str, str], Any] = OrderedDict()
_WHISPER_MODEL_LOCK = asyncio.Lock()
# Inference concurrency is gated by a semaphore (not a lock) so multi-GPU or
# high-VRAM GPUs can serve requests in parallel. Default 1 preserves the
# previous serialized behaviour. Set OMNIVOICE_MAX_CONCURRENCY > 1 to enable.
_API_INFER_SEM = asyncio.Semaphore(
    int(os.environ.get("OMNIVOICE_MAX_CONCURRENCY", "1"))
)
_WHISPER_INFER_SEM = asyncio.Semaphore(
    int(os.environ.get("WHISPER_MAX_CONCURRENCY", "1"))
)

_DURATION_ESTIMATOR = RuleDurationEstimator()
_VOICE_PROMPT_CACHE: OrderedDict[str, Any] = OrderedDict()
_MAX_VOICE_PROMPT_CACHE_SIZE = int(
    os.environ.get("OMNIVOICE_VOICE_PROMPT_CACHE_SIZE", "100")
)
# Vendored VoxCPM2 backend state (parallel to the OmniVoice _API_MODEL globals).
# Loaded lazily on first /api/voxcpm/synthesize request.
_VOXCPM_MODEL = None
_VOXCPM_MODEL_ID = VOXCPM_MODEL_ID
_VOXCPM_LOAD_LOCK = asyncio.Lock()
_VOXCPM_VOICES: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_VOXCPM_DENOISE_AVAILABLE = None
_VOXCPM_NORMALIZE_AVAILABLE = None
# Whether max_duration_ms should hard-reject requests whose natural duration
# exceeds the limit. Default warn-only to avoid breaking existing callers.
_ENFORCE_MAX_DURATION = str(
    os.environ.get("OMNIVOICE_ENFORCE_MAX_DURATION", "0")
).strip().lower() in {"1", "true", "yes", "on"}
# Number of duration-refinement attempts on the first synthesis pass. Increase
# this (env var) when cloud callers frequently see duration_off_target severe
# issues and the extra latency is acceptable.
_DURATION_REFINEMENT_INITIAL_ATTEMPTS = int(
    os.environ.get("OMNIVOICE_DURATION_REFINEMENT_INITIAL_ATTEMPTS", "3")
)


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


def _assess_reference_quality(audio_bytes: Optional[bytes]) -> Dict[str, Any]:
    """Compute lightweight quality metrics for a reference audio clip.

    Returns a dict with duration/activity/peak/RMS/SNR and a list of issue
    flags.  ``is_poor`` is True when any issue is detected; downstream code
    uses this to fall back to a more conservative generation profile.
    """
    if not audio_bytes:
        return {"has_ref": False, "is_poor": False, "issues": []}

    try:
        y, sr = _decode_audio_bytes_mono(audio_bytes, 24000)
    except Exception as exc:
        return {
            "has_ref": True,
            "is_poor": True,
            "issues": ["decode_error"],
            "error": str(exc)[:200],
        }

    if y.size == 0:
        return {"has_ref": True, "is_poor": True, "issues": ["empty_reference"]}

    duration = float(y.size) / sr
    profile = _waveform_loudness_profile(y, sr)
    active_ratio = profile.get("activity_ratio")
    peak = float(np.max(np.abs(y)))
    rms = _compute_rms(y)

    intervals = _active_intervals_from_rms(y, sr)
    active_speech = sum(max(0.0, end - start) for start, end in intervals)
    active_speech_ratio = active_speech / duration if duration > 0 else 0.0

    mean_db = profile.get("mean_volume_db")
    active_mean_db = profile.get("active_mean_volume_db")
    snr = (
        (active_mean_db - mean_db)
        if mean_db is not None and active_mean_db is not None
        else None
    )
    snr_reliable = bool(
        snr is not None
        and active_ratio is not None
        # Require enough non-active frames for the noise-floor estimate to be
        # meaningful. When almost the whole clip is active speech, mean_db and
        # active_mean_db are nearly identical and SNR collapses to ~0 dB even
        # for clean audio, producing false low_snr flags.
        and active_ratio < 0.80
        and active_speech_ratio < 0.80
    )

    issues = []
    if active_ratio is not None and active_ratio < 0.30:
        issues.append("low_activity")
    if active_speech_ratio < 0.20:
        issues.append("mostly_silence")
    if duration < 1.0:
        issues.append("short_reference")
    if peak < 0.03:
        issues.append("too_quiet")
    if peak > 0.99:
        issues.append("clipping")
    if rms < 0.005:
        issues.append("low_rms")
    if snr_reliable and snr < 10.0:
        issues.append("low_snr")

    return {
        "has_ref": True,
        "duration": _round_float(duration, 3),
        "active_ratio": active_ratio,
        "active_speech_ratio": _round_float(active_speech_ratio, 3),
        "peak": _round_float(peak, 4),
        "rms": _round_float(rms, 5),
        "snr_db": _round_float(snr, 2) if snr is not None else None,
        "snr_reliable": snr_reliable,
        "is_poor": bool(issues),
        "issues": issues,
    }


def _select_quality_profile(
    ref_duration: Optional[float],
    ref_quality: Optional[Dict[str, Any]] = None,
) -> str:
    """Select quality profile based on reference audio duration and quality.

    Poor-quality references are downgraded to a more conservative profile so
    the model uses more steps / lower temperature and is less likely to emit
    pops, silence, or overshoot the requested duration.
    """
    if ref_duration is None:
        base = "medium"
    elif ref_duration < 2.0:
        base = "short"
    elif ref_duration < 4.0:
        base = "medium"
    elif ref_duration <= 5.0:
        base = "optimal"
    else:
        base = "long"

    if not ref_quality or not ref_quality.get("is_poor"):
        return base

    # Downgrade one tier toward "short" (the most conservative built-in profile).
    downgrade = {
        "long": "optimal",
        "optimal": "medium",
        "medium": "short",
        "short": "short",
    }
    return downgrade.get(base, base)


def _apply_quality_conservative_overrides(
    profile: Dict[str, Any], ref_quality: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply conservative overrides when reference audio quality is poor."""
    if not ref_quality or not ref_quality.get("is_poor"):
        return profile

    result = dict(profile)
    # More diffusion steps -> stabler output for noisy/short references.
    result["num_step"] = min(int(result.get("num_step", 32)) + 8, 64)
    # Slightly lower guidance reduces artifacts on weak references.
    result["guidance_scale"] = max(float(result.get("guidance_scale", 2.0)) - 0.2, 1.4)
    # Lower sampling temperatures for less randomness.
    result["position_temperature"] = max(
        float(result.get("position_temperature", 4.0)) - 1.0, 1.5
    )
    result["t_shift"] = max(float(result.get("t_shift", 0.08)) - 0.02, 0.03)
    return result


def _get_adaptive_params(
    ref_duration: Optional[float],
    user_cfg: Optional[float] = None,
    user_steps: Optional[int] = None,
    ref_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get adaptive parameters based on reference audio duration and quality.

    User-specified values take precedence over adaptive defaults.
    """
    profile_name = _select_quality_profile(ref_duration, ref_quality)
    profile = _QUALITY_PROFILES[profile_name].copy()
    profile = _apply_quality_conservative_overrides(profile, ref_quality)

    # User values override adaptive defaults
    if user_cfg is not None:
        profile["guidance_scale"] = user_cfg
    if user_steps is not None:
        profile["num_step"] = user_steps

    logger.info(
        f"Adaptive profile: {profile_name} (ref_duration={ref_duration}s, "
        f"ref_quality_issues={ref_quality.get('issues') if ref_quality else None}), "
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


def _whisper_device(preferred="auto"):
    value = str(preferred or "auto").strip().lower()
    if value == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_whisper_compute_type(device):
    return "float16" if device == "cuda" else "int8"


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


def _round_float(value, digits=3):
    try:
        return float(round(float(value), digits))
    except (TypeError, ValueError):
        return None


def _db_from_power(power):
    try:
        power = float(power)
    except (TypeError, ValueError):
        return None
    if power <= 1e-20:
        return None
    return 10.0 * np.log10(power)


def _db_from_peak(peak):
    try:
        peak = float(peak)
    except (TypeError, ValueError):
        return None
    if peak <= 1e-10:
        return None
    return 20.0 * np.log10(peak)


def _audio_loudness_profile(path, frame_seconds=0.4):
    frame_powers = []
    frame_dbs = []
    total_weighted_power = 0.0
    total_samples = 0
    peak = 0.0
    duration = 0.0

    with sf.SoundFile(str(path)) as audio:
        sample_rate = audio.samplerate or 48000
        channels = max(1, audio.channels or 1)
        frame_size = max(1, int(sample_rate * frame_seconds))
        duration = (len(audio) / sample_rate) if sample_rate else 0.0
        for block in audio.blocks(blocksize=frame_size, dtype="float32", always_2d=True):
            if block.size == 0:
                continue
            samples = block.reshape(-1)
            power = float(np.mean(np.square(samples, dtype=np.float64)))
            sample_count = int(block.shape[0] * channels)
            total_weighted_power += power * sample_count
            total_samples += sample_count
            if power > 1e-20:
                frame_powers.append(power)
                frame_dbs.append(10.0 * np.log10(power))
            peak = max(peak, float(np.max(np.abs(samples))) if samples.size else 0.0)

    mean_db = _db_from_power(total_weighted_power / total_samples) if total_samples else None
    profile = {
        "mean_volume_db": _round_float(mean_db, 2),
        "max_volume_db": _round_float(_db_from_peak(peak), 2),
        "duration_seconds": _round_float(duration, 3),
        "analysis_method": "frame_rms_active_loudness",
    }
    if not frame_dbs:
        return profile

    db_values = np.asarray(frame_dbs, dtype=np.float64)
    power_values = np.asarray(frame_powers, dtype=np.float64)
    high_db = float(np.percentile(db_values, 90))
    gate_db = max(-60.0, high_db - 35.0)
    active_mask = db_values >= gate_db
    if int(active_mask.sum()) < min(3, len(db_values)):
        active_mask = db_values >= float(np.percentile(db_values, 65))
    active_dbs = db_values[active_mask]
    active_powers = power_values[active_mask]
    active_mean_db = _db_from_power(float(np.mean(active_powers))) if active_powers.size else None
    active_p70_db = float(np.percentile(active_dbs, 70)) if active_dbs.size else active_mean_db
    profile.update(
        {
            "active_mean_volume_db": _round_float(active_mean_db, 2),
            "active_p70_volume_db": _round_float(active_p70_db, 2),
            "activity_ratio": _round_float(float(active_dbs.size / db_values.size), 3) if db_values.size else None,
            "active_gate_db": _round_float(gate_db, 2),
            "frame_seconds": frame_seconds,
        }
    )
    return profile


def _waveform_loudness_profile(waveform, sampling_rate: int, frame_seconds=0.04):
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    sampling_rate = int(sampling_rate or 0)
    duration = float(arr.size) / sampling_rate if sampling_rate > 0 else 0.0
    if arr.size == 0:
        return {
            "mean_volume_db": None,
            "max_volume_db": None,
            "duration_seconds": _round_float(duration, 3),
            "analysis_method": "waveform_frame_rms_loudness",
        }
    frame_size = max(1, int(max(0.01, float(frame_seconds)) * max(1, sampling_rate)))
    frame_powers = []
    frame_dbs = []
    total_power = float(np.mean(np.square(arr, dtype=np.float64)))
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    for start in range(0, arr.size, frame_size):
        block = arr[start : start + frame_size]
        if block.size == 0:
            continue
        power = float(np.mean(np.square(block, dtype=np.float64)))
        if power > 1e-20:
            frame_powers.append(power)
            frame_dbs.append(10.0 * np.log10(power))
    profile = {
        "mean_volume_db": _round_float(_db_from_power(total_power), 2),
        "max_volume_db": _round_float(_db_from_peak(peak), 2),
        "duration_seconds": _round_float(duration, 3),
        "analysis_method": "waveform_frame_rms_loudness",
        "frame_seconds": frame_seconds,
    }
    if not frame_dbs:
        return profile
    db_values = np.asarray(frame_dbs, dtype=np.float64)
    power_values = np.asarray(frame_powers, dtype=np.float64)
    high_db = float(np.percentile(db_values, 90))
    gate_db = max(-60.0, high_db - 35.0)
    active_mask = db_values >= gate_db
    if int(active_mask.sum()) < min(3, len(db_values)):
        active_mask = db_values >= float(np.percentile(db_values, 65))
    active_dbs = db_values[active_mask]
    active_powers = power_values[active_mask]
    active_mean_db = _db_from_power(float(np.mean(active_powers))) if active_powers.size else None
    active_p70_db = float(np.percentile(active_dbs, 70)) if active_dbs.size else active_mean_db
    profile.update(
        {
            "active_mean_volume_db": _round_float(active_mean_db, 2),
            "active_p70_volume_db": _round_float(active_p70_db, 2),
            "activity_ratio": _round_float(float(active_dbs.size / db_values.size), 3) if db_values.size else None,
            "active_gate_db": _round_float(gate_db, 2),
        }
    )
    return profile


def _waveform_speech_intervals(waveform, sampling_rate: int, frame_seconds=0.04):
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    sampling_rate = int(sampling_rate or 0)
    if arr.size == 0 or sampling_rate <= 0:
        return []
    frame_size = max(1, int(max(0.01, float(frame_seconds)) * sampling_rate))
    rows = []
    dbs = []
    for frame_index, start in enumerate(range(0, arr.size, frame_size)):
        block = arr[start : start + frame_size]
        if block.size == 0:
            continue
        power = float(np.mean(np.square(block, dtype=np.float64)))
        db = _db_from_power(power)
        dbs.append(db if db is not None else -120.0)
        rows.append((frame_index, start, min(start + block.size, arr.size), db if db is not None else -120.0))
    if not rows:
        return []
    db_values = np.asarray(dbs, dtype=np.float64)
    high_db = float(np.percentile(db_values, 90))
    gate_db = max(-52.0, high_db - 32.0)
    intervals = []
    current = None
    for _frame_index, start, end, db in rows:
        if db >= gate_db:
            if current is None:
                current = [start / sampling_rate, end / sampling_rate]
            else:
                current[1] = end / sampling_rate
        elif current is not None:
            if current[1] - current[0] >= 0.06:
                intervals.append(tuple(current))
            current = None
    if current is not None and current[1] - current[0] >= 0.06:
        intervals.append(tuple(current))
    return _merge_time_intervals(intervals, gap=0.12)


def _build_synth_audio_qc(waveform, sampling_rate: int, quality_issues=None, spike_locations=None):
    duration = _audio_duration(waveform, sampling_rate)
    speech_intervals = _waveform_speech_intervals(waveform, sampling_rate)
    speech_total = sum(max(0.0, end - start) for start, end in speech_intervals)
    loudness = _waveform_loudness_profile(waveform, sampling_rate)
    return {
        "version": 1,
        "source": "omnivoice_cloud_synthesize",
        "analysis_method": "waveform_frame_energy",
        "duration_sec": _round_float(duration),
        "speech_total_sec": _round_float(speech_total),
        "speech_ratio": _round_float(speech_total / duration) if duration > 0 else None,
        "speech_interval_count": len(speech_intervals),
        "speech_intervals": [
            {"start": _round_float(start), "end": _round_float(end), "duration": _round_float(end - start)}
            for start, end in speech_intervals[:500]
        ],
        "loudness": loudness,
        "quality_issues": list(quality_issues or []),
        "spike_locations": list(spike_locations or []),
    }


# ---------------------------------------------------------------------------
# Audio QC endpoints: offload CPU-heavy signal analysis from the dubbing host.
# ---------------------------------------------------------------------------

# Reference endpoint guard constants (mirrored from dubbing reference_guard.py).
_QC_REF_GUARD_ENABLED = str(os.environ.get("OMNIVOICE_REF_GUARD_ENABLED", "1")).strip().lower() in {
    "1", "true", "yes", "on",
}
_QC_REF_GUARD_SECONDS = 0.35
_QC_REF_TRIM_SECONDS = 0.25
_QC_REF_MIN_DURATION = 1.40
_QC_REF_MIN_FINAL_DURATION = 1.00
_QC_REF_FADE_SECONDS = 0.025


def _rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in values) / len(values))


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _dbfs(amplitude):
    if amplitude <= 0:
        return -120.0
    return 20.0 * math.log10(min(1.0, float(amplitude) / 32768.0))


def _downsample_int16(samples, sample_rate, target_rate=8000):
    if sample_rate <= target_rate:
        return samples, sample_rate
    step = max(1, int(round(sample_rate / target_rate)))
    return samples[::step], int(round(sample_rate / step))


def _estimate_frame_f0(frame, sample_rate):
    """Autocorrelation F0 estimator ported from dubbing reference_guard.py."""
    if not frame:
        return None
    mean = sum(frame) / len(frame)
    centered = [float(sample) - mean for sample in frame]
    energy = sum(sample * sample for sample in centered)
    if energy <= 0:
        return None
    min_lag = max(1, int(sample_rate / 320.0))
    max_lag = min(len(centered) - 2, int(sample_rate / 70.0))
    if max_lag <= min_lag:
        return None
    best_lag = None
    best_corr = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        numerator = sum(a * b for a, b in zip(left, right))
        left_energy = sum(a * a for a in left)
        right_energy = sum(b * b for b in right)
        if left_energy <= 0 or right_energy <= 0:
            continue
        corr = numerator / math.sqrt(left_energy * right_energy)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_lag is None or best_corr < 0.45:
        return None
    return sample_rate / best_lag, best_corr


def _segment_profile(samples, sample_rate, start, end):
    """Return gender/f0 profile for a reference-clip segment."""
    start_index = max(0, int(start * sample_rate))
    end_index = min(len(samples), int(end * sample_rate))
    if end_index <= start_index:
        return {"gender": "unknown", "confidence": 0.0, "f0": None}
    segment = samples[start_index:end_index]
    frame_size = max(1, int(0.080 * sample_rate))
    if len(segment) < frame_size:
        return {"gender": "unknown", "confidence": 0.0, "f0": None}
    frame_count = min(7, max(1, int((len(segment) - frame_size) / max(1, frame_size)) + 1))
    offsets = []
    if frame_count == 1:
        offsets = [(len(segment) - frame_size) // 2]
    else:
        span = len(segment) - frame_size
        offsets = [int(round(span * i / (frame_count - 1))) for i in range(frame_count)]
    frame_rms = [_rms(segment[offset : offset + frame_size]) for offset in offsets]
    active_floor = max(120.0, max(frame_rms or [0.0]) * 0.20)
    f0_values = []
    corr_values = []
    for offset, rms_value in zip(offsets, frame_rms):
        if rms_value < active_floor:
            continue
        estimate = _estimate_frame_f0(segment[offset : offset + frame_size], sample_rate)
        if not estimate:
            continue
        f0, corr = estimate
        f0_values.append(f0)
        corr_values.append(corr)
    median_f0 = _median(f0_values)
    median_corr = _median(corr_values) or 0.0
    voiced_ratio = len(f0_values) / max(1, len(offsets))
    if median_f0 is None or voiced_ratio < 0.35:
        return {"gender": "unknown", "confidence": 0.0, "f0": median_f0}
    if median_f0 <= 155.0:
        distance = min(1.0, max(0.0, (165.0 - median_f0) / 55.0))
        gender = "male"
    elif median_f0 >= 190.0:
        distance = min(1.0, max(0.0, (median_f0 - 180.0) / 80.0))
        gender = "female"
    else:
        return {"gender": "unknown", "confidence": 0.0, "f0": median_f0}
    confidence = max(0.0, min(1.0, median_corr * 0.70 + voiced_ratio * 0.20 + distance * 0.10))
    if confidence < 0.55:
        gender = "unknown"
    return {"gender": gender, "confidence": confidence, "f0": median_f0}


def _opposite_gender(left, right):
    return {left, right} == {"male", "female"}


def _decode_base64_audio_to_bytes(b64_data):
    b64_data = str(b64_data or "").strip()
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
    return base64.b64decode(b64_data)


def _reference_quality_legacy(audio_bytes):
    """Return dubbing-compatible reference quality dict from raw WAV bytes."""
    enhanced = _assess_reference_quality(audio_bytes)
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except Exception as exc:
        return {"ok": False, "error": f"read_failed:{exc}"}
    if sample_width != 2 or channels <= 0:
        return {"ok": False, "error": "unsupported_wav"}
    data = array.array("h")
    data.frombytes(raw)
    if channels == 1:
        samples = list(data)
    else:
        samples = [
            int(sum(data[index : index + channels]) / channels)
            for index in range(0, len(data) - channels + 1, channels)
        ]
    if not samples or sample_rate <= 0:
        return {"ok": False, "error": "empty_reference"}
    samples, sample_rate = _downsample_int16(samples, sample_rate)
    duration = len(samples) / float(sample_rate)
    peak = max(abs(int(sample)) for sample in samples) if samples else 0
    rms_value = _rms(samples)
    frame_size = max(1, int(0.050 * sample_rate))
    frame_rms = [
        _rms(samples[offset : offset + frame_size])
        for offset in range(0, max(0, len(samples) - frame_size + 1), frame_size)
    ]
    if not frame_rms and samples:
        frame_rms = [rms_value]
    sorted_rms = sorted(frame_rms)
    floor_count = max(1, int(len(sorted_rms) * 0.20)) if sorted_rms else 1
    noise_floor = sum(sorted_rms[:floor_count]) / floor_count if sorted_rms else 0.0
    max_frame = max(frame_rms or [0.0])
    noise_threshold = min(noise_floor * 2.8, max_frame * 0.35) if max_frame > 0 else 0.0
    active_threshold = max(120.0, noise_threshold, max_frame * 0.12)
    active_frames = sum(1 for value in frame_rms if value >= active_threshold)
    active_ratio = active_frames / max(1, len(frame_rms))
    result = {
        "ok": True,
        "duration": round(duration, 3),
        "peak_db": round(_dbfs(peak), 1),
        "rms_db": round(_dbfs(rms_value), 1),
        "active_ratio": round(active_ratio, 3),
    }
    for key in ("issues", "snr_db", "snr_reliable", "active_speech_ratio"):
        if key in enhanced:
            result[key] = enhanced[key]
    return result


def _reference_endpoint_guard(audio_bytes):
    """Run F0-based endpoint guard on raw WAV bytes; return guard result dict.

    If the clip edges contain opposite-gender speech compared to the body,
    return trim amounts. The caller can apply them locally or request trimmed
    audio bytes via ``trim_on_guard``.
    """
    if not _QC_REF_GUARD_ENABLED:
        return {"trimmed": False, "reason": "disabled"}
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except Exception as exc:
        return {"trimmed": False, "reason": f"read_failed:{exc}"}
    if sample_width != 2 or channels <= 0:
        return {"trimmed": False, "reason": "unsupported_wav"}
    data = array.array("h")
    data.frombytes(raw)
    if channels == 1:
        samples = list(data)
    else:
        samples = [
            int(sum(data[index : index + channels]) / channels)
            for index in range(0, len(data) - channels + 1, channels)
        ]
    if not samples or sample_rate <= 0:
        return {"trimmed": False, "reason": "empty_reference"}
    samples, sample_rate = _downsample_int16(samples, sample_rate)
    duration = len(samples) / float(sample_rate)
    edge = min(_QC_REF_GUARD_SECONDS, max(0.0, (duration - 0.50) / 3.0))
    if duration < _QC_REF_MIN_DURATION or edge < 0.18:
        return {"trimmed": False, "reason": "too_short", "duration": duration}
    body_start = edge
    body_end = duration - edge
    if body_end - body_start < 0.45:
        return {"trimmed": False, "reason": "body_too_short", "duration": duration}
    body = _segment_profile(samples, sample_rate, body_start, body_end)
    if body["gender"] == "unknown":
        return {"trimmed": False, "reason": "body_uncertain", "duration": duration, "body": body}
    start_profile = _segment_profile(samples, sample_rate, 0.0, edge)
    end_profile = _segment_profile(samples, sample_rate, duration - edge, duration)
    start_polluted = (
        start_profile["confidence"] >= 0.60 and _opposite_gender(body["gender"], start_profile["gender"])
    )
    end_polluted = end_profile["confidence"] >= 0.60 and _opposite_gender(body["gender"], end_profile["gender"])
    if not start_polluted and not end_polluted:
        return {
            "trimmed": False,
            "reason": "clean",
            "duration": duration,
            "body": body,
            "start": start_profile,
            "end": end_profile,
        }
    trim_unit = min(_QC_REF_TRIM_SECONDS, edge)
    allowed_trim = max(0.0, duration - _QC_REF_MIN_FINAL_DURATION)
    end_trim = min(trim_unit if end_polluted else 0.0, allowed_trim)
    allowed_trim -= end_trim
    start_trim = min(trim_unit if start_polluted else 0.0, allowed_trim)
    if start_trim <= 0.0 and end_trim <= 0.0:
        return {"trimmed": False, "reason": "min_duration_guard", "duration": duration}
    return {
        "trimmed": True,
        "start_trim": start_trim,
        "end_trim": end_trim,
        "duration": duration,
        "body": body,
        "start": start_profile,
        "end": end_profile,
    }


def _trim_audio_bytes(audio_bytes, start_trim, end_trim):
    """Apply start/end trim with short fades and return WAV bytes."""
    buf = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_wav(buf)
    duration_sec = len(audio) / 1000.0
    final_duration = max(0.05, duration_sec - start_trim - end_trim)
    fade = min(_QC_REF_FADE_SECONDS, final_duration / 4.0)
    start_ms = int(start_trim * 1000)
    end_ms = int((duration_sec - end_trim) * 1000)
    trimmed = audio[start_ms:end_ms]
    if fade > 0:
        trimmed = trimmed.fade_in(int(fade * 1000)).fade_out(int(fade * 1000))
    out = io.BytesIO()
    trimmed.export(out, format="wav")
    return out.getvalue()


def _build_loudness_profile_from_bytes(audio_bytes, frame_seconds=0.4):
    """Return active-gated loudness profile from raw audio bytes."""
    try:
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T
    except Exception as exc:
        return {"error": str(exc)}
    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    arr = data.reshape(-1)
    if arr.size == 0 or sr <= 0:
        return {"error": "empty_audio"}
    frame_size = max(1, int(sr * frame_seconds))
    frame_powers = []
    frame_dbs = []
    total_weighted_power = 0.0
    total_samples = 0
    peak = 0.0
    for start in range(0, arr.size, frame_size):
        block = arr[start : start + frame_size]
        if block.size == 0:
            continue
        power = float(np.mean(np.square(block, dtype=np.float64)))
        sample_count = block.size
        total_weighted_power += power * sample_count
        total_samples += sample_count
        if power > 1e-20:
            frame_powers.append(power)
            frame_dbs.append(10.0 * np.log10(power))
        peak = max(peak, float(np.max(np.abs(block))) if block.size else 0.0)
    duration = arr.size / sr
    mean_db = _db_from_power(total_weighted_power / total_samples) if total_samples else None
    profile = {
        "mean_volume_db": _round_float(mean_db, 2),
        "max_volume_db": _round_float(_db_from_peak(peak), 2),
        "duration_seconds": _round_float(duration, 3),
        "analysis_method": "frame_rms_active_loudness",
    }
    if not frame_dbs:
        return profile
    db_values = np.asarray(frame_dbs, dtype=np.float64)
    power_values = np.asarray(frame_powers, dtype=np.float64)
    high_db = float(np.percentile(db_values, 90))
    gate_db = max(-60.0, high_db - 35.0)
    active_mask = db_values >= gate_db
    if int(active_mask.sum()) < min(3, len(db_values)):
        active_mask = db_values >= float(np.percentile(db_values, 65))
    active_dbs = db_values[active_mask]
    active_powers = power_values[active_mask]
    active_mean_db = _db_from_power(float(np.mean(active_powers))) if active_powers.size else None
    active_p70_db = float(np.percentile(active_dbs, 70)) if active_dbs.size else active_mean_db
    profile.update(
        {
            "active_mean_volume_db": _round_float(active_mean_db, 2),
            "active_p70_volume_db": _round_float(active_p70_db, 2),
            "activity_ratio": _round_float(float(active_dbs.size / db_values.size), 3) if db_values.size else None,
            "active_gate_db": _round_float(gate_db, 2),
            "frame_seconds": frame_seconds,
        }
    )
    return profile


def _build_speech_intervals_from_bytes(audio_bytes, frame_seconds=0.04):
    """Return speech intervals from raw audio bytes."""
    try:
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T
    except Exception as exc:
        return {"error": str(exc)}
    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    arr = data.reshape(-1)
    if arr.size == 0 or sr <= 0:
        return {"error": "empty_audio"}
    intervals = _waveform_speech_intervals(arr, sr)
    duration = arr.size / sr
    speech_total = sum(max(0.0, end - start) for start, end in intervals)
    return {
        "duration_sec": _round_float(duration),
        "speech_total_sec": _round_float(speech_total),
        "speech_ratio": _round_float(speech_total / duration) if duration > 0 else None,
        "speech_interval_count": len(intervals),
        "speech_intervals": [
            {"start": _round_float(start), "end": _round_float(end), "duration": _round_float(end - start)}
            for start, end in intervals[:500]
        ],
        "analysis_method": "waveform_frame_energy",
    }


def _merge_time_intervals(intervals, gap=0.10):
    cleaned = sorted((max(0.0, float(s)), max(0.0, float(e))) for s, e in intervals if e > s)
    merged = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1] + gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _whisper_speech_intervals(segments, duration):
    intervals = []
    word_count = 0
    for segment in segments or []:
        words = segment.get("words") if isinstance(segment, dict) else None
        if words:
            for word in words:
                try:
                    start = float(word.get("start"))
                    end = float(word.get("end"))
                except (TypeError, ValueError):
                    continue
                if end > start:
                    intervals.append((start, end))
                    word_count += 1
            continue
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        if end > start:
            intervals.append((start, end))
    duration = float(duration or 0.0)
    merged = _merge_time_intervals(
        [(max(0.0, min(s, duration)), max(0.0, min(e, duration))) for s, e in intervals],
        gap=0.16,
    )
    return merged, word_count


def _build_whisper_audio_qc(audio_path, result):
    duration = result.get("duration") or _audio_duration_seconds(audio_path) or 0.0
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    segments = result.get("segments") or []
    speech_intervals, word_count = _whisper_speech_intervals(segments, duration)
    speech_total = sum(max(0.0, end - start) for start, end in speech_intervals)
    loudness = {}
    try:
        loudness = _audio_loudness_profile(audio_path)
    except Exception as exc:
        loudness = {"error": str(exc)[:300]}
    return {
        "version": 1,
        "source": "whisper_cloud",
        "analysis_method": "whisper_segments+soundfile_loudness",
        "duration_sec": _round_float(duration),
        "speech_total_sec": _round_float(speech_total),
        "speech_ratio": _round_float(speech_total / duration) if duration > 0 else None,
        "speech_interval_count": len(speech_intervals),
        "speech_intervals": [
            {"start": _round_float(start), "end": _round_float(end), "duration": _round_float(end - start)}
            for start, end in speech_intervals[:2000]
        ],
        "segment_count": len(segments),
        "word_count": word_count,
        "language": result.get("language"),
        "language_probability": result.get("language_probability"),
        "loudness": loudness,
    }


def _find_cli(name):
    candidates = [ROOT / ".venv" / "bin" / name, shutil.which(name)]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def _ffmpeg_binary():
    return os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg") or "ffmpeg"


def _safe_filename(name, default="input"):
    name = Path(str(name or default)).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or default


def _audio_separator_config_candidates(model):
    return {
        "vocals_mel_band_roformer.ckpt": ["vocals_mel_band_roformer.yaml"],
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt": ["model_bs_roformer_ep_317_sdr_12.9755.yaml"],
    }.get(model, [])


def _separator_model_files_present(model_dir, model):
    required = [Path(model_dir) / model]
    required.extend(Path(model_dir) / name for name in _audio_separator_config_candidates(model))
    return all(path.exists() for path in required)


def _find_audio_stem(root, names, excludes=()):
    root = Path(root)
    wanted = {re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") for name in names}
    blocked = {re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") for name in excludes}
    audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
    files = [p for p in root.glob("**/*") if p.is_file() and p.suffix.lower() in audio_exts]

    def normalized(path):
        return re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")

    for path in files:
        stem = normalized(path)
        if stem in wanted and stem not in blocked:
            return path
    for path in files:
        stem = normalized(path)
        if any(block in stem for block in blocked):
            continue
        if any(name in stem for name in wanted):
            return path
    return None


def _locate_separator_stems(output_dir):
    output_dir = Path(output_dir)
    vocals = output_dir / "vocals.wav"
    background = output_dir / "no_vocals.wav"
    if not vocals.exists():
        vocals = _find_audio_stem(output_dir, {"vocals", "vocal"}, excludes={"no_vocals", "instrumental", "other"})
    if not background.exists():
        background = _find_audio_stem(output_dir, {"no_vocals", "instrumental", "instrumentals", "other"})
    return vocals, background


def _clean_separator_outputs(output_dir, keep=()):
    keep_paths = {Path(path).resolve() for path in keep}
    audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
    for path in Path(output_dir).glob("*"):
        if not path.is_file() or path.suffix.lower() not in audio_exts:
            continue
        try:
            if path.resolve() in keep_paths:
                continue
        except FileNotFoundError:
            continue
        path.unlink(missing_ok=True)


def _run_cmd(cmd, *, env=None, check=True):
    result = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _prepare_separator_model(audio_separator_cli, model_dir, model):
    model_dir.mkdir(parents=True, exist_ok=True)
    if _separator_model_files_present(model_dir, model):
        logger.info("Audio Separator model files already present for %s in %s", model, model_dir)
        return
    _run_cmd(
        [
            audio_separator_cli,
            "--model_filename",
            model,
            "--model_file_dir",
            model_dir,
            "--download_model_only",
        ],
    )


def _separate_audio_sync(input_path, output_dir, options):
    audio_separator_cli = _find_cli("audio-separator")
    if not audio_separator_cli:
        raise RuntimeError("audio-separator CLI not found. Install the audio-separator dependency.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_audio = output_dir / "input_audio.wav"
    _run_cmd(
        [
            _ffmpeg_binary(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            input_audio,
        ],
    )
    _clean_separator_outputs(output_dir, keep={input_audio})

    model = str(options.get("model") or DEFAULT_SEPARATOR_MODEL).strip() or DEFAULT_SEPARATOR_MODEL
    model_dir = Path(options.get("model_dir") or SEPARATION_MODEL_DIR)
    _prepare_separator_model(audio_separator_cli, model_dir, model)

    normalization = float(options.get("normalization", 0.9))
    chunk_duration = options.get("chunk_duration")
    use_autocast = _bool_option(options.get("use_autocast"), False)
    batch_size = int(float(options.get("batch_size", 1) or 1))
    segment_size = int(float(options.get("segment_size", 1) or 1))
    output_names = json.dumps({"Vocals": "vocals", "Instrumental": "no_vocals", "Other": "no_vocals"})

    cmd = [
        audio_separator_cli,
        input_audio,
        "--model_filename",
        model,
        "--output_dir",
        output_dir,
        "--model_file_dir",
        model_dir,
        "--output_format",
        "WAV",
        "--normalization",
        str(max(0.1, min(1.0, normalization))),
        "--custom_output_names",
        output_names,
    ]
    if chunk_duration not in (None, ""):
        cmd.extend(["--chunk_duration", str(max(30.0, min(3600.0, float(chunk_duration))))])
    if use_autocast:
        cmd.append("--use_autocast")

    is_mdx = model.lower().endswith(".onnx")
    if is_mdx:
        if batch_size > 1:
            cmd.extend(["--mdx_batch_size", str(max(1, min(64, batch_size)))])
        if segment_size >= 32:
            cmd.extend(["--mdx_segment_size", str(max(32, min(4096, segment_size)))])
    else:
        if batch_size > 1:
            cmd.extend(["--mdxc_batch_size", str(max(1, min(64, batch_size)))])
        if segment_size >= 32:
            cmd.extend(["--mdxc_segment_size", str(max(32, min(4096, segment_size)))])

    result = _run_cmd(cmd, check=False)
    vocals, background = _locate_separator_stems(output_dir)
    if (not vocals or not background or not vocals.exists() or not background.exists()) and model.lower().endswith(".ckpt"):
        _clean_separator_outputs(output_dir, keep={input_audio})
        result = _run_cmd([*cmd, "--mdxc_override_model_segment_size"], check=False)
        vocals, background = _locate_separator_stems(output_dir)

    if not vocals or not background or not vocals.exists() or not background.exists():
        raise RuntimeError(
            "Audio Separator did not write vocals/no_vocals stems.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    return {
        "vocals": vocals,
        "background": background,
        "model": model,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


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


def _reference_quality_score(quality: Optional[Dict[str, Any]]) -> float:
    if not quality or not quality.get("has_ref"):
        return -1000.0
    duration = quality.get("duration") or 0.0
    active_ratio = quality.get("active_ratio") or 0.0
    peak = quality.get("peak") or 0.0
    rms = quality.get("rms") or 0.0

    # Prefer ~4s active references with healthy level. Poor references are
    # still ranked against each other so a noisy primary can be replaced by a
    # materially better alternate instead of being kept only because every
    # candidate has some issue.
    duration_score = -abs(duration - 4.0)
    active_score = (active_ratio - 0.5) * 4.0
    level_score = min(1.0, peak * 2.0) + min(1.0, rms * 20.0)
    snr = quality.get("snr_db")
    snr_score = 0.0 if snr is None else max(-5.0, min(5.0, (snr - 15.0) / 5.0))
    issue_penalties = {
        "mostly_silence": 8.0,
        "low_activity": 4.0,
        "too_quiet": 4.0,
        "clipping": 3.0,
        "low_rms": 2.5,
        "short_reference": 1.5,
        "low_snr": 1.0,
    }
    penalty = sum(issue_penalties.get(issue, 1.0) for issue in quality.get("issues") or [])
    return duration_score + active_score + level_score + snr_score - penalty


def _select_best_reference(primary_bytes, primary_quality, alternate_refs, alternate_texts):
    """Score references; return best bytes/quality/text plus best and primary scores.

    Falls back to the primary reference when alternates are worse or un-decodable.
    """
    best_bytes = primary_bytes
    best_quality = primary_quality
    best_text = ""
    primary_score = _reference_quality_score(primary_quality)
    if not alternate_refs:
        return best_bytes, best_quality, best_text, primary_score, primary_score
    candidates = [(primary_bytes, primary_quality, "")]
    for ref_b64, text in zip_longest(alternate_refs, alternate_texts, fillvalue=""):
        try:
            raw = _decode_base64_audio_bytes(ref_b64)
            quality = _assess_reference_quality(raw)
            candidates.append((raw, quality, text or ""))
        except Exception:
            continue

    candidates.sort(key=lambda item: _reference_quality_score(item[1]), reverse=True)
    best = candidates[0]
    return best[0], best[1], best[2], _reference_quality_score(best[1]), primary_score


# ---------------------------------------------------------------------------
# Voice clone prompt cache
# ---------------------------------------------------------------------------


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


def _clamp_waveform_to_max_duration(
    waveform,
    sampling_rate: int,
    max_duration_sec: Optional[float],
    fade_out_ms: float = 20.0,
):
    """Hard-trim generated audio to max_duration_sec with a short fade-out.

    Returns (trimmed_waveform, was_trimmed).
    """
    if max_duration_sec is None or max_duration_sec <= 0:
        return waveform, False
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    max_samples = int(max_duration_sec * sampling_rate)
    if arr.size <= max_samples:
        return waveform, False
    trimmed = arr[:max_samples].copy()
    fade_samples = min(int(fade_out_ms / 1000.0 * sampling_rate), trimmed.size // 4)
    if fade_samples > 1:
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed[-fade_samples:] *= fade_out
    return trimmed, True


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
        if _EXCLUSIVE_MODE and _VOXCPM_MODEL is not None:
            _unload_voxcpm_model_sync()
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


# ---------------------------------------------------------------------------
# VoxCPM2 backend: lazy loader, memory release, voice registry
# ---------------------------------------------------------------------------

def _release_memory():
    import gc
    gc.collect()
    if sys.platform != "win32":
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _voxcpm_optional_available(name: str) -> bool:
    """Lazy capability probe for optional VoxCPM deps (modelscope/wetext)."""
    try:
        import importlib
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _voxcpm_denoise_available() -> bool:
    global _VOXCPM_DENOISE_AVAILABLE
    if _VOXCPM_DENOISE_AVAILABLE is None:
        _VOXCPM_DENOISE_AVAILABLE = _voxcpm_optional_available("modelscope")
    return _VOXCPM_DENOISE_AVAILABLE


def _voxcpm_normalize_available() -> bool:
    global _VOXCPM_NORMALIZE_AVAILABLE
    if _VOXCPM_NORMALIZE_AVAILABLE is None:
        _VOXCPM_NORMALIZE_AVAILABLE = (
            _voxcpm_optional_available("wetext")
            and _voxcpm_optional_available("inflect")
        )
    return _VOXCPM_NORMALIZE_AVAILABLE


def _load_voxcpm_model_sync():
    from voxcpm import VoxCPM
    model_id = _VOXCPM_MODEL_ID
    load_denoiser = VOXCPM_LOAD_DENOISER and _voxcpm_denoise_available()
    if VOXCPM_LOAD_DENOISER and not load_denoiser:
        logger.warning(
            "VOXCPM_LOAD_DENOISER=1 but modelscope is not installed; "
            "starting VoxCPM without the ZipEnhancer denoiser."
        )
    logger.info(f"加载 VoxCPM 模型: {model_id}, 设备: {_API_DEVICE}, denoiser={load_denoiser} ...")
    model = VoxCPM.from_pretrained(
        model_id,
        load_denoiser=load_denoiser,
        optimize=VOXCPM_OPTIMIZE,
        device=_API_DEVICE,
    )
    logger.info("VoxCPM 模型加载完成！")
    return model


def _unload_voxcpm_model_sync():
    global _VOXCPM_MODEL
    count = 1 if _VOXCPM_MODEL is not None else 0
    _VOXCPM_MODEL = None
    _VOXCPM_VOICES.clear()
    _release_memory()
    return count


async def _ensure_voxcpm_model():
    global _VOXCPM_MODEL, _API_MODEL
    if _VOXCPM_MODEL is not None:
        return _VOXCPM_MODEL
    async with _VOXCPM_LOAD_LOCK:
        if _VOXCPM_MODEL is not None:
            return _VOXCPM_MODEL
        if _EXCLUSIVE_MODE and _API_MODEL is not None:
            # Evict OmniVoice to make room on a single GPU.
            _API_MODEL = None
            _VOICE_PROMPT_CACHE.clear()
            _release_memory()
        _VOXCPM_MODEL = await asyncio.to_thread(_load_voxcpm_model_sync)
    return _VOXCPM_MODEL


def _voxcpm_sample_rate(model) -> int:
    sr = getattr(getattr(model, "tts_model", None), "sample_rate", None)
    if not sr:
        sr = getattr(model, "sample_rate", 48000)
    return int(sr)


def _voice_id_for_bytes(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes).hexdigest()[:16]


def _register_voxcpm_voice(reference_bytes: bytes, prompt_bytes=None, prompt_text=""):
    voice_id = _voice_id_for_bytes(reference_bytes)
    ref_dur = _bytes_audio_duration(reference_bytes) or 0.0
    entry = {
        "reference_bytes": reference_bytes,
        "prompt_bytes": prompt_bytes,
        "prompt_text": prompt_text or "",
        "reference_duration_ms": int(round(ref_dur * 1000)),
        "created_at": time.time(),
    }
    _VOXCPM_VOICES[voice_id] = entry
    _VOXCPM_VOICES.move_to_end(voice_id)
    while len(_VOXCPM_VOICES) > VOXCPM_VOICES_CACHE_SIZE:
        _VOXCPM_VOICES.popitem(last=False)
    return voice_id, entry


def _voxcpm_voice_meta(entry) -> Dict[str, Any]:
    return {
        "voice_id": _voice_id_for_bytes(entry["reference_bytes"]),
        "reference_duration_ms": entry.get("reference_duration_ms"),
        "has_prompt": entry.get("prompt_bytes") is not None,
        "created_at": round(entry.get("created_at") or 0.0, 3),
    }


def _stable_voxcpm_seed(data, text, prompt_text, reference_audio_base64, prompt_wav_base64):
    """Deterministic seed for VoxCPM, aligned with the dubbing caller's stable_voxcpm_seed."""
    explicit = _normalize_seed(data.get("seed") or data.get("voxcpm_seed"))
    if explicit is not None:
        return explicit
    if str(os.environ.get("OMNIVOICE_DETERMINISTIC", "1")).lower() in {"0", "false", "no", "off"}:
        return None
    payload = {
        "text": text,
        "prompt_text": prompt_text,
        "reference_audio_sha256": _sha256_text(reference_audio_base64),
        "prompt_audio_sha256": _sha256_text(prompt_wav_base64),
        "model_id": data.get("model_id") or _VOXCPM_MODEL_ID,
        "cfg_value": data.get("cfg_value", 2.0),
        "inference_timesteps": data.get("inference_timesteps", 10),
        "denoise": _bool_option(data.get("denoise"), False),
        "normalize": _bool_option(data.get("normalize"), False),
        "control_instruction": data.get("control_instruction") or data.get("instruct") or "",
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16) % OMNIVOICE_SEED_MOD


_LANG_CODE_ALIASES = {
    "tl": "fil",
    "filipino": "fil",
}
_CJK_SCRIPT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_CJK_PROMPT_LANGUAGE_CODES = {"zh", "ja", "ko", "yue", "cmn", "zho", "jpn", "kor"}


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


def _language_allows_cjk_prompt(language) -> bool:
    resolved = _resolve_language(language)
    code = str(resolved or language or "").strip().lower().split("-", 1)[0]
    return code in _CJK_PROMPT_LANGUAGE_CODES


def _sanitize_cross_language_prompt_text(prompt_text: str, language) -> str:
    prompt_text = re.sub(r"\s+", " ", (prompt_text or "").strip())
    if not prompt_text:
        return ""
    if _CJK_SCRIPT_RE.search(prompt_text) and not _language_allows_cjk_prompt(language):
        return ""
    return prompt_text


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
    ratio_clamp=None,
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

    target_duration = float(target_duration)
    # Default tolerance: 5% of target or 50ms, whichever is larger, so that
    # callers who only pass target_duration still get refinement.
    if duration_tolerance is None or duration_tolerance <= 0:
        duration_tolerance = max(0.05, target_duration * 0.05)

    # Use a wider correction range when the first attempt is far off; tighten
    # afterwards to avoid oscillation.
    ratio_clamp = ratio_clamp or (0.33, 3.0)

    current_duration = target_duration
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
        error = abs(actual_duration - target_duration)
        attempt_log.append(
            {
                "attempt": attempt + 1,
                "target_duration": current_duration,
                "actual_duration": actual_duration,
                "error": error,
            }
        )

        if error <= duration_tolerance:
            return audio, attempt + 1, attempt_log

        if error < best_error:
            best_error = error
            best_audio = audio

        if attempt < max_attempts - 1 and actual_duration > 0:
            # If the model is roughly linear in the duration parameter,
            # actual = k * current_duration. We want next such that
            # k * next = target_duration, so next = target_duration / k
            # = target_duration * current_duration / actual_duration.
            # Clamp to avoid divergence when the model output is wildly off
            # (e.g. actual=0.1s for a 10s target).
            raw_ratio = target_duration / actual_duration
            # Tighten clamp after first attempt.
            if attempt == 0:
                low, high = 0.25, 4.0
            else:
                low, high = ratio_clamp
            clamped_ratio = max(low, min(high, raw_ratio))
            next_duration = current_duration * clamped_ratio
            if max_duration is not None and next_duration > max_duration:
                logger.warning(
                    "Duration refinement ratio %.3f would push duration param to %.3fs, "
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
        duration_tolerance,
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


def _measure_active_speech_ratio(waveform, sampling_rate: int) -> float:
    """Estimate speech activity using frame RMS instead of per-sample zeros."""
    duration = _audio_duration(waveform, sampling_rate)
    if duration <= 0:
        return 0.0
    intervals = _active_intervals_from_rms(waveform, sampling_rate)
    speech_total = sum(max(0.0, end - start) for start, end in intervals)
    return max(0.0, min(1.0, speech_total / duration))


def _compute_rms(waveform) -> float:
    arr = np.asarray(waveform)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _mono_float32(waveform) -> np.ndarray:
    arr = waveform
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        return np.zeros(0, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=0 if arr.shape[0] <= arr.shape[1] else 1)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _decode_audio_bytes_mono(raw: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    buf = io.BytesIO(raw)
    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T
    except Exception:
        import librosa

        buf.seek(0)
        data, sr = librosa.load(buf, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != target_sr:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=target_sr
        ).numpy()
        sr = target_sr
    return _mono_float32(data), int(sr)


def _frame_rms_profile(
    waveform: np.ndarray,
    sampling_rate: int,
    frame_seconds: float = 0.08,
    hop_seconds: float = 0.04,
) -> list[tuple[float, float, float]]:
    y = _mono_float32(waveform)
    if y.size == 0 or sampling_rate <= 0:
        return []
    frame = max(1, int(sampling_rate * frame_seconds))
    hop = max(1, int(sampling_rate * hop_seconds))
    if y.size < frame:
        return [(0.0, y.size / sampling_rate, _compute_rms(y))]
    out = []
    for start in range(0, y.size - frame + 1, hop):
        end = start + frame
        out.append((start / sampling_rate, end / sampling_rate, _compute_rms(y[start:end])))
    return out


def _active_intervals_from_rms(
    waveform: np.ndarray,
    sampling_rate: int,
    min_duration: float = 0.16,
    merge_gap: float = 0.14,
) -> list[tuple[float, float]]:
    frames = _frame_rms_profile(waveform, sampling_rate)
    if not frames:
        return []
    rms_values = np.asarray([r for _, _, r in frames], dtype=np.float64)
    high = float(np.percentile(rms_values, 90)) if rms_values.size else 0.0
    floor = max(0.003, high * 0.16)
    raw = [(s, e) for s, e, r in frames if r >= floor]
    merged = _merge_time_intervals(raw, gap=merge_gap)
    return [(s, e) for s, e in merged if e - s >= min_duration]


def _concat_intervals(
    waveform: np.ndarray,
    sampling_rate: int,
    intervals: list[tuple[float, float]],
) -> np.ndarray:
    y = _mono_float32(waveform)
    chunks = []
    for start, end in intervals:
        s = max(0, int(start * sampling_rate))
        e = min(y.size, int(end * sampling_rate))
        if e > s:
            chunks.append(y[s:e])
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _basic_voice_feature(waveform: np.ndarray, sampling_rate: int) -> Optional[np.ndarray]:
    y = _mono_float32(waveform)
    if y.size < max(256, int(0.18 * sampling_rate)):
        return None
    frame = max(128, int(0.04 * sampling_rate))
    hop = max(64, int(0.02 * sampling_rate))
    if y.size < frame:
        frame = y.size
    window = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sampling_rate)
    rows = []
    rms_values = []
    for start in range(0, max(1, y.size - frame + 1), hop):
        chunk = y[start : start + frame]
        if chunk.size < frame:
            break
        rms = _compute_rms(chunk)
        rms_values.append(rms)
        spec = np.abs(np.fft.rfft(chunk * window)).astype(np.float64)
        power = spec**2
        total = float(np.sum(power)) + 1e-12
        centroid = float(np.sum(freqs * power) / total)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))
        cumulative = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumulative, total * 0.85, side="left"))
        rolloff_idx = min(max(0, rolloff_idx), len(freqs) - 1)
        rolloff = float(freqs[rolloff_idx])
        zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk)))))
        low = float(np.sum(power[(freqs >= 80) & (freqs < 400)]) / total)
        mid = float(np.sum(power[(freqs >= 400) & (freqs < 1600)]) / total)
        high = float(np.sum(power[(freqs >= 1600) & (freqs < 5000)]) / total)
        flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power) + 1e-12))
        rows.append([rms, centroid, bandwidth, rolloff, zcr, low, mid, high, flatness])
    if not rows:
        return None
    rms_arr = np.asarray(rms_values, dtype=np.float64)
    active_floor = max(0.003, float(np.percentile(rms_arr, 80)) * 0.18)
    active_rows = np.asarray(
        [row for row in rows if row[0] >= active_floor],
        dtype=np.float64,
    )
    if active_rows.size == 0:
        active_rows = np.asarray(rows, dtype=np.float64)
    feat = np.concatenate([np.mean(active_rows, axis=0), np.std(active_rows, axis=0)])
    scale = np.asarray(
        [0.1, 3000.0, 3000.0, 5000.0, 0.5, 1.0, 1.0, 1.0, 1.0] * 2,
        dtype=np.float64,
    )
    feat = np.nan_to_num(feat / scale, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(feat))
    if norm <= 1e-8:
        return None
    return feat / norm


def _voice_feature(waveform: np.ndarray, sampling_rate: int) -> Optional[np.ndarray]:
    y = _mono_float32(waveform)
    if y.size < max(256, int(0.18 * sampling_rate)):
        return None
    try:
        import librosa

        n_fft = min(1024, 2 ** int(np.floor(np.log2(max(256, y.size)))))
        hop_length = max(1, int(0.025 * sampling_rate))
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sampling_rate,
            n_mfcc=20,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length
        )
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    except Exception:
        return _basic_voice_feature(y, sampling_rate)

    stat_parts = [
        np.mean(mfcc[1:], axis=1),
        np.std(mfcc[1:], axis=1),
        np.mean(spec_centroid, axis=1),
        np.std(spec_centroid, axis=1),
        np.mean(zcr, axis=1),
        np.std(zcr, axis=1),
    ]
    feat = np.concatenate(stat_parts).astype(np.float64)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(feat))
    if norm <= 1e-8:
        return None
    return feat / norm


def _mean_voice_feature(features: list[np.ndarray]) -> Optional[np.ndarray]:
    if not features:
        return None
    feat = np.mean(np.stack(features, axis=0), axis=0)
    norm = float(np.linalg.norm(feat))
    if norm <= 1e-8:
        return None
    return feat / norm


def _feature_similarity(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(np.dot(left, right))


def _edge_trim_min_keep_duration(
    original_duration: float,
    text: str,
    ref_text: Optional[str],
    ref_duration: Optional[float],
    target_duration: Optional[float],
) -> float:
    floors = [0.6, original_duration * 0.55]
    if target_duration and target_duration > 0:
        floors.append(min(original_duration * 0.9, target_duration * 0.65))
    if text:
        estimated = _estimate_natural_duration(text, ref_text, ref_duration)
        floors.append(min(original_duration * 0.9, estimated * 0.6))
    return min(original_duration, max(floors))


def _trim_edge_voice_mismatch(
    waveform,
    sampling_rate: int,
    *,
    text: str,
    ref_text: Optional[str],
    ref_duration: Optional[float],
    target_duration: Optional[float],
    ref_audio_bytes: Optional[bytes],
    max_edge_trim_seconds: float = 2.0,
    similarity_threshold: float = 0.86,
    similarity_margin: float = 0.08,
) -> tuple[np.ndarray, Dict[str, Any]]:
    y = _mono_float32(waveform)
    duration = y.size / sampling_rate if sampling_rate > 0 else 0.0
    info: Dict[str, Any] = {
        "enabled": True,
        "status": "skipped",
        "basis": "none",
        "original_duration_sec": _round_float(duration),
        "trim_start_sec": 0.0,
        "trim_end_sec": 0.0,
        "threshold": similarity_threshold,
        "margin": similarity_margin,
        "edge_candidates": [],
    }
    if y.size == 0 or duration < 1.0:
        info["reason"] = "audio_too_short"
        return y, info

    max_edge = max(0.2, min(float(max_edge_trim_seconds), duration * 0.25))
    min_keep = _edge_trim_min_keep_duration(
        duration, text, ref_text, ref_duration, target_duration
    )
    active_intervals = _active_intervals_from_rms(y, sampling_rate)
    if len(active_intervals) < 2:
        info["reason"] = "no_separate_edge_speech"
        info["active_interval_count"] = len(active_intervals)
        return y, info

    ref_feature = None
    if ref_audio_bytes:
        try:
            ref_y, ref_sr = _decode_audio_bytes_mono(ref_audio_bytes, sampling_rate)
            ref_active = _active_intervals_from_rms(ref_y, ref_sr)
            ref_samples = _concat_intervals(ref_y, ref_sr, ref_active) if ref_active else ref_y
            ref_feature = _voice_feature(ref_samples, ref_sr)
        except Exception as exc:
            info["ref_feature_error"] = str(exc)[:300]

    edge_first = active_intervals[0] if active_intervals[0][0] <= 0.35 else None
    edge_last = active_intervals[-1] if active_intervals[-1][1] >= duration - 0.35 else None

    excluded = {edge_first, edge_last}
    main_intervals = [
        interval
        for interval in active_intervals
        if interval not in excluded
        and interval[0] >= 0.0
        and interval[1] <= duration
    ]
    if not main_intervals:
        center_start = duration * 0.25
        center_end = duration * 0.75
        main_intervals = [
            (max(center_start, s), min(center_end, e))
            for s, e in active_intervals
            if min(center_end, e) > max(center_start, s)
        ]
    main_features = [
        _voice_feature(_concat_intervals(y, sampling_rate, [interval]), sampling_rate)
        for interval in main_intervals
    ]
    main_feature = _mean_voice_feature([f for f in main_features if f is not None])
    if main_feature is None:
        info["reason"] = "main_voice_feature_unavailable"
        return y, info

    basis = "ref_wav" if ref_feature is not None else "main_voice"
    info["basis"] = basis
    main_ref_similarity = _feature_similarity(main_feature, ref_feature)
    if main_ref_similarity is not None:
        info["main_ref_similarity"] = _round_float(main_ref_similarity)

    def candidate_is_mismatch(interval: tuple[float, float], side: str) -> tuple[bool, Dict[str, Any]]:
        samples = _concat_intervals(y, sampling_rate, [interval])
        feat = _voice_feature(samples, sampling_rate)
        sim_main = _feature_similarity(feat, main_feature)
        sim_ref = _feature_similarity(feat, ref_feature)
        candidate = {
            "side": side,
            "start": _round_float(interval[0]),
            "end": _round_float(interval[1]),
            "duration": _round_float(interval[1] - interval[0]),
            "similarity_to_main": _round_float(sim_main),
            "similarity_to_ref": _round_float(sim_ref),
        }
        if feat is None or sim_main is None:
            candidate["decision"] = "skip_feature_unavailable"
            return False, candidate
        if interval[1] - interval[0] > max_edge:
            candidate["decision"] = "keep_too_long_for_edge"
            return False, candidate
        if basis == "ref_wav" and sim_ref is not None and main_ref_similarity is not None:
            mismatch = (
                sim_ref < similarity_threshold
                and sim_ref + similarity_margin < main_ref_similarity
                and sim_main < 0.96
            )
        else:
            mismatch = sim_main < similarity_threshold
        candidate["decision"] = "trim" if mismatch else "keep"
        return mismatch, candidate

    trim_start = 0.0
    trim_end = 0.0
    if edge_first is not None:
        mismatch, candidate = candidate_is_mismatch(edge_first, "start")
        info["edge_candidates"].append(candidate)
        if mismatch:
            trim_start = min(max_edge, edge_first[1])
    if edge_last is not None:
        mismatch, candidate = candidate_is_mismatch(edge_last, "end")
        info["edge_candidates"].append(candidate)
        if mismatch:
            trim_end = min(max_edge, duration - edge_last[0])

    if trim_start + trim_end <= 0:
        info["status"] = "pass"
        info["reason"] = "edge_voice_matches_main"
        return y, info
    if duration - trim_start - trim_end < min_keep:
        overflow = min_keep - (duration - trim_start - trim_end)
        if trim_end >= trim_start:
            trim_end = max(0.0, trim_end - overflow)
        else:
            trim_start = max(0.0, trim_start - overflow)
    if duration - trim_start - trim_end < 0.5:
        info["status"] = "skipped"
        info["reason"] = "trim_would_remove_too_much_audio"
        return y, info

    start_sample = int(trim_start * sampling_rate)
    end_sample = y.size - int(trim_end * sampling_rate)
    trimmed = y[start_sample:end_sample].astype(np.float32, copy=False)
    fade_samples = min(int(0.02 * sampling_rate), trimmed.size // 4)
    if fade_samples > 1:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        trimmed = trimmed.copy()
        trimmed[:fade_samples] *= fade_in
        trimmed[-fade_samples:] *= fade_out
    info.update(
        {
            "status": "trimmed" if (trim_start > 0 or trim_end > 0) else "pass",
            "trim_start_sec": _round_float(trim_start),
            "trim_end_sec": _round_float(trim_end),
            "trimmed_duration_sec": _round_float(trimmed.size / sampling_rate),
            "min_keep_duration_sec": _round_float(min_keep),
        }
    )
    return trimmed, info


def _detect_plosive_spikes(
    waveform,
    sampling_rate: int,
    window_seconds: float = 0.005,
    hop_seconds: float = 0.0025,
    min_spike_ratio: float = 0.003,
) -> tuple[list[str], list[dict]]:
    """Detect isolated impulse spikes / plosive artifacts in generated audio.

    OmniVoice sometimes emits short, sharp pops (especially on short words or
    after post-processing limiting) that are not full clipping but still sound
    bad. We flag these by looking for very short windows where the local peak is
    much larger than the local RMS (high crest factor) and the absolute peak is
    significant.

    Thresholds are adaptive to the global signal level so that quiet but clean
    plosives are not mis-flagged.

    Returns:
        (issue_labels, spike_locations) where spike_locations is a list of
        dicts with 'time_sec' and 'crest' for any extreme spike found.
    """
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sampling_rate <= 0:
        return [], []

    global_peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    global_rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0

    # Adaptive thresholds: require a spike to be both relatively loud compared
    # to the global signal and locally imbalanced.
    spike_peak_threshold = max(0.18, global_rms * 3.0, global_peak * 0.35)
    spike_crest_threshold = max(10.0, 12.0)
    extreme_crest_threshold = max(16.0, 20.0)

    window_size = max(1, int(window_seconds * sampling_rate))
    hop_size = max(1, int(hop_seconds * sampling_rate))
    total_windows = 0
    spike_windows = 0
    extreme_windows = 0
    spike_locations = []

    for start in range(0, arr.size, hop_size):
        block = arr[start : start + window_size]
        if block.size == 0:
            continue
        total_windows += 1
        local_peak = float(np.max(np.abs(block)))
        local_rms = float(np.sqrt(np.mean(np.square(block))))
        if local_peak < spike_peak_threshold:
            continue
        crest = local_peak / max(local_rms, 1e-6)
        if crest > extreme_crest_threshold:
            extreme_windows += 1
            spike_locations.append(
                {
                    "time_sec": _round_float(start / sampling_rate),
                    "crest": _round_float(crest),
                    "peak": _round_float(local_peak),
                }
            )
        elif crest > spike_crest_threshold:
            spike_windows += 1

    issues = []
    if total_windows > 0:
        if extreme_windows > 0:
            issues.append("impulsive_spike")
        elif spike_windows / total_windows > min_spike_ratio:
            issues.append("plosive")

    # Also flag globally imbalanced crest factor when the overall level is high enough.
    if global_peak > 0.3 and global_rms > 1e-6 and (global_peak / global_rms) > 15.0 and "plosive" not in issues:
        issues.append("plosive")

    return issues, spike_locations


def _detect_periodic_pulse_artifact(
    waveform,
    sampling_rate: int,
    frame_seconds: float = 0.04,
    hop_seconds: float = 0.01,
) -> tuple[list[str], list[dict]]:
    """Detect rhythmic mechanical/chugging artifacts in generated speech.

    The failure mode sounds like a steady train-like pulse: the waveform is not
    clipped and may have normal duration, but its short-time energy envelope is
    dominated by a strong low-frequency rhythm. Normal speech also has syllabic
    rhythm, so thresholds are intentionally conservative and this detector only
    acts as a retry signal.
    """
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sampling_rate <= 0:
        return [], []
    duration = arr.size / float(sampling_rate)
    if duration < 1.2:
        return [], []

    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    if peak < 0.08 or rms < 0.008:
        return [], []

    frame_size = max(1, int(frame_seconds * sampling_rate))
    hop_size = max(1, int(hop_seconds * sampling_rate))
    if arr.size < frame_size * 8:
        return [], []

    envelope = []
    for start in range(0, arr.size - frame_size + 1, hop_size):
        block = arr[start : start + frame_size]
        envelope.append(float(np.sqrt(np.mean(np.square(block)))))
    env = np.asarray(envelope, dtype=np.float32)
    if env.size < 40:
        return [], []

    env_mean = float(np.mean(env))
    if env_mean <= 1e-6:
        return [], []
    modulation = float(np.std(env) / env_mean)
    if modulation < 0.55:
        return [], []

    centered = env - env_mean
    energy = float(np.dot(centered, centered))
    if energy <= 1e-9:
        return [], []

    autocorr = np.correlate(centered, centered, mode="full")[env.size - 1 :] / energy
    min_lag = max(1, int(0.06 / hop_seconds))  # ~16.7 Hz upper bound
    max_lag = min(len(autocorr) - 1, int(0.25 / hop_seconds))  # ~4 Hz lower bound
    if max_lag <= min_lag:
        return [], []
    band = autocorr[min_lag : max_lag + 1]
    best_offset = int(np.argmax(band))
    periodicity = float(band[best_offset])
    lag = min_lag + best_offset
    pulse_rate_hz = 1.0 / max(lag * hop_seconds, 1e-6)

    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(centered.size, d=hop_seconds)
    valid = freqs > 0.5
    pulse_band = (freqs >= 4.0) & (freqs <= 16.7)
    total_spec = float(np.sum(spectrum[valid])) if np.any(valid) else 0.0
    pulse_spec = float(np.max(spectrum[pulse_band])) if np.any(pulse_band) else 0.0
    pulse_ratio = pulse_spec / max(total_spec, 1e-9)

    if periodicity < 0.68 or pulse_ratio < 0.32:
        return [], []

    return [
        "periodic_pulse",
    ], [
        {
            "type": "periodic_pulse",
            "rate_hz": _round_float(pulse_rate_hz, 2),
            "score": _round_float(periodicity, 3),
            "modulation": _round_float(modulation, 3),
            "spectral_ratio": _round_float(pulse_ratio, 3),
        }
    ]


def _detect_harsh_high_frequency_artifact(
    waveform,
    sampling_rate: int,
) -> list[str]:
    """Detect overly bright/harsh outputs that tend to sound sharp or cracked."""
    arr = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if arr.size == 0 or sampling_rate <= 0:
        return []
    duration = arr.size / float(sampling_rate)
    if duration < 0.4:
        return []

    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    issues = []
    if peak > 0.965:
        issues.append("near_clipping")
    if rms < 0.012 or sampling_rate < 12000:
        return issues

    windowed = arr * np.hanning(arr.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / sampling_rate)
    speech_band = (freqs >= 80.0) & (freqs <= min(sampling_rate / 2.0, 12000.0))
    high_band = (freqs >= 4800.0) & (freqs <= min(sampling_rate / 2.0, 12000.0))
    if not np.any(speech_band) or not np.any(high_band):
        return issues
    total = float(np.sum(spectrum[speech_band]))
    if total <= 1e-12:
        return issues
    high = float(np.sum(spectrum[high_band]))
    centroid = float(np.sum(freqs[speech_band] * spectrum[speech_band]) / total)
    high_ratio = high / total

    # Conservative threshold: normal speech has some sibilance, but a sustained
    # high ratio plus high centroid usually corresponds to brittle/over-bright
    # generated audio or post-processing edge artifacts.
    if high_ratio > 0.34 and centroid > 3600.0 and peak > 0.18:
        issues.append("harsh_high_freq")
    return issues


def _apply_peak_ceiling(waveform, ceiling: float = OUTPUT_PEAK_CEILING):
    arr = np.asarray(waveform, dtype=np.float32)
    if arr.size == 0 or ceiling <= 0:
        return waveform, False
    peak = float(np.max(np.abs(arr)))
    if peak <= ceiling:
        return waveform, False
    scaled = arr * (ceiling / max(peak, 1e-9))
    return scaled.astype(np.float32), True


def _check_audio_quality(
    waveform,
    sampling_rate: int,
    target_duration: Optional[float] = None,
    duration_tolerance: Optional[float] = None,
    ref_duration: Optional[float] = None,
) -> tuple[list[str], list[dict]]:
    """Check generated audio for common badcase patterns.

    Returns:
        (issue_labels, spike_locations); empty lists mean no detected issue.
    """
    issues = []
    arr = np.asarray(waveform)
    duration = _audio_duration(arr, sampling_rate)
    peak = float(np.abs(arr).max()) if arr.size > 0 else 0.0
    rms = _compute_rms(arr)
    silence_ratio = _measure_silence_ratio(arr)
    active_speech_ratio = _measure_active_speech_ratio(arr, sampling_rate)

    if arr.size == 0 or duration < 0.05:
        issues.append("empty")
    if duration >= 0.5 and active_speech_ratio < 0.35 and silence_ratio > 0.65:
        issues.append("too_much_silence")
    # Cross-check with the same frame-energy intervals used in audio_qc; some
    # very short utterances fill the window with near-silent padding and only
    # briefly pop, which the per-sample silence ratio can miss. Only run when
    # the per-sample check did not already flag it — the frame sweep is the
    # expensive part here and normal audio almost never needs it.
    if duration >= 0.5 and "too_much_silence" not in issues:
        speech_intervals = _waveform_speech_intervals(arr, sampling_rate)
        speech_total = sum(max(0.0, end - start) for start, end in speech_intervals)
        frame_speech_ratio = speech_total / duration
        if frame_speech_ratio < 0.30:
            issues.append("too_much_silence")

    if peak > 0.99:
        issues.append("clipping")
    if 0 < rms < 0.005:
        issues.append("too_quiet")

    if target_duration is not None and target_duration > 0:
        tol = duration_tolerance if duration_tolerance is not None else 0.0
        # Flag only when deviation is clearly outside normal model variance.
        # Use a relative threshold (15% of target) so longer cues are not
        # penalized for sub-second drift, plus an absolute floor for short cues.
        if abs(duration - target_duration) > max(tol * 2, target_duration * 0.15, 0.5):
            issues.append("duration_off_target")

    if ref_duration is not None and ref_duration >= MIN_REFERENCE_DURATION_FOR_DURATION_RATIO:
        ratio = duration / ref_duration
        if ratio > 3.0 or ratio < 0.33:
            issues.append("duration_off_reference")

    plosive_issues, spike_locations = _detect_plosive_spikes(arr, sampling_rate)
    issues.extend(plosive_issues)
    pulse_issues, pulse_locations = _detect_periodic_pulse_artifact(arr, sampling_rate)
    issues.extend(pulse_issues)
    spike_locations.extend(pulse_locations)
    issues.extend(_detect_harsh_high_frequency_artifact(arr, sampling_rate))
    return issues, spike_locations


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

    if "too_much_silence" in issues or "empty" in issues or "periodic_pulse" in issues or "harsh_high_freq" in issues:
        # Tighter position sampling to reduce random unmasking of silences.
        fallback["position_temperature"] = max(
            float(fallback.get("position_temperature", 5.0)) * 0.6, 1.0
        )

    if "harsh_high_freq" in issues or "near_clipping" in issues:
        fallback["cfg_value"] = max(float(fallback.get("cfg_value", 2.0)) * 0.85, 1.2)

    if "periodic_pulse" in issues or "harsh_high_freq" in issues:
        fallback["t_shift"] = max(float(fallback.get("t_shift", 0.1)) * 0.6, 0.03)

    if (
        "clipping" in issues
        or "near_clipping" in issues
        or "plosive" in issues
        or "impulsive_spike" in issues
        or "periodic_pulse" in issues
        or "harsh_high_freq" in issues
    ):
        # Disable post-processing in case aggressive trimming/leveling/spike limiting
        # caused the artifact; let the raw diffusion output through.
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
        max_attempts=_DURATION_REFINEMENT_INITIAL_ATTEMPTS,
        voice_clone_prompt=voice_clone_prompt,
        max_duration=max_duration,
        **gen_kwargs,
    )
    issues, _spike_locs = _check_audio_quality(
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
    # Retry with duration refinement when the issue is duration-related or when
    # a target_duration was supplied. For other quality issues we still allow one
    # refinement attempt so the result doesn't drift.
    retry_tolerance = duration_tolerance
    retry_attempts = 2
    if any(i in issues for i in ("duration_off_target", "duration_off_reference")):
        retry_attempts = 3
    audio2, attempts2, log2 = _generate_with_duration_refinement(
        model,
        text,
        target_duration=target_duration,
        duration_tolerance=retry_tolerance,
        max_attempts=retry_attempts,
        voice_clone_prompt=voice_clone_prompt,
        max_duration=max_duration,
        **fallback_kwargs,
    )
    issues2, _spike_locs2 = _check_audio_quality(
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


def _serialize_whisper_segments(segments):
    out = []
    for segment in segments:
        words = []
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            words.append({"start": word.start, "end": word.end, "word": word.word})
        out.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": str(segment.text or "").strip(),
                "words": words,
            }
        )
    return out


def _load_whisper_model_sync(model_name, device, compute_type):
    from faster_whisper import WhisperModel

    WHISPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "loading faster-whisper model=%s device=%s compute_type=%s download_root=%s",
        model_name,
        device,
        compute_type,
        WHISPER_MODEL_DIR,
    )
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(WHISPER_MODEL_DIR),
    )


async def _ensure_whisper_model(model_name, device, compute_type):
    key = (model_name, device, compute_type)
    model = _WHISPER_MODELS.get(key)
    if model is not None:
        _WHISPER_MODELS.move_to_end(key)
        return model
    async with _WHISPER_MODEL_LOCK:
        model = _WHISPER_MODELS.get(key)
        if model is not None:
            _WHISPER_MODELS.move_to_end(key)
            return model
        model = await asyncio.to_thread(
            _load_whisper_model_sync,
            model_name,
            device,
            compute_type,
        )
        _WHISPER_MODELS[key] = model
        while len(_WHISPER_MODELS) > max(1, WHISPER_MAX_MODELS):
            _WHISPER_MODELS.popitem(last=False)
    return model


def _transcribe_whisper_sync(model, audio_path, options):
    language = _whisper_language_code(options.get("language"))

    kwargs = {
        "language": language or None,
        "beam_size": int(float(options.get("beam_size") or 5)),
        "vad_filter": _bool_option(options.get("vad_filter"), True),
        "word_timestamps": _bool_option(options.get("word_timestamps"), True),
        "condition_on_previous_text": _bool_option(
            options.get("condition_on_previous_text"),
            False,
        ),
    }
    kwargs["vad_parameters"] = {
        "threshold": float(options.get("vad_threshold") or 0.4),
        "min_silence_duration_ms": int(float(options.get("vad_min_silence_ms") or 300)),
        "speech_pad_ms": int(float(options.get("vad_speech_pad_ms") or 200)),
    }
    kwargs["no_speech_threshold"] = float(options.get("no_speech_threshold") or 0.6)
    initial_prompt = str(options.get("initial_prompt") or "").strip()
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    segments_iter, info = model.transcribe(str(audio_path), **kwargs)
    segments = _serialize_whisper_segments(list(segments_iter))
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": segments,
    }


def _normalize_language_code(value) -> str:
    code = str(value or "").strip().lower()
    if code == "tl":
        return "fil"
    if "-" in code:
        code = code.split("-", 1)[0]
    return code


def _whisper_language_code(value) -> str:
    code = _normalize_language_code(value)
    if code == "fil":
        return "tl"
    return code


def _text_qc_tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^\w\s]+", " ", str(text or "").lower(), flags=re.UNICODE)
    return [token for token in normalized.split() if token]


def _lcs_coverage(expected_tokens: list[str], actual_tokens: list[str]) -> float:
    if not expected_tokens:
        return 1.0
    if not actual_tokens:
        return 0.0
    previous = [0] * (len(actual_tokens) + 1)
    for expected in expected_tokens:
        current = [0]
        for j, actual in enumerate(actual_tokens, start=1):
            if expected == actual:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1] / max(1, len(expected_tokens))


def _should_run_output_text_qc(data: Dict[str, Any], language, text: str) -> bool:
    override = data.get("output_text_qc")
    if override is not None:
        return _bool_option(override, False)
    code = _normalize_language_code(language)
    if code not in OUTPUT_TEXT_QC_LANGS:
        return False
    return len(_text_qc_tokens(text)) >= OUTPUT_TEXT_QC_MIN_TOKENS


async def _build_output_text_qc(audio_path: str, expected_text: str, language, data: Dict[str, Any]) -> Dict[str, Any]:
    code = _normalize_language_code(language)
    whisper_code = _whisper_language_code(language)
    model_name = str(data.get("output_text_qc_model") or OUTPUT_TEXT_QC_MODEL).strip()
    device = _whisper_device(data.get("output_text_qc_device") or "auto")
    compute_type = str(
        data.get("output_text_qc_compute_type")
        or _default_whisper_compute_type(device)
    ).strip()
    model = await _ensure_whisper_model(model_name, device, compute_type)
    result = await asyncio.to_thread(
        _transcribe_whisper_sync,
        model,
        audio_path,
        {
            "language": whisper_code or None,
            "beam_size": data.get("output_text_qc_beam_size") or 3,
            "vad_filter": data.get("output_text_qc_vad_filter", True),
            "word_timestamps": False,
            "condition_on_previous_text": False,
            "vad_threshold": 0.35,
            "vad_min_silence_ms": 250,
            "vad_speech_pad_ms": 120,
            "no_speech_threshold": 0.5,
            "initial_prompt": expected_text[:200],
        },
    )
    actual_text = " ".join(
        str(segment.get("text") or "").strip()
        for segment in result.get("segments") or []
        if isinstance(segment, dict)
    ).strip()
    expected_tokens = _text_qc_tokens(expected_text)
    actual_tokens = _text_qc_tokens(actual_text)
    coverage = _lcs_coverage(expected_tokens, actual_tokens)
    status = "pass" if coverage >= OUTPUT_TEXT_QC_MIN_COVERAGE else "incomplete"
    source_script_residue = (
        bool(actual_text)
        and _CJK_SCRIPT_RE.search(actual_text) is not None
        and not _language_allows_cjk_prompt(code)
    )
    if source_script_residue:
        status = "incomplete"
    return {
        "version": 1,
        "status": status,
        "language": code,
        "whisper_requested_language": whisper_code,
        "model": model_name,
        "coverage": _round_float(coverage, 3),
        "min_coverage": OUTPUT_TEXT_QC_MIN_COVERAGE,
        "expected_token_count": len(expected_tokens),
        "actual_token_count": len(actual_tokens),
        "actual_text": actual_text[:500],
        "whisper_language": result.get("language"),
        "whisper_language_probability": result.get("language_probability"),
        "source_script_residue": source_script_residue,
    }


def _qc_language_mismatch_triggers_retry(
    text_qc: Optional[Dict[str, Any]],
    expected_language,
) -> bool:
    """Decide whether a text-completeness QC result warrants a regenerate.

    Whisper may transcribe a correct TTS output in the wrong language when the
    text is dominated by proper nouns (names, places) that sound like another
    language. Regenerating on a lone language mismatch would misfire on those
    cases and, with the same seed, produce identical audio. So we only trigger
    when the mismatch coincides with low coverage — i.e. the output is both
    incomplete *and* the detected language drifted, which together make a real
    synthesis error far more likely than a transcription quirk.
    """
    if not text_qc or text_qc.get("status") != "incomplete":
        return False
    coverage = text_qc.get("coverage")
    if coverage is None or coverage >= OUTPUT_TEXT_QC_MIN_COVERAGE:
        return False
    detected = text_qc.get("whisper_language")
    if not detected:
        return False
    expected = _whisper_language_code(expected_language)
    if not expected:
        return False
    return str(detected).lower() != expected.lower()


routes = web.RouteTableDef()


@routes.get("/health")
@routes.get("/api/health")
async def health(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    return _json_response({"ok": True, "service": "voxcpm2_api"})


@routes.get("/v1/models")
async def models(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    return _json_response({
        "ok": True,
        "data": [{
            "id": _API_MODEL_ID,
            "object": "model",
            "loaded": _API_MODEL is not None,
            "device": _API_DEVICE,
        }],
    })


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
        "whisper_models_cached": len(_WHISPER_MODELS),
        "whisper_cached_models": [
            {"model": model, "device": device, "compute_type": compute_type}
            for model, device, compute_type in _WHISPER_MODELS.keys()
        ],
    })


@routes.post("/api/unload")
async def unload(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    global _API_MODEL
    count = 1 if _API_MODEL is not None else 0
    whisper_count = len(_WHISPER_MODELS)
    _API_MODEL = None
    _VOICE_PROMPT_CACHE.clear()
    _WHISPER_MODELS.clear()
    import gc
    gc.collect()
    if sys.platform != "win32":
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return _json_response({"ok": True, "unloaded": count, "whisper_unloaded": whisper_count})


@routes.post("/api/audio_qc/reference")
@routes.post("/api/reference/qc")
async def audio_qc_reference(request):
    """Assess reference-audio quality and optionally run endpoint guard.

    Offloads the CPU-heavy F0/gender endpoint-guard work from the dubbing host.
    """
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")
    try:
        data = await request.json()
    except Exception as exc:
        return _error(f"Invalid JSON body: {exc}", status=400)

    b64_data = data.get("reference_audio_base64") or data.get("prompt_wav_base64") or data.get("audio_base64")
    if not b64_data:
        return _error("reference_audio_base64 or audio_base64 is required")
    try:
        audio_bytes = _decode_base64_audio_to_bytes(b64_data)
    except Exception as exc:
        return _error(f"Failed to decode audio: {exc}")
    if not audio_bytes:
        return _error("Decoded audio is empty")

    run_guard = str(data.get("run_endpoint_guard", "true")).strip().lower() not in {"0", "false", "no", "off"}
    trim_on_guard = str(data.get("trim_on_guard", "false")).strip().lower() in {"1", "true", "yes", "on"}

    quality = _reference_quality_legacy(audio_bytes)
    guard = {}
    trimmed_b64 = None
    if run_guard and quality.get("ok"):
        guard = _reference_endpoint_guard(audio_bytes)
        if trim_on_guard and guard.get("trimmed"):
            try:
                trimmed_bytes = _trim_audio_bytes(
                    audio_bytes, guard.get("start_trim", 0.0), guard.get("end_trim", 0.0)
                )
                trimmed_b64 = base64.b64encode(trimmed_bytes).decode("ascii")
                quality_after = _reference_quality_legacy(trimmed_bytes)
                if quality_after.get("ok"):
                    quality = quality_after
            except Exception as exc:
                guard["trim_error"] = str(exc)[:200]

    payload = {"ok": True, "quality": quality, "guard": guard}
    if trimmed_b64:
        payload["trimmed_audio_base64"] = f"data:audio/wav;base64,{trimmed_b64}"
    return _json_response(payload)


@routes.post("/api/audio_qc/loudness")
async def audio_qc_loudness(request):
    """Return active-gated loudness profile for a small audio clip."""
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")
    try:
        data = await request.json()
    except Exception as exc:
        return _error(f"Invalid JSON body: {exc}", status=400)
    b64_data = data.get("audio_base64")
    if not b64_data:
        return _error("audio_base64 is required")
    try:
        audio_bytes = _decode_base64_audio_to_bytes(b64_data)
    except Exception as exc:
        return _error(f"Failed to decode audio: {exc}")
    frame_seconds = 0.4
    try:
        frame_seconds = max(0.01, float(data.get("frame_seconds", frame_seconds)))
    except (TypeError, ValueError):
        pass
    profile = _build_loudness_profile_from_bytes(audio_bytes, frame_seconds=frame_seconds)
    if profile.get("error"):
        return _error(f"Loudness analysis failed: {profile['error']}", status=502)
    return _json_response({"ok": True, "profile": profile})


@routes.post("/api/audio_qc/speech_intervals")
async def audio_qc_speech_intervals(request):
    """Return speech intervals and ratios for a small audio clip."""
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")
    try:
        data = await request.json()
    except Exception as exc:
        return _error(f"Invalid JSON body: {exc}", status=400)
    b64_data = data.get("audio_base64")
    if not b64_data:
        return _error("audio_base64 is required")
    try:
        audio_bytes = _decode_base64_audio_to_bytes(b64_data)
    except Exception as exc:
        return _error(f"Failed to decode audio: {exc}")
    result = _build_speech_intervals_from_bytes(audio_bytes)
    if result.get("error"):
        return _error(f"Speech interval analysis failed: {result['error']}", status=502)
    return _json_response({"ok": True, **result})


@routes.post("/api/audio_qc/stem_levels")
async def audio_qc_stem_levels(request):
    """Measure active-gated loudness of separated vocal and background stems.

    The dubbing host uses these levels to align dubbed voice and background to
    the original mix instead of relying on fixed volume factors.
    """
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")
    try:
        data = await request.json()
    except Exception as exc:
        return _error(f"Invalid JSON body: {exc}", status=400)

    def _stem_levels(b64_data):
        if not b64_data:
            return None
        try:
            audio_bytes = _decode_base64_audio_to_bytes(b64_data)
        except Exception:
            return None
        return _build_loudness_profile_from_bytes(audio_bytes)

    vocal_levels = _stem_levels(data.get("vocal_base64") or data.get("vocal_audio_base64"))
    background_levels = _stem_levels(data.get("background_base64") or data.get("background_audio_base64"))
    if vocal_levels is None and background_levels is None:
        return _error("At least one of vocal_base64 / background_base64 is required", status=400)
    voice_target_db = -19.0
    voice_gain = _recommend_gain(vocal_levels, voice_target_db)
    background_gain, gap_basis = _recommend_background_gain_preserve_gap(
        vocal_levels, background_levels, voice_target_db
    )
    return _json_response({
        "ok": True,
        "vocal_levels": vocal_levels,
        "background_levels": background_levels,
        "target_integrated_loudness_db": voice_target_db,
        "recommended_voice_gain": voice_gain,
        "recommended_background_gain": background_gain,
        "mix_basis": gap_basis,
    })


def _active_loudness_db(levels):
    if not isinstance(levels, dict):
        return None
    return levels.get("active_p70_volume_db") or levels.get("active_mean_volume_db") or levels.get("mean_volume_db")


def _recommend_background_gain_preserve_gap(vocal_levels, background_levels, voice_target_db):
    """Suggest a background gain that preserves the original voice-background gap."""
    fallback_background_under_voice_db = 10.0
    minimum_background_under_voice_db = 6.0
    maximum_background_under_voice_db = 14.0
    minimum_background_loudness_db = -32.0
    voice_db = _active_loudness_db(vocal_levels)
    background_db = _active_loudness_db(background_levels)
    basis = {
        "voice_target_db": voice_target_db,
        "voice_active_db": voice_db,
        "background_active_db": background_db,
    }
    if background_db is None:
        basis["fallback_reason"] = "missing_background_levels"
        return _recommend_gain(background_levels, voice_target_db - fallback_background_under_voice_db), basis
    source_gap_db = None
    if voice_db is not None:
        try:
            source_gap_db = float(voice_db) - float(background_db)
        except (TypeError, ValueError):
            pass
    if source_gap_db is None:
        desired_gap_db = fallback_background_under_voice_db
    else:
        desired_gap_db = max(
            minimum_background_under_voice_db,
            min(maximum_background_under_voice_db, source_gap_db),
        )
    target_background_db = max(minimum_background_loudness_db, voice_target_db - desired_gap_db)
    basis["source_voice_background_gap_db"] = source_gap_db
    basis["desired_background_under_voice_db"] = desired_gap_db
    basis["target_background_db"] = target_background_db
    return _recommend_gain(background_levels, target_background_db), basis


def _recommend_gain(levels, target_db):
    """Suggest a linear gain so the stem lands near target_db integrated loudness."""
    if not isinstance(levels, dict):
        return None
    mean_db = levels.get("mean_volume_db")
    active_db = levels.get("active_p70_volume_db") or levels.get("active_mean_volume_db") or levels.get("mean_volume_db")
    if active_db is None:
        return None
    try:
        active_db = float(active_db)
    except (TypeError, ValueError):
        return None
    # Use active loudness as the perceptual anchor; 1 LU ≈ 1 dB.
    db_change = target_db - active_db
    # Clamp to a safe range to avoid extreme boosts/cuts.
    return round(max(0.2, min(5.0, 10 ** (db_change / 20.0))), 4)


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

    language = (
        data.get("language")
        or data.get("target_lang")
        or data.get("target_language")
        or data.get("output_language_code")
    )
    reference_audio_base64 = data.get("reference_audio_base64")
    prompt_wav_base64 = data.get("prompt_wav_base64") or data.get("prompt_audio_base64") or data.get("prompt_wav")
    raw_prompt_text = re.sub(r"\s+", " ", (data.get("prompt_text") or "").strip())
    prompt_text = _sanitize_cross_language_prompt_text(raw_prompt_text, language)
    if raw_prompt_text and not prompt_text:
        logger.info(
            f"[{req_id}] dropped cross-language prompt_text for target language={language!r}"
        )
    effective_prompt_text = (
        prompt_text if (reference_audio_base64 or prompt_wav_base64) else ""
    )

    # Optional same-speaker alternate references: server can select strongest.
    alternate_refs = data.get("alternate_reference_audio_base64") or []
    alternate_texts = data.get("alternate_prompt_texts") or []
    if isinstance(alternate_refs, str):
        alternate_refs = [alternate_refs]
    if isinstance(alternate_texts, str):
        alternate_texts = [alternate_texts]
    alternate_texts = [
        _sanitize_cross_language_prompt_text(str(text or ""), language)
        for text in alternate_texts
    ]

    # Get user-specified values (None means use adaptive defaults)
    user_cfg = data.get("cfg_value")
    user_steps = data.get("inference_timesteps")
    denoise = _bool_option(data.get("denoise"), True)
    optimize = _bool_option(data.get("optimize"), False)
    target_duration_ms = data.get("target_duration_ms")
    max_duration_ms = data.get("max_duration_ms")
    duration_tolerance_ms = data.get("duration_tolerance_ms")
    requested_max_duration_ms = max_duration_ms
    duration_cap_relaxed = False
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

    # Tighten tolerance for short cues so that sub-second utterances do not drift
    # by a large fraction of their window.
    if duration_tolerance_sec is not None and duration_tolerance_sec > 0:
        if target_duration_sec is not None and target_duration_sec > 0 and target_duration_sec < 2.0:
            duration_tolerance_sec = min(duration_tolerance_sec, max(0.03, target_duration_sec * 0.05))

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

    # Assess reference audio quality before choosing generation profile.
    ref_quality = _assess_reference_quality(ref_audio_bytes or prompt_audio_bytes)
    if ref_quality.get("is_poor"):
        logger.warning(
            f"[{req_id}] Reference quality issues: {ref_quality.get('issues')}, "
            f"duration={ref_quality.get('duration')}, active_ratio={ref_quality.get('active_ratio')}, "
            f"peak={ref_quality.get('peak')}, snr={ref_quality.get('snr_db')}"
        )

    # If alternate references are provided, score them and use the best one.
    if alternate_refs:
        current_ref_bytes = ref_audio_bytes or prompt_audio_bytes
        best_alt, best_quality, best_text, best_score, primary_score = _select_best_reference(
            current_ref_bytes,
            ref_quality,
            alternate_refs,
            alternate_texts,
        )
        if best_alt is not None and best_alt is not current_ref_bytes:
            best_issues = set((best_quality or {}).get("issues") or [])
            fatal_alt_issues = {"mostly_silence", "low_activity", "too_quiet", "clipping"}
            clean_alt = best_quality is None or not best_quality.get("is_poor", True)
            materially_better_alt = (
                best_score >= primary_score + 0.75
                and not (best_issues & fatal_alt_issues)
            )
            if clean_alt or materially_better_alt:
                logger.info(
                    f"[{req_id}] Selected alternate reference: "
                    f"duration={best_quality.get('duration')}, issues={best_quality.get('issues')}, "
                    f"score={best_score:.2f}, primary_score={primary_score:.2f}"
                )
                ref_audio_bytes = best_alt
                ref_duration = _bytes_audio_duration(best_alt)
                effective_prompt_text = best_text or effective_prompt_text
                ref_quality = best_quality or ref_quality
            else:
                logger.info(
                    f"[{req_id}] Keeping primary reference: "
                    f"best_alt_issues={(best_quality or {}).get('issues')}, "
                    f"best_score={best_score:.2f}, primary_score={primary_score:.2f}"
                )

    # Get adaptive parameters based on reference audio duration and quality
    adaptive_params = _get_adaptive_params(
        ref_duration=ref_duration,
        user_cfg=float(user_cfg) if user_cfg is not None else None,
        user_steps=int(user_steps) if user_steps is not None else None,
        ref_quality=ref_quality,
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
            f"OMNIVOICE_ENFORCE_MAX_DURATION is not set; relaxing max_duration_ms "
            f"for synthesis and returning full audio."
        )
        max_duration_sec = None
        max_duration_ms = None
        duration_cap_relaxed = True

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
        # Only borrow reference duration when it is close to the target text's
        # natural duration. A long reference with short target text otherwise
        # tends to produce long silence and quality retries.
        min_match_duration = max(estimated_natural_duration * 0.85, estimated_natural_duration - 0.35)
        max_match_duration = max(
            estimated_natural_duration * 1.6,
            estimated_natural_duration + 1.0,
        )
        ref_quality_issues = set((ref_quality or {}).get("issues") or [])
        weak_ref_too_short_for_text = (
            "low_snr" in ref_quality_issues
            and estimated_natural_duration > ref_duration + max(0.12, estimated_natural_duration * 0.05)
        )
        if weak_ref_too_short_for_text:
            effective_duration = estimated_natural_duration
            logger.info(
                f"[{req_id}] Skipping ref_duration={ref_duration}s "
                f"(low_snr ref and text needs est_natural={estimated_natural_duration:.1f}s), "
                f"using estimated natural duration={effective_duration:.3f}s"
            )
        elif min_match_duration <= ref_duration <= max_match_duration:
            effective_duration = ref_duration
            logger.info(
                f"[{req_id}] Using ref_duration={ref_duration}s as target "
                f"(text_len={len(text)}, est_natural={estimated_natural_duration:.1f}s)"
            )
        elif ref_duration > max_match_duration:
            effective_duration = estimated_natural_duration
            logger.info(
                f"[{req_id}] Skipping ref_duration={ref_duration}s "
                f"(too long for text, est_natural={estimated_natural_duration:.1f}s), "
                f"using estimated natural duration={effective_duration:.3f}s"
            )
        else:
            # Text is too long for ref_duration, use natural estimation to avoid badcase
            effective_duration = estimated_natural_duration
            logger.info(
                f"[{req_id}] Skipping ref_duration={ref_duration}s "
                f"(text too long, est_natural={estimated_natural_duration:.1f}s), "
                f"using estimated natural duration={effective_duration:.3f}s"
            )
    elif user_duration is None and ref_duration is not None and user_speed != 1.0:
        # User specified speed, respect it but log for debugging
        logger.info(f"[{req_id}] User specified speed={user_speed}, skipping ref_duration matching")
    elif user_duration is None and target_duration_sec is None:
        effective_duration = estimated_natural_duration
        logger.info(
            f"[{req_id}] Using estimated natural duration={effective_duration:.3f}s "
            f"as target because no target_duration_ms was provided"
        )

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
        f"alt_refs={len(alternate_refs)}, "
        f"ref_duration={ref_duration}s, has_prompt_wav={bool(prompt_wav_base64)}, "
        f"prompt_len={len(effective_prompt_text)}, requested_model={data.get('model_id') or ''}, "
        f"loaded_model={_API_MODEL_ID}, device={_API_DEVICE}, cfg={cfg_value}, "
        f"steps={inference_timesteps}, t_shift={t_shift}, denoise={denoise}, "
        f"layer_penalty={layer_penalty_factor}, pos_temp={position_temperature}, "
        f"class_temp={class_temperature}, duration={effective_duration}, speed={effective_speed}, "
        f"target_ms={target_duration_ms}, max_ms={max_duration_ms}, "
        f"requested_max_ms={requested_max_duration_ms}, cap_relaxed={duration_cap_relaxed}, "
        f"tolerance_ms={duration_tolerance_ms}, seed={seed if seed is not None else '-'}"
    )

    voice_trim_value = data.get("voice_consistency_trim")
    if voice_trim_value is None:
        voice_trim_value = data.get("speaker_consistency_trim")
    if voice_trim_value is None:
        voice_trim_value = data.get("trim_voice_mismatch")
    voice_consistency_trim = _bool_option(
        voice_trim_value,
        str(os.environ.get("OMNIVOICE_VOICE_CONSISTENCY_TRIM", "1")).lower()
        in {"1", "true", "yes", "on"},
    )
    voice_consistency_max_edge_trim_sec = (
        float(
            data.get(
                "voice_consistency_max_edge_trim_ms",
                os.environ.get("OMNIVOICE_VOICE_CONSISTENCY_MAX_EDGE_TRIM_MS", "2000"),
            )
        )
        / 1000.0
    )
    voice_consistency_threshold = float(
        data.get(
            "voice_consistency_threshold",
            os.environ.get("OMNIVOICE_VOICE_CONSISTENCY_THRESHOLD", "0.86"),
        )
    )
    voice_consistency_margin = float(
        data.get(
            "voice_consistency_margin",
            os.environ.get("OMNIVOICE_VOICE_CONSISTENCY_MARGIN", "0.08"),
        )
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
                "requested_max_duration_ms": requested_max_duration_ms,
                "duration_cap_relaxed": duration_cap_relaxed,
                "duration_tolerance_ms": duration_tolerance_ms,
                "seed": seed,
                "voice_consistency_trim": voice_consistency_trim,
                "voice_consistency_max_edge_trim_sec": voice_consistency_max_edge_trim_sec,
                "voice_consistency_threshold": voice_consistency_threshold,
                "voice_consistency_margin": voice_consistency_margin,
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
                prompt_for_clone = effective_prompt_text
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
            # Re-run QC on the final waveform to capture spike locations after any
            # model-side trimming (voice_consistency_trim / max_duration clamp).
            _final_issues, spike_locations = _check_audio_quality(
                audio_waveform,
                model.sampling_rate,
                target_duration=effective_duration,
                duration_tolerance=duration_tolerance_sec,
                ref_duration=ref_duration,
            )
            # Preserve any issues already flagged by quality retry.
            quality_issues = sorted(set(quality_issues) | set(_final_issues))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Synthesis failed: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Synthesis failed: {exc}\n{tb}", status=502)

    voice_consistency_info: Dict[str, Any] = {"enabled": False}
    if voice_consistency_trim:
        try:
            audio_waveform, voice_consistency_info = await asyncio.to_thread(
                _trim_edge_voice_mismatch,
                audio_waveform,
                model.sampling_rate,
                text=text,
                ref_text=prompt_text or effective_prompt_text or None,
                ref_duration=ref_duration,
                target_duration=effective_duration,
                ref_audio_bytes=ref_audio_bytes or prompt_audio_bytes,
                max_edge_trim_seconds=voice_consistency_max_edge_trim_sec,
                similarity_threshold=voice_consistency_threshold,
                similarity_margin=voice_consistency_margin,
            )
            logger.info(
                f"[{req_id}] voice consistency trim: "
                f"status={voice_consistency_info.get('status')}, "
                f"basis={voice_consistency_info.get('basis')}, "
                f"trim_start={voice_consistency_info.get('trim_start_sec')}, "
                f"trim_end={voice_consistency_info.get('trim_end_sec')}"
            )
        except Exception as exc:
            logger.warning(f"[{req_id}] Voice consistency trim failed: {exc}")
            voice_consistency_info = {
                "enabled": True,
                "status": "error",
                "error": str(exc)[:500],
            }

    # Hard-trim to max_duration_ms as a last-resort guard against overlapping
    # subsequent cues in the downstream dubbing pipeline. This runs after all
    # model-side refinement and voice-consistency trimming so it only clips
    # genuinely over-long outputs, not model artifacts.
    audio_waveform, was_trimmed = _clamp_waveform_to_max_duration(
        audio_waveform, model.sampling_rate, max_duration_sec
    )
    if was_trimmed:
        logger.warning(
            f"[{req_id}] Output trimmed to max_duration_ms={max_duration_ms} "
            f"({max_duration_sec:.3f}s) to prevent downstream overlap"
        )
        quality_issues = list(quality_issues)
        if "duration_off_target" not in quality_issues:
            quality_issues.append("duration_off_target")

    audio_waveform, peak_limited = _apply_peak_ceiling(audio_waveform, OUTPUT_PEAK_CEILING)
    if peak_limited:
        logger.info(
            f"[{req_id}] output peak ceiling applied: ceiling={OUTPUT_PEAK_CEILING:.3f}"
        )

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

    text_completeness_qc = None
    if output_path_for_response and _should_run_output_text_qc(data, language, text):
        try:
            text_completeness_qc = await _build_output_text_qc(
                output_path_for_response,
                text,
                language,
                data,
            )
            if text_completeness_qc.get("status") == "incomplete":
                quality_issues = list(quality_issues)
                if "text_incomplete" not in quality_issues:
                    quality_issues.append("text_incomplete")
                if text_completeness_qc.get("source_script_residue") and "source_script_residue" not in quality_issues:
                    quality_issues.append("source_script_residue")
                logger.warning(
                    f"[{req_id}] Output text completeness issue: "
                    f"coverage={text_completeness_qc.get('coverage')}, "
                    f"expected_tokens={text_completeness_qc.get('expected_token_count')}, "
                    f"actual_tokens={text_completeness_qc.get('actual_token_count')}"
                )
        except Exception as exc:
            logger.warning(f"[{req_id}] Output text completeness QC failed: {exc}")
            text_completeness_qc = {
                "version": 1,
                "status": "error",
                "error": str(exc)[:500],
            }

    elapsed = round(time.time() - start_time, 3)
    audio_duration = round(audio_waveform.shape[-1] / model.sampling_rate, 3)
    audio_qc = None
    severe_issues = sorted({i for i in (quality_issues or []) if i in _SEVERE_ISSUE_LABELS})
    if _bool_option(data.get("include_audio_qc"), True):
        try:
            audio_qc = _build_synth_audio_qc(
                audio_waveform,
                model.sampling_rate,
                quality_issues=quality_issues,
                spike_locations=spike_locations,
            )
            if text_completeness_qc is not None:
                audio_qc["text_completeness"] = text_completeness_qc
            audio_qc["peak_limited"] = peak_limited
            audio_qc["peak_ceiling"] = OUTPUT_PEAK_CEILING
            audio_qc["severe_issues"] = severe_issues
        except Exception as exc:
            logger.warning(f"[{req_id}] Failed to build synthesis audio_qc: {exc}")
            audio_qc = {
                "version": 1,
                "status": "error",
                "error": str(exc)[:500],
                "severe_issues": severe_issues,
            }
    logger.info(
        f"[{req_id}] synthesis finished in {elapsed}s, output_size={len(wav_bytes)} bytes, "
        f"audio_duration={audio_duration}s, duration_attempts={attempts_made}, "
        f"quality_issues={quality_issues}, quality_retried={quality_retried}, severe_issues={severe_issues}"
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
        "requested_max_duration_ms": requested_max_duration_ms,
        "duration_cap_relaxed": duration_cap_relaxed,
        "duration_tolerance_ms": duration_tolerance_ms,
        "seed": seed,
        "duration_attempts": attempts_made,
        "duration_refinement_log": attempt_log,
        "quality_issues": quality_issues,
        "quality_retried": quality_retried,
        "severe_issues": severe_issues,
        "audio_qc": audio_qc or {},
        "voice_consistency_trim": voice_consistency_info,
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


# ---------------------------------------------------------------------------
# VoxCPM2 routes: status / unload / voices registry / synthesize
# ---------------------------------------------------------------------------

@routes.get("/api/voxcpm/status")
async def status_voxcpm(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    cached_models = []
    if _VOXCPM_MODEL is not None:
        cached_models.append({
            "model_id": _VOXCPM_MODEL_ID,
            "device": _API_DEVICE,
            "load_denoiser": VOXCPM_LOAD_DENOISER and _voxcpm_denoise_available(),
            "optimize": VOXCPM_OPTIMIZE,
        })
    return _json_response({
        "ok": True,
        "engine": "voxcpm2",
        "models_cached": len(cached_models),
        "cached_models": cached_models,
        "voices_cached": len(_VOXCPM_VOICES),
        "voices_cache_size": VOXCPM_VOICES_CACHE_SIZE,
        "exclusive_mode": _EXCLUSIVE_MODE,
    })


@routes.post("/api/voxcpm/unload")
async def unload_voxcpm(request):
    logger.info(f"[{request.method}] {request.path} from {request.remote}")
    count = _unload_voxcpm_model_sync()
    return _json_response({"ok": True, "engine": "voxcpm2", "unloaded": count})


@routes.get("/api/voxcpm/voices")
async def voxcpm_voices_list(request):
    return _json_response({
        "ok": True,
        "voices": [_voxcpm_voice_meta(e) for e in _VOXCPM_VOICES.values()],
        "count": len(_VOXCPM_VOICES),
        "cache_size": VOXCPM_VOICES_CACHE_SIZE,
    })


@routes.get("/api/voxcpm/voices/{voice_id}")
async def voxcpm_voices_get(request):
    voice_id = request.match_info.get("voice_id", "")
    entry = _VOXCPM_VOICES.get(voice_id)
    if entry is None:
        return _error(f"voice_id not found (evicted or never registered): {voice_id}", status=404)
    return _json_response({"ok": True, **_voxcpm_voice_meta(entry)})


@routes.post("/api/voxcpm/voices")
async def voxcpm_voices_register(request):
    req_id = uuid.uuid4().hex[:8]
    client_ip = request.remote or "-"
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")
    try:
        data = await request.json()
    except Exception as exc:
        return _error(f"Invalid JSON body: {exc}", status=400)

    ref_b64 = data.get("reference_audio_base64")
    if not ref_b64:
        return _error("reference_audio_base64 is required")
    try:
        reference_bytes = _decode_base64_audio_bytes(ref_b64)
    except Exception as exc:
        return _error(f"Failed to decode reference audio: {exc}")
    if not reference_bytes:
        return _error("Decoded reference audio is empty")

    prompt_bytes = None
    prompt_b64 = data.get("prompt_wav_base64") or data.get("prompt_audio_base64")
    if prompt_b64:
        try:
            prompt_bytes = _decode_base64_audio_bytes(prompt_b64)
        except Exception as exc:
            return _error(f"Failed to decode prompt audio: {exc}")
    prompt_text = str(data.get("prompt_text") or "")

    voice_id, entry = _register_voxcpm_voice(reference_bytes, prompt_bytes, prompt_text)
    logger.info(f"[{req_id}] registered voice_id={voice_id} ref_ms={entry['reference_duration_ms']}")
    return _json_response({"ok": True, "voice_id": voice_id, **_voxcpm_voice_meta(entry)})


@routes.delete("/api/voxcpm/voices/{voice_id}")
async def voxcpm_voices_delete(request):
    voice_id = request.match_info.get("voice_id", "")
    removed = _VOXCPM_VOICES.pop(voice_id, None) is not None
    return _json_response({"ok": True, "voice_id": voice_id, "removed": removed})


def _voxcpm_adaptive_params(
    cfg_value: float,
    inference_timesteps: int,
    ref_duration: Optional[float],
    ref_quality: Optional[Dict[str, Any]],
) -> tuple[float, int, str]:
    """Lightweight VoxCPM2 cfg/steps adaptation by reference quality.

    VoxCPM2 only accepts ``cfg_value`` and ``inference_timesteps`` (no
    t_shift / layer_penalty), so we adapt just those two. Strong references
    keep the caller's values; weak references nudge cfg up for clone stability
    and add a few diffusion steps. User-specified higher values are never
    downgraded (we take the max), so this only makes generation more careful,
    never sloppier.

    Returns ``(cfg_value, inference_timesteps, reason)``.
    """
    # Weak reference: short, low-activity, low-snr, or flagged poor.
    is_poor = bool(ref_quality and ref_quality.get("is_poor"))
    ref_short = ref_duration is not None and ref_duration < 2.0
    low_snr = bool(
        ref_quality
        and ref_quality.get("snr_reliable")
        and (ref_quality.get("snr_db") or 999) < 10.0
    )
    if not (is_poor or ref_short or low_snr):
        return cfg_value, inference_timesteps, ""

    adaptive_cfg = max(cfg_value, 2.3)
    # Only nudge steps up toward the cap; a caller's already-higher value wins.
    if inference_timesteps < 14:
        adaptive_steps = 14
    else:
        adaptive_steps = inference_timesteps
    reason = (
        f"weak reference (poor={is_poor}, short={ref_short}, low_snr={low_snr}); "
        f"cfg {cfg_value}->{adaptive_cfg}, steps {inference_timesteps}->{adaptive_steps}"
    )
    return adaptive_cfg, adaptive_steps, reason


def _detect_prompt_leak(
    generated_waveform,
    sample_rate: int,
    prompt_audio_bytes: bytes,
    min_leak_sec: float = 0.15,
    check_sec: float = 2.5,
    max_leak_sec: float = 0.6,
    drop_sustain_sec: float = 0.15,
):
    """Detect whether the start of generated audio echoes the prompt (source) tail.

    In VoxCPM continuation mode the model sometimes re-produces the prompt
    audio's tail in the first generated patches; the built-in 3-patch context
    trim does not remove this echo.

    Cloned output naturally shares the prompt's timbre, so an *absolute*
    similarity threshold would flag the whole clip (false positives that trim
    real speech). Instead we use a *relative* test: a real echo makes the
    leading frames stand out well above the clip's own baseline similarity.
    Uniform timbre similarity (the common case) produces no such peak, so the
    detector stays silent and adds no overhead.

    Returns ``(leak_detected, leak_samples)``. ``leak_samples`` is capped at
    ``max_leak_sec`` so a misfire can never remove more than a short prefix.
    """
    try:
        gen = np.asarray(generated_waveform, dtype=np.float32).reshape(-1)
        prompt_wav, _ = _decode_audio_bytes_mono(prompt_audio_bytes, int(sample_rate))
    except Exception:
        return False, 0
    if gen.size < int(0.3 * sample_rate) or prompt_wav.size < int(0.3 * sample_rate):
        return False, 0

    import librosa

    prompt_tail = prompt_wav[-int(1.5 * sample_rate):]
    gen_head = gen[:int(check_sec * sample_rate)]
    n_fft = min(1024, 2 ** int(np.floor(np.log2(max(256, min(gen_head.size, prompt_tail.size))))))
    hop_length = max(1, int(0.025 * sample_rate))

    try:
        gen_mel = librosa.feature.melspectrogram(
            y=gen_head, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=32,
        )
        prompt_mel = librosa.feature.melspectrogram(
            y=prompt_tail, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=32,
        )
    except Exception:
        return False, 0

    gen_mel = librosa.power_to_db(gen_mel + 1e-10)
    prompt_mel = librosa.power_to_db(prompt_mel + 1e-10)

    def _l2_norm_frames(m):
        f = m.T  # [frames, mels]
        norm = np.linalg.norm(f, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return f / norm

    gen_frames = _l2_norm_frames(gen_mel)
    prompt_frames = _l2_norm_frames(prompt_mel)
    # max cosine similarity per generated frame against any prompt-tail frame
    sim = gen_frames @ prompt_frames.T
    max_sim = sim.max(axis=1)  # [gen_frames]

    # Baseline = median similarity of the clip's second half (the echo, if any,
    # is at the start, so the tail is clean). A frame counts as echo only when
    # it clearly exceeds this baseline — uniform timbre similarity never does.
    half = max_sim.shape[0] // 2
    baseline = float(np.median(max_sim[half:])) if half > 0 else float(np.median(max_sim))
    sim_threshold = max(0.82, baseline + 0.20)

    high = max_sim > sim_threshold
    drop_sustain_frames = max(1, int(drop_sustain_sec * sample_rate / hop_length))
    leak_frames = 0
    for i in range(len(high)):
        if high[i]:
            leak_frames = i + 1
        elif i - leak_frames >= drop_sustain_frames:
            break

    leak_samples = int(leak_frames * hop_length)
    leak_samples = min(leak_samples, int(max_leak_sec * sample_rate))
    leak_detected = leak_samples > int(min_leak_sec * sample_rate)
    return leak_detected, leak_samples


def _generate_voxcpm_sync(
    model,
    text,
    ref_path,
    prompt_path,
    prompt_text,
    cfg_value,
    inference_timesteps,
    min_len,
    max_len,
    normalize,
    denoise,
    retry_badcase,
    retry_badcase_max_times,
    retry_badcase_ratio_threshold,
    trim_silence_vad,
    seed,
):
    """Run VoxCPM.generate in a worker thread. Returns a 1-D numpy waveform."""
    kwargs = {
        "text": text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "min_len": int(min_len),
        "max_len": int(max_len),
        "normalize": bool(normalize),
        "denoise": bool(denoise),
        "retry_badcase": bool(retry_badcase),
        "retry_badcase_max_times": int(retry_badcase_max_times),
        "retry_badcase_ratio_threshold": float(retry_badcase_ratio_threshold),
        "trim_silence_vad": bool(trim_silence_vad),
        "seed": seed,
    }
    if ref_path is not None:
        kwargs["reference_wav_path"] = ref_path
    if prompt_path is not None and prompt_text:
        kwargs["prompt_wav_path"] = prompt_path
        kwargs["prompt_text"] = prompt_text
    wav = model.generate(**kwargs)
    arr = np.asarray(wav).reshape(-1).astype(np.float32)
    return arr


@routes.post("/api/voxcpm/synthesize")
async def synthesize_voxcpm(request):
    """VoxCPM2 synthesis. Mirrors the OmniVoice /api/synthesize contract.

    Dual-mode cloning: ``reference_audio_base64`` (or ``voice_id``) drives timbre
    cloning; ``prompt_wav_base64`` + ``prompt_text`` drive exact (ultimate) cloning.
    The caller (dubbing VoxCPMBackend) already sends these as two separate fields.
    """
    req_id = uuid.uuid4().hex[:8]
    client_ip = request.remote or "-"
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")
    start_time = time.time()

    try:
        data = await request.json()
    except Exception as exc:
        return _error(f"Invalid JSON body: {exc}", status=400)

    text = re.sub(r"\s+", " ", (data.get("text") or "").strip())
    if not text:
        return _error("text is required and cannot be empty")
    if len(text) > MAX_TEXT_LEN:
        return _error(f"text exceeds max length {MAX_TEXT_LEN}")

    out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve reference audio: voice_id (cached) takes precedence, else base64.
    voice_id = str(data.get("voice_id") or "").strip()
    ref_audio_bytes = None
    prompt_audio_bytes = None
    prompt_text = str(data.get("prompt_text") or "")

    if voice_id:
        entry = _VOXCPM_VOICES.get(voice_id)
        if entry is None:
            return _error(
                f"voice_id not found (evicted or never registered): {voice_id}",
                status=404,
            )
        ref_audio_bytes = entry.get("reference_bytes")
        # Fall back to the registered prompt only if the request doesn't override.
        if not data.get("prompt_wav_base64") and entry.get("prompt_bytes"):
            prompt_audio_bytes = entry["prompt_bytes"]
            if not prompt_text:
                prompt_text = entry.get("prompt_text") or ""
    else:
        ref_b64 = data.get("reference_audio_base64")
        if ref_b64:
            try:
                ref_audio_bytes = _decode_base64_audio_bytes(ref_b64)
            except Exception as exc:
                return _error(f"Failed to decode reference audio: {exc}")

    # Prompt (ultimate cloning) audio + transcript — distinct from reference.
    if prompt_audio_bytes is None:
        prompt_b64 = data.get("prompt_wav_base64") or data.get("prompt_audio_base64")
        if prompt_b64 and prompt_b64 != data.get("reference_audio_base64"):
            try:
                prompt_audio_bytes = _decode_base64_audio_bytes(prompt_b64)
            except Exception:
                prompt_audio_bytes = None

    # prompt_wav without a transcript can't do ultimate cloning; demote to ref.
    if prompt_audio_bytes and not prompt_text:
        if ref_audio_bytes is None:
            ref_audio_bytes = prompt_audio_bytes
        prompt_audio_bytes = None

    ref_duration = _bytes_audio_duration(ref_audio_bytes) if ref_audio_bytes else None

    # Continuation (ultimate cloning) mode sanity check: if the prompt transcript
    # does not match the prompt audio (e.g. an inaccurate source-line
    # transcription), continuation cloning introduces badcase — the model tries
    # to align mismatched text/audio. When the prompt text's estimated natural
    # duration diverges too far from the prompt audio duration, demote to
    # reference-only cloning (drop prompt_wav/prompt_text) so only the timbre
    # anchor is used. The dubbing emotion_prompt path is the main caller.
    prompt_demoted_to_ref = False
    if prompt_audio_bytes and prompt_text:
        prompt_wav_duration = _bytes_audio_duration(prompt_audio_bytes)
        if prompt_wav_duration and prompt_wav_duration > 0:
            prompt_text_natural = _estimate_natural_duration(prompt_text, None, None)
            ratio = prompt_text_natural / prompt_wav_duration
            # Allow a wide band: transcription may trim filler words or split
            # mid-phrase. Only reject clear mismatches (<0.35 or >3.0).
            if ratio < 0.35 or ratio > 3.0:
                logger.warning(
                    f"[{req_id}] voxcpm prompt/text mismatch "
                    f"(prompt_text~{prompt_text_natural:.2f}s vs prompt_wav={prompt_wav_duration:.2f}s "
                    f"ratio={ratio:.2f}); demoting to reference-only cloning"
                )
                if ref_audio_bytes is None:
                    ref_audio_bytes = prompt_audio_bytes
                    ref_duration = prompt_wav_duration
                prompt_audio_bytes = None
                prompt_text = ""
                prompt_demoted_to_ref = True

    # Generation parameters (VoxCPM2 defaults; caller may override).
    cfg_value = float(data.get("cfg_value", 2.0))
    inference_timesteps = int(data.get("inference_timesteps", 10))
    min_len = int(data.get("min_len", 2))
    max_len = int(data.get("max_len", 4096))
    retry_badcase = _bool_option(data.get("retry_badcase"), True)
    retry_badcase_max_times = int(data.get("retry_badcase_max_times", 3))
    retry_badcase_ratio_threshold = float(data.get("retry_badcase_ratio_threshold", 6.0))
    trim_silence_vad = _bool_option(data.get("trim_silence_vad"), True)

    # Lightweight cfg/steps adaptation by reference quality. On by default;
    # a caller can disable it with ``voxcpm_adaptive=false``. Only nudges
    # values up (never downgrades), so user-specified higher values win.
    voxcpm_adaptive = _bool_option(data.get("voxcpm_adaptive"), True)
    voxcpm_adaptive_reason = ""
    if voxcpm_adaptive and ref_audio_bytes:
        ref_quality_profile = _assess_reference_quality(ref_audio_bytes)
        cfg_value, inference_timesteps, voxcpm_adaptive_reason = _voxcpm_adaptive_params(
            cfg_value, inference_timesteps, ref_duration, ref_quality_profile,
        )
        if voxcpm_adaptive_reason:
            logger.info(f"[{req_id}] voxcpm adaptive: {voxcpm_adaptive_reason}")

    normalize = _bool_option(data.get("normalize"), False)
    if normalize and not _voxcpm_normalize_available():
        logger.warning(f"[{req_id}] normalize requested but wetext/inflect not installed; skipping")
        normalize = False
    denoise = _bool_option(data.get("denoise"), False)
    if denoise and not (VOXCPM_LOAD_DENOISER and _voxcpm_denoise_available()):
        if denoise:
            logger.warning(f"[{req_id}] denoise requested but modelscope/denoiser not available; skipping")
        denoise = False

    quality_retry = _bool_option(data.get("quality_retry"), True)
    quality_retry_max = int(data.get("quality_retry_max", 2))
    # control_instruction is accepted for API parity but VoxCPM2 has no
    # instruction-following mode, so it is ignored.
    _ = data.get("control_instruction") or data.get("instruct")

    target_duration_ms = data.get("target_duration_ms")
    duration_tolerance_ms = data.get("duration_tolerance_ms")
    max_duration_ms = data.get("max_duration_ms")
    try:
        target_duration_ms = int(target_duration_ms) if target_duration_ms is not None else None
    except (TypeError, ValueError):
        target_duration_ms = None
    try:
        duration_tolerance_ms = int(duration_tolerance_ms) if duration_tolerance_ms is not None else None
    except (TypeError, ValueError):
        duration_tolerance_ms = None
    try:
        max_duration_ms = int(max_duration_ms) if max_duration_ms is not None else None
    except (TypeError, ValueError):
        max_duration_ms = None
    target_duration_sec = (target_duration_ms / 1000.0) if target_duration_ms and target_duration_ms > 0 else None
    duration_tolerance_sec = (
        duration_tolerance_ms / 1000.0
        if duration_tolerance_ms and duration_tolerance_ms > 0
        else None
    )
    max_duration_sec = (max_duration_ms / 1000.0) if max_duration_ms and max_duration_ms > 0 else None

    # Relax max_duration when the target text's natural duration exceeds the
    # hard cap — hard-trimming would cut the end of the text off mid-sentence.
    # Mirror the OmniVoice main synth path (see _ENFORCE_MAX_DURATION logic).
    # The dubbing pipeline's local atempo fits the full audio back to the cue
    # window, so returning un-trimmed audio does not cause overlap.
    duration_cap_relaxed = False
    if max_duration_sec is not None:
        estimated_natural_duration = _estimate_natural_duration(
            text, prompt_text or None, ref_duration,
        )
        if estimated_natural_duration > max_duration_sec:
            logger.warning(
                f"[{req_id}] voxcpm relaxing max_duration_ms={max_duration_ms} "
                f"(estimated natural duration {estimated_natural_duration:.2f}s "
                f"exceeds cap); returning full audio for local atempo fit."
            )
            max_duration_sec = None
            max_duration_ms = None
            duration_cap_relaxed = True

    language = (
        data.get("language")
        or data.get("target_lang")
        or data.get("target_language")
        or data.get("output_language_code")
    )

    seed = _stable_voxcpm_seed(
        data, text, prompt_text,
        data.get("reference_audio_base64") or "",
        data.get("prompt_wav_base64") or "",
    )

    # Load model (lazy, serialized).
    # VoxCPM needs at least a reference (timbre clone) or prompt (ultimate clone).
    if ref_audio_bytes is None and prompt_audio_bytes is None:
        return _error(
            "VoxCPM synthesis requires reference_audio_base64, voice_id, "
            "or prompt_wav_base64+prompt_text"
        )

    # Load model (lazy, serialized) — only after we know the request is valid.
    model = await _ensure_voxcpm_model()
    sample_rate = _voxcpm_sample_rate(model)

    # Materialize temp wav paths for VoxCPM (it requires file paths).
    ref_temp_path = None
    prompt_temp_path = None
    ref_path = None
    prompt_path = None
    try:
        if ref_audio_bytes is not None:
            ref_temp_path = out_dir / f"voxcpm_ref_{uuid.uuid4().hex}.wav"
            ref_temp_path.write_bytes(ref_audio_bytes)
            ref_path = str(ref_temp_path)
        if prompt_audio_bytes is not None:
            prompt_temp_path = out_dir / f"voxcpm_prompt_{uuid.uuid4().hex}.wav"
            prompt_temp_path.write_bytes(prompt_audio_bytes)
            prompt_path = str(prompt_temp_path)

        gen_args = (
            model,
            text,
            ref_path,
            prompt_path,
            prompt_text,
            cfg_value,
            inference_timesteps,
            min_len,
            max_len,
            normalize,
            denoise,
            retry_badcase,
            retry_badcase_max_times,
            retry_badcase_ratio_threshold,
            trim_silence_vad,
            seed,
        )

        quality_issues: list = []
        spike_locations: list = []
        quality_retried = False
        attempts = 0

        async with _API_INFER_SEM:
            logger.info(f"[{req_id}] voxcpm synthesis started (cfg={cfg_value}, steps={inference_timesteps})")
            audio_waveform = await asyncio.to_thread(_generate_voxcpm_sync, *gen_args)
            attempts = 1
            _issues, spike_locations = _check_audio_quality(
                audio_waveform, sample_rate,
                target_duration=target_duration_sec,
                duration_tolerance=duration_tolerance_sec,
                ref_duration=ref_duration,
            )
            quality_issues = list(_issues)

            # Retry trigger set: at this stage quality_issues comes only from
            # _check_audio_quality, so duration_off_* (added later by the
            # max_duration clamp) cannot appear — _SEVERE_ISSUE_LABELS is safe.
            severe = [i for i in quality_issues if i in _SEVERE_ISSUE_LABELS]

            # One OmniVoice-style quality retry: bump cfg/steps and regenerate.
            if quality_retry and severe and quality_retry_max >= 1 and "empty" not in severe:
                retry_cfg = min(cfg_value + 0.2, 3.0)
                retry_steps = min(int(inference_timesteps * 1.5), 64)
                retry_args = list(gen_args)
                retry_args[5] = retry_cfg        # cfg_value
                retry_args[6] = retry_steps      # inference_timesteps
                logger.info(
                    f"[{req_id}] voxcpm quality retry: cfg {cfg_value}->{retry_cfg}, "
                    f"steps {inference_timesteps}->{retry_steps} (issues={severe})"
                )
                retry_wav = await asyncio.to_thread(_generate_voxcpm_sync, *retry_args)
                retry_issues, retry_spikes = _check_audio_quality(
                    retry_wav, sample_rate,
                    target_duration=target_duration_sec,
                    duration_tolerance=duration_tolerance_sec,
                    ref_duration=ref_duration,
                )
                retry_severe = [i for i in retry_issues if i in _SEVERE_ISSUE_LABELS]
                quality_retried = True
                attempts = 2
                # Keep whichever output has fewer severe issues.
                if len(retry_severe) < len(severe) or (
                    len(retry_severe) == len(severe) and len(retry_issues) <= len(quality_issues)
                ):
                    audio_waveform = retry_wav
                    quality_issues = list(retry_issues)
                    spike_locations = retry_spikes
                    severe = retry_severe
                # Track the cfg/steps that produced the chosen audio, so the
                # prompt-leak retry below regenerates from the same (possibly
                # raised) parameters instead of the original gen_args.
                leak_base_cfg = retry_cfg
                leak_base_steps = retry_steps
            else:
                leak_base_cfg = cfg_value
                leak_base_steps = inference_timesteps

            # VoxCPM has no direct duration control knob. If the first chosen
            # candidate is clearly off the cue target, try one fresh seed and keep
            # it only when it is less severe and closer to the requested duration.
            if (
                quality_retry
                and target_duration_sec is not None
                and "duration_off_target" in quality_issues
                and quality_retry_max >= 2
            ):
                duration_args = list(gen_args)
                duration_args[-1] = ((seed or 0) + 13) % (2**31 - 1)
                logger.info(
                    f"[{req_id}] voxcpm duration candidate retry "
                    f"(target={target_duration_sec:.3f}s, new seed={duration_args[-1]})"
                )
                duration_wav = await asyncio.to_thread(_generate_voxcpm_sync, *duration_args)
                attempts += 1
                duration_issues, duration_spikes = _check_audio_quality(
                    duration_wav, sample_rate,
                    target_duration=target_duration_sec,
                    duration_tolerance=duration_tolerance_sec,
                    ref_duration=ref_duration,
                )
                current_severe = [i for i in quality_issues if i in _SEVERE_ISSUE_LABELS]
                duration_severe = [i for i in duration_issues if i in _SEVERE_ISSUE_LABELS]
                current_error = _duration_error(audio_waveform, sample_rate, target_duration_sec)
                duration_error = _duration_error(duration_wav, sample_rate, target_duration_sec)
                if len(duration_severe) < len(current_severe) or (
                    len(duration_severe) == len(current_severe)
                    and duration_error + 0.03 < current_error
                ):
                    audio_waveform = duration_wav
                    quality_issues = list(duration_issues)
                    spike_locations = duration_spikes
                    severe = duration_severe
                    logger.info(
                        f"[{req_id}] voxcpm duration candidate accepted "
                        f"(error {current_error:.3f}s->{duration_error:.3f}s, issues={duration_issues})"
                    )
                else:
                    logger.info(
                        f"[{req_id}] voxcpm duration candidate rejected "
                        f"(error {duration_error:.3f}s vs current {current_error:.3f}s, "
                        f"issues={duration_issues})"
                    )

            # Continuation-mode prompt-echo mitigation: the model sometimes
            # re-produces the prompt (source) audio tail in the first generated
            # patches, which the built-in 3-patch context trim cannot remove.
            # The detector uses a relative threshold (see _detect_prompt_leak),
            # so uniform timbre similarity does not trigger it — only a clear
            # echo peak does. One seed retry (reusing the quality-retry-raised
            # cfg/steps and nudging steps up once more — under-diffusion makes
            # prompt-tail echo more likely), then a short capped trim fallback.
            if prompt_path is not None and prompt_audio_bytes is not None:
                leak_detected, leak_samples = _detect_prompt_leak(
                    audio_waveform, sample_rate, prompt_audio_bytes
                )
                if leak_detected:
                    leak_args = list(gen_args)
                    leak_args[-1] = (seed + 1) % (2**31 - 1)  # bump seed
                    leak_args[5] = leak_base_cfg               # keep raised cfg
                    leak_args[6] = min(leak_base_steps + 2, 64)  # one more diffusion step
                    logger.info(
                        f"[{req_id}] voxcpm prompt-leak retry "
                        f"(leak~{leak_samples / sample_rate:.2f}s, new seed={leak_args[-1]})"
                    )
                    leak_wav = await asyncio.to_thread(_generate_voxcpm_sync, *leak_args)
                    attempts += 1
                    retry_detected, retry_leak = _detect_prompt_leak(
                        leak_wav, sample_rate, prompt_audio_bytes
                    )
                    # Keep the retry only if it reduced the echo. leak_detected /
                    # leak_samples now describe the chosen audio_waveform, so the
                    # trim fallback below reuses them instead of re-detecting.
                    if retry_leak < leak_samples:
                        audio_waveform = leak_wav
                        leak_detected = retry_detected
                        leak_samples = retry_leak
                # Fallback: trim any residual echo with a short fade-in.
                # Guard against over-trim (leak >= audio length) so we never
                # produce an empty clip.
                audio_arr = np.asarray(audio_waveform, dtype=np.float32).reshape(-1)
                if (
                    leak_detected
                    and 0 < leak_samples < audio_arr.size - int(0.1 * sample_rate)
                ):
                    fade = min(int(0.02 * sample_rate), leak_samples // 2)
                    audio_arr = audio_arr[leak_samples:].copy()
                    if 1 < fade < audio_arr.size:
                        audio_arr[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                    audio_waveform = audio_arr
                    if "prompt_leak" not in quality_issues:
                        quality_issues.append("prompt_leak")
                    logger.warning(
                        f"[{req_id}] voxcpm prompt-leak trimmed {leak_samples} samples "
                        f"({leak_samples / sample_rate:.2f}s) after retry"
                    )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] VoxCPM synthesis failed: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Synthesis failed: {exc}\n{tb}", status=502)

    # Post-processing (reuse OmniVoice waveform helpers — all model-agnostic).
    audio_waveform, was_trimmed = _clamp_waveform_to_max_duration(
        audio_waveform, sample_rate, max_duration_sec
    )
    if was_trimmed:
        logger.warning(
            f"[{req_id}] voxcpm output trimmed to max_duration_ms={max_duration_ms}"
        )
        if "duration_off_target" not in quality_issues:
            quality_issues.append("duration_off_target")

    audio_waveform, peak_limited = _apply_peak_ceiling(audio_waveform, OUTPUT_PEAK_CEILING)

    try:
        wav_bytes = _waveform_to_wav_bytes(audio_waveform, sample_rate)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Failed to encode voxcpm output: {exc}\n{tb}")
        _cleanup_temp_paths(ref_temp_path, prompt_temp_path)
        return _error(f"Failed to encode output audio: {exc}\n{tb}", status=502)

    key = hashlib.sha256(
        json.dumps({
            "engine": "voxcpm2",
            "text": text,
            "ref_sha": _sha256_text(data.get("reference_audio_base64") or voice_id or ""),
            "prompt_sha": _sha256_text(data.get("prompt_wav_base64") or ""),
            "prompt_text": prompt_text,
            "cfg": cfg_value,
            "steps": inference_timesteps,
            "seed": seed,
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    out_path = out_dir / f"voxcpm2_{key}.wav"
    output_path_for_response = str(out_path.resolve())
    try:
        out_path.write_bytes(wav_bytes)
    except Exception as exc:
        logger.warning(f"[{req_id}] Failed to persist voxcpm output (continuing): {exc}")
        output_path_for_response = ""

    # Output text completeness QC (same language-gated path as OmniVoice).
    text_completeness_qc = None
    if output_path_for_response and _should_run_output_text_qc(data, language, text):
        try:
            text_completeness_qc = await _build_output_text_qc(
                output_path_for_response, text, language, data
            )
            if text_completeness_qc.get("status") == "incomplete":
                if "text_incomplete" not in quality_issues:
                    quality_issues.append("text_incomplete")
        except Exception as exc:
            logger.warning(f"[{req_id}] voxcpm output text QC failed: {exc}")
            text_completeness_qc = {"version": 1, "status": "error", "error": str(exc)[:500]}

    # Language-mismatch regenerate: when the output is both incomplete and
    # whisper detected a different language (a strong signal the synth actually
    # mis-spoke rather than a proper-noun transcription quirk), regenerate once
    # with a bumped seed and keep whichever QC result is better. Shares the
    # quality_retry budget (at most one extra generation).
    lang_regen = False
    if (
        output_path_for_response
        and quality_retry
        and _qc_language_mismatch_triggers_retry(text_completeness_qc, language)
    ):
        regen_args = list(gen_args)
        regen_args[-1] = ((seed or 0) + 7) % (2**31 - 1)  # fresh seed, offset from leak retry
        logger.info(
            f"[{req_id}] voxcpm language-mismatch regenerate "
            f"(detected={text_completeness_qc.get('whisper_language')} "
            f"expected={_whisper_language_code(language)}, "
            f"coverage={text_completeness_qc.get('coverage')}, new seed={regen_args[-1]})"
        )
        try:
            async with _API_INFER_SEM:
                regen_wav = await asyncio.to_thread(_generate_voxcpm_sync, *regen_args)
            attempts += 1
            lang_regen = True
            regen_wav, _ = _clamp_waveform_to_max_duration(regen_wav, sample_rate, max_duration_sec)
            regen_wav, _ = _apply_peak_ceiling(regen_wav, OUTPUT_PEAK_CEILING)
            regen_wav_bytes = _waveform_to_wav_bytes(regen_wav, sample_rate)
            regen_path = out_dir / f"voxcpm2_{key}_langregen.wav"
            try:
                regen_path.write_bytes(regen_wav_bytes)
                regen_qc = await _build_output_text_qc(
                    str(regen_path.resolve()), text, language, data
                )
            finally:
                _cleanup_temp_paths(regen_path)
            # Keep the regenerate only if it lifted coverage above the
            # threshold (a genuine fix); otherwise fall back to the original.
            if regen_qc.get("coverage", 0) >= OUTPUT_TEXT_QC_MIN_COVERAGE:
                audio_waveform = regen_wav
                wav_bytes = regen_wav_bytes
                out_path.write_bytes(wav_bytes)
                if "text_incomplete" in quality_issues:
                    quality_issues.remove("text_incomplete")
                text_completeness_qc = regen_qc
                logger.info(
                    f"[{req_id}] voxcpm language-mismatch regenerate accepted "
                    f"(coverage->{regen_qc.get('coverage')})"
                )
            else:
                logger.info(
                    f"[{req_id}] voxcpm language-mismatch regenerate rejected "
                    f"(coverage {regen_qc.get('coverage')} did not recover)"
                )
        except Exception as exc:
            logger.warning(f"[{req_id}] voxcpm language-mismatch regenerate failed: {exc}")

    elapsed = round(time.time() - start_time, 3)
    audio_duration = round(audio_waveform.shape[-1] / sample_rate, 3)

    severe_issues = sorted({i for i in quality_issues if i in _SEVERE_ISSUE_LABELS})

    audio_qc = None
    if _bool_option(data.get("include_audio_qc"), True):
        try:
            audio_qc = _build_synth_audio_qc(
                audio_waveform, sample_rate,
                quality_issues=quality_issues, spike_locations=spike_locations,
            )
            if text_completeness_qc is not None:
                audio_qc["text_completeness"] = text_completeness_qc
            audio_qc["peak_limited"] = peak_limited
            audio_qc["peak_ceiling"] = OUTPUT_PEAK_CEILING
            audio_qc["severe_issues"] = severe_issues
        except Exception as exc:
            logger.warning(f"[{req_id}] Failed to build voxcpm audio_qc: {exc}")
            audio_qc = {"version": 1, "status": "error", "error": str(exc)[:500],
                        "severe_issues": severe_issues}

    logger.info(
        f"[{req_id}] voxcpm synthesis finished in {elapsed}s, audio_duration={audio_duration}s, "
        f"attempts={attempts}, quality_issues={quality_issues}, "
        f"quality_retried={quality_retried}, severe_issues={severe_issues}"
    )

    output_base64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
    _cleanup_temp_paths(ref_temp_path, prompt_temp_path)

    return _json_response({
        "ok": True,
        "engine": "voxcpm2",
        "audio_base64": output_base64,
        "output_path": output_path_for_response,
        "relative_path": _relative_path(Path(output_path_for_response)) if output_path_for_response else "",
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": audio_duration,
        "target_duration_ms": target_duration_ms,
        "duration_tolerance_ms": duration_tolerance_ms,
        "max_duration_ms": max_duration_ms,
        "duration_cap_relaxed": duration_cap_relaxed,
        "prompt_demoted_to_ref": prompt_demoted_to_ref,
        "voice_id": voice_id or None,
        "seed": seed,
        "duration_attempts": attempts,
        "quality_issues": quality_issues,
        "quality_retried": quality_retried,
        "language_regen": lang_regen,
        "severe_issues": severe_issues,
        "audio_qc": audio_qc or {},
        "duration_match": {
            "ref_duration": ref_duration,
            "target_duration": target_duration_sec,
            "actual_duration": audio_duration,
            "target_delta": round(audio_duration - target_duration_sec, 3) if target_duration_sec and audio_duration else None,
            "match_ratio": round(audio_duration / ref_duration, 3) if ref_duration and audio_duration else None,
        },
        "adaptive_params": {
            "ref_duration": ref_duration,
            "num_step": inference_timesteps,
            "guidance_scale": cfg_value,
            "voxcpm_adaptive": bool(voxcpm_adaptive_reason),
            "voxcpm_adaptive_reason": voxcpm_adaptive_reason,
        },
    })


async def _read_separation_request(request, req_id, out_dir):
    options: Dict[str, Any] = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if request.content_type.startswith("multipart/"):
        reader = await request.multipart()
        input_path = None
        async for field in reader:
            if field.name in {"audio", "file", "video", "input"}:
                filename = _safe_filename(field.filename, "input.wav")
                input_path = out_dir / f"{req_id}_{filename}"
                with input_path.open("wb") as f:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                value = await field.text()
                if field.name:
                    options[field.name] = value
        if input_path is None:
            raise ValueError("multipart request must include an audio/file/video field")
        return input_path, options

    data = await request.json()
    options.update(data)
    audio_b64 = data.get("audio_base64") or data.get("file_base64") or data.get("video_base64")
    if not audio_b64:
        raise ValueError("audio_base64, file_base64, or video_base64 is required")
    filename = _safe_filename(data.get("filename") or data.get("audio_filename") or "input.wav")
    input_path = out_dir / f"{req_id}_{filename}"
    _write_base64_audio(audio_b64, input_path)
    return input_path, options


@routes.post("/api/separate")
@routes.post("/api/separation/separate")
async def separate(request):
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")

    out_root = Path(os.environ.get("SEPARATION_OUTPUT_DIR") or (WORK_ROOT / "separation_outputs"))
    request_dir = out_root / req_id
    input_path = None
    start_time = time.time()
    try:
        input_path, options = await _read_separation_request(request, req_id, request_dir)
        logger.info(
            f"[{req_id}] separation started: input={input_path}, "
            f"model={options.get('model') or DEFAULT_SEPARATOR_MODEL}"
        )
        result = await asyncio.to_thread(
            _separate_audio_sync,
            input_path,
            request_dir / "stems",
            options,
        )
    except ValueError as exc:
        tb = traceback.format_exc()
        logger.warning(f"[{req_id}] Invalid separation request: {exc}\n{tb}")
        return _error(f"Invalid separation request: {exc}", status=400)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Separation failed: {exc}\n{tb}")
        return _error(f"Separation failed: {exc}\n{tb}", status=502)
    finally:
        if input_path is not None:
            _cleanup_temp_paths(input_path)

    vocals_path = Path(result["vocals"])
    background_path = Path(result["background"])
    vocals_bytes = vocals_path.read_bytes()
    background_bytes = background_path.read_bytes()
    elapsed = round(time.time() - start_time, 3)
    logger.info(
        f"[{req_id}] separation finished in {elapsed}s, "
        f"vocals={len(vocals_bytes)} bytes, background={len(background_bytes)} bytes"
    )

    return _json_response({
        "ok": True,
        "model": result["model"],
        "elapsed_seconds": elapsed,
        "vocals_base64": "data:audio/wav;base64," + base64.b64encode(vocals_bytes).decode("ascii"),
        "background_base64": "data:audio/wav;base64," + base64.b64encode(background_bytes).decode("ascii"),
        "vocals_path": str(vocals_path.resolve()),
        "background_path": str(background_path.resolve()),
        "relative_vocals_path": _relative_path(vocals_path),
        "relative_background_path": _relative_path(background_path),
        "separator_returncode": result["returncode"],
        "separator_stdout": result["stdout"],
        "separator_stderr": result["stderr"],
    })


@routes.post("/api/whisper/transcribe")
@routes.post("/api/asr/whisper")
async def whisper_transcribe(request):
    client_ip = request.remote or "-"
    req_id = uuid.uuid4().hex[:8]
    logger.info(f"[{req_id}] [{request.method}] {request.path} from {client_ip}")

    out_root = Path(os.environ.get("WHISPER_OUTPUT_DIR") or (WORK_ROOT / "whisper_outputs"))
    request_dir = out_root / req_id
    input_path = None
    start_time = time.time()
    try:
        input_path, options = await _read_separation_request(request, req_id, request_dir)
        model_name = str(options.get("model") or DEFAULT_WHISPER_MODEL).strip() or DEFAULT_WHISPER_MODEL
        device = _whisper_device(options.get("device") or "auto")
        compute_type = str(options.get("compute_type") or _default_whisper_compute_type(device)).strip()
        logger.info(
            "[%s] whisper started: input=%s model=%s device=%s compute_type=%s",
            req_id,
            input_path,
            model_name,
            device,
            compute_type,
        )
        model = await _ensure_whisper_model(model_name, device, compute_type)
        async with _WHISPER_INFER_SEM:
            result = await asyncio.to_thread(
                _transcribe_whisper_sync,
                model,
                input_path,
                options,
            )
        if _bool_option(options.get("include_audio_qc"), True):
            try:
                result["audio_qc"] = await asyncio.to_thread(
                    _build_whisper_audio_qc,
                    input_path,
                    result,
                )
            except Exception as exc:
                result["audio_qc"] = {"version": 1, "status": "error", "error": str(exc)[:500]}
    except ValueError as exc:
        tb = traceback.format_exc()
        logger.warning(f"[{req_id}] Invalid whisper request: {exc}\n{tb}")
        return _error(f"Invalid whisper request: {exc}", status=400)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Whisper transcription failed: {exc}\n{tb}")
        return _error(f"Whisper transcription failed: {exc}\n{tb}", status=502)
    finally:
        if input_path is not None:
            _cleanup_temp_paths(input_path)

    elapsed = round(time.time() - start_time, 3)
    logger.info(
        "[%s] whisper finished in %.3fs, segments=%d",
        req_id,
        elapsed,
        len(result.get("segments") or []),
    )
    return _json_response({
        "ok": True,
        "model": model_name,
        "elapsed_seconds": elapsed,
        **result,
    })


@routes.get("/")
async def index(request):
    return web.Response(text="ok", content_type="text/plain")


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
