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
import shutil
import subprocess
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


def _build_synth_audio_qc(waveform, sampling_rate: int, quality_issues=None):
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
    language = options.get("language")
    if language == "tl":
        language = "fil"
    if language and "-" in str(language):
        language = str(language).split("-", 1)[0]

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
        "whisper_models_cached": len(_WHISPER_MODELS),
        "whisper_cached_models": [
            {"model": model, "device": device, "compute_type": compute_type}
            for model, device, compute_type in _WHISPER_MODELS.keys()
        ],
    })


@routes.post("/api/voxcpm/unload")
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
    audio_qc = None
    if _bool_option(data.get("include_audio_qc"), True):
        try:
            audio_qc = _build_synth_audio_qc(
                audio_waveform,
                model.sampling_rate,
                quality_issues=quality_issues,
            )
        except Exception as exc:
            logger.warning(f"[{req_id}] Failed to build synthesis audio_qc: {exc}")
            audio_qc = {"version": 1, "status": "error", "error": str(exc)[:500]}
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
