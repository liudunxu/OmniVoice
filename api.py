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
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
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
_API_LOCK = asyncio.Lock()


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


def _write_base64_audio(b64_data, out_path):
    """Decode base64 audio data and write to file. Supports data URI prefix."""
    b64_data = str(b64_data or "").strip()
    if b64_data.startswith("data:"):
        b64_data = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
    audio_bytes = base64.b64decode(b64_data)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    return out_path


def _read_audio_base64(path):
    """Read audio file and return base64 encoded string with data URI prefix."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


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
    if _API_MODEL is None:
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


def _write_generated_audio(model, audio, out_path):
    wav = audio[0]
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu()
    if hasattr(wav, "numpy"):
        wav = wav.numpy()
    waveform = np.squeeze(wav).astype(np.float32)
    sf.write(str(out_path), waveform, int(model.sampling_rate), subtype="PCM_16")


def _create_voice_clone_prompt(model, reference_audio, prompt_audio=None, prompt_text=""):
    ref_text_clean = prompt_text.strip() if prompt_text else None
    if prompt_audio and prompt_audio != reference_audio:
        candidates = [
            {"ref_audio": reference_audio, "prompt_audio": prompt_audio, "ref_text": ref_text_clean},
            {"ref_audio": reference_audio, "prompt_wav": prompt_audio, "ref_text": ref_text_clean},
            {"ref_audio": reference_audio, "prompt_wav_path": prompt_audio, "ref_text": ref_text_clean},
        ]
        for kwargs in candidates:
            try:
                return model.create_voice_clone_prompt(**kwargs)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
    return model.create_voice_clone_prompt(
        ref_audio=reference_audio,
        ref_text=ref_text_clean,
    )


def _is_empty_reference_after_preprocess(exc):
    return "Reference audio is empty after silence removal" in str(exc)


def _synthesize_omnivoice_to_file(
    model,
    text,
    out_path,
    reference_audio=None,
    prompt_audio=None,
    prompt_text="",
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
    def make_gen_config(enable_preprocess):
        return OmniVoiceGenerationConfig(
            num_step=int(inference_timesteps),
            guidance_scale=float(cfg_value),
            t_shift=float(t_shift),
            layer_penalty_factor=float(layer_penalty_factor),
            position_temperature=float(position_temperature),
            class_temperature=float(class_temperature),
            denoise=bool(denoise),
            preprocess_prompt=bool(enable_preprocess),
            postprocess_output=bool(postprocess_output),
            audio_chunk_duration=float(audio_chunk_duration),
            audio_chunk_threshold=float(audio_chunk_threshold),
        )

    gen_config = make_gen_config(preprocess_prompt)
    kw: Dict[str, Any] = {
        "text": text.strip(),
        "language": _resolve_language(language),
        "generation_config": gen_config,
    }
    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)
    clone_audio = reference_audio or prompt_audio
    if clone_audio:
        kw["voice_clone_prompt"] = _create_voice_clone_prompt(
            model,
            clone_audio,
            prompt_audio=prompt_audio if reference_audio else None,
            prompt_text=prompt_text,
        )
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
        kw["generation_config"] = make_gen_config(False)
        audio = generate_with_seed()
    _write_generated_audio(model, audio, out_path)


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

    # Prepare output directory
    out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decode reference audio first to measure duration for adaptive params
    ref_temp_path = None
    prompt_temp_path = None
    resolved_ref = None
    resolved_prompt = None
    ref_duration = None

    if reference_audio_base64:
        ref_temp_path = out_dir / f"ref_{uuid.uuid4().hex}.wav"
        ref_temp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_base64_audio(reference_audio_base64, ref_temp_path)
            resolved_ref = str(ref_temp_path)
            ref_duration = _get_ref_audio_duration(ref_temp_path)
            logger.info(f"[{req_id}] reference audio decoded: {ref_temp_path} ({ref_temp_path.stat().st_size} bytes), duration={ref_duration}s")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[{req_id}] Failed to decode reference_audio_base64: {exc}\n{tb}")
            if ref_temp_path and ref_temp_path.exists():
                ref_temp_path.unlink(missing_ok=True)
            return _error(f"Failed to decode reference_audio_base64: {exc}\n{tb}")

    if prompt_wav_base64:
        out_dir = Path(data.get("output_dir") or OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt_temp_path = out_dir / f"prompt_{uuid.uuid4().hex}.wav"
        try:
            _write_base64_audio(prompt_wav_base64, prompt_temp_path)
            resolved_prompt = str(prompt_temp_path)
            logger.info(f"[{req_id}] prompt wav decoded: {prompt_temp_path} ({prompt_temp_path.stat().st_size} bytes)")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(f"[{req_id}] Failed to decode prompt_wav_base64: {exc}\n{tb}")
            if ref_temp_path and ref_temp_path.exists():
                ref_temp_path.unlink(missing_ok=True)
            if prompt_temp_path and prompt_temp_path.exists():
                prompt_temp_path.unlink(missing_ok=True)
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

    # Adaptive duration: try to match ref audio length when user doesn't specify duration
    # Strategy: use ref audio duration as target, but avoid badcases by checking text length
    effective_duration = user_duration
    effective_speed = user_speed
    if user_duration is None and ref_duration is not None and user_speed == 1.0:
        # Estimate if forcing ref_duration would cause badcase
        # Heuristic: if text is short enough relative to ref audio, use ref_duration
        # A typical speaker produces ~3-5 chars/sec for Chinese, ~10-15 chars/sec for English
        is_chinese_text = bool(re.search(r'[\u4e00-\u9fff]', text))
        if is_chinese_text:
            chars_per_sec = 4.0  # conservative estimate for Chinese
        else:
            chars_per_sec = 12.0  # conservative estimate for English

        estimated_natural_duration = len(text) / chars_per_sec
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

    # Move ref_temp_path to out_dir if it was created in WORK_ROOT
    if ref_temp_path and ref_temp_path.parent != out_dir:
        new_ref_path = out_dir / ref_temp_path.name
        if ref_temp_path.exists():
            ref_temp_path.rename(new_ref_path)
            ref_temp_path = new_ref_path
            resolved_ref = str(ref_temp_path)

    start_time = time.time()
    try:
        async with _API_LOCK:
            model = await _ensure_api_model()
            logger.info(f"[{req_id}] synthesis started -> {out_path}")
            await asyncio.to_thread(
                _synthesize_omnivoice_to_file,
                model,
                text,
                out_path,
                reference_audio=resolved_ref,
                prompt_audio=resolved_prompt,
                prompt_text=effective_prompt_text if resolved_prompt else "",
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                denoise=denoise,
                speed=effective_speed,
                duration=effective_duration,
                language=data.get("language")
                or data.get("target_lang")
                or data.get("target_language")
                or data.get("output_language_code"),
                instruct=data.get("instruct"),
                preprocess_prompt=_bool_option(data.get("preprocess_prompt"), True),
                postprocess_output=_bool_option(data.get("postprocess_output"), True),
                seed=seed,
                t_shift=t_shift,
                layer_penalty_factor=layer_penalty_factor,
                position_temperature=position_temperature,
                class_temperature=class_temperature,
                audio_chunk_duration=float(data.get("audio_chunk_duration", 15.0)),
                audio_chunk_threshold=float(data.get("audio_chunk_threshold", 30.0)),
            )
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Synthesis failed: {exc}\n{tb}")
        if ref_temp_path and ref_temp_path.exists():
            ref_temp_path.unlink(missing_ok=True)
        if prompt_temp_path and prompt_temp_path.exists():
            prompt_temp_path.unlink(missing_ok=True)
        return _error(f"Synthesis failed: {exc}\n{tb}", status=502)

    elapsed = round(time.time() - start_time, 3)
    if not out_path.exists():
        logger.error(f"[{req_id}] Output file not created: {out_path}")
        if ref_temp_path and ref_temp_path.exists():
            ref_temp_path.unlink(missing_ok=True)
        if prompt_temp_path and prompt_temp_path.exists():
            prompt_temp_path.unlink(missing_ok=True)
        return _error("Synthesis finished but output file was not created.", status=502)

    audio_duration = _audio_duration_seconds(out_path)
    logger.info(
        f"[{req_id}] synthesis finished in {elapsed}s, output: {out_path} "
        f"({out_path.stat().st_size} bytes), audio_duration={audio_duration}"
    )

    try:
        output_base64 = _read_audio_base64(out_path)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[{req_id}] Failed to encode output audio: {exc}\n{tb}")
        if ref_temp_path and ref_temp_path.exists():
            ref_temp_path.unlink(missing_ok=True)
        if prompt_temp_path and prompt_temp_path.exists():
            prompt_temp_path.unlink(missing_ok=True)
        return _error(f"Failed to encode output audio: {exc}\n{tb}", status=502)

    if ref_temp_path and ref_temp_path.exists():
        ref_temp_path.unlink(missing_ok=True)
    if prompt_temp_path and prompt_temp_path.exists():
        prompt_temp_path.unlink(missing_ok=True)

    logger.info(f"[{req_id}] response sent, audio_base64_len={len(output_base64)}")
    return _json_response({
        "ok": True,
        "audio_base64": output_base64,
        "output_path": str(out_path.resolve()),
        "relative_path": _relative_path(out_path),
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": audio_duration,
        "target_duration_ms": target_duration_ms,
        "max_duration_ms": max_duration_ms,
        "duration_tolerance_ms": duration_tolerance_ms,
        "seed": seed,
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
  "target_duration_ms": 2200,
  "max_duration_ms": 3000,
  "duration_tolerance_ms": 176,
  "audio_chunk_duration": 15.0,        // 可选；长文本分块时长
  "audio_chunk_threshold": 30.0,       // 可选；触发分块阈值
  "seed": 123456789                    // 可选；不传时默认派生稳定 seed
}

Adaptive Quality Profiles (based on reference audio length):
  < 2s:  num_step=48, guidance_scale=1.8, t_shift=0.05 (conservative)
  2-4s: num_step=40, guidance_scale=2.0, t_shift=0.08 (balanced)
  4-5s: num_step=36, guidance_scale=2.0, t_shift=0.10 (optimal)
  > 5s: num_step=32, guidance_scale=2.2, t_shift=0.10 (standard)

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