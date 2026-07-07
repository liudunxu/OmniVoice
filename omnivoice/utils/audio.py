#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audio I/O and processing utilities.

Provides functions for loading, resampling, silence removal,
chunking, cross-fading, and format conversion.

All public functions in this module operate on **numpy float32 arrays**
with shape ``(C, T)`` (channels-first).
"""

import io
import logging

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, detect_nonsilent, split_on_silence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_waveform(audio_path: str):
    """Load audio from a file path, returning (data, sample_rate).

    Tries two backends in order:
    1. soundfile — covers WAV/FLAC/OGG etc., no ffmpeg needed.
    2. librosa — covers MP3/M4A etc. via audioread + ffmpeg.

    Returns:
        (data, sample_rate) where data is a numpy float32 array of
        shape (C, T).
    """
    try:
        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        return data.T, sr  # (T, C) → (C, T)
    except Exception:
        # soundfile cannot handle MP3/M4A etc., fall back to librosa.
        import librosa

        data, sr = librosa.load(audio_path, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data, sr


def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    """Load a waveform from file and resample to the target rate.

    Parameters:
        audio_path: path of the audio.
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).
    """
    data, sr = load_waveform(audio_path)

    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


def load_audio_bytes(raw: bytes, sampling_rate: int) -> np.ndarray:
    """Load audio from in-memory bytes and resample.

    Parameters:
        raw: raw audio file bytes (e.g. from WebDataset).
        sampling_rate: target sampling rate.

    Returns:
        Numpy float32 array of shape (1, T).
    """
    buf = io.BytesIO(raw)

    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        data = data.T  # (T, C) → (C, T)
    except Exception:
        import librosa

        buf.seek(0)
        data, sr = librosa.load(buf, sr=None, mono=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]

    if data.shape[0] > 1:
        data = np.mean(data, axis=0, keepdims=True)
    if sr != sampling_rate:
        data = torchaudio.functional.resample(
            torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate
        ).numpy()

    return data


# ---------------------------------------------------------------------------
# Audio processing (all numpy in / numpy out)
# ---------------------------------------------------------------------------


def numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """Convert a numpy float32 array of shape (C, T) to a pydub AudioSegment."""
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()  # interleave channels
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def audiosegment_to_numpy(aseg: AudioSegment) -> np.ndarray:
    """Convert a pydub AudioSegment to a numpy float32 array of shape (C, T)."""
    data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
    if aseg.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, aseg.channels).T


def compress_silence(
    audio: np.ndarray,
    sampling_rate: int,
    min_silence_len: int = 300,
    silence_thresh: float = -50,
    target_silence_len: int = 200,
    seek_step: int = 10,
) -> np.ndarray:
    """Compress middle silences longer than *min_silence_len* ms down to *target_silence_len* ms.

    Unlike remove_silence(), this preserves natural phrasing and breath pauses,
    only shortening excessively long gaps that would otherwise make the output
    sound disjointed or too short.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        min_silence_len: gaps longer than this (ms) are compressed.
        silence_thresh: dBFS threshold for silence.
        target_silence_len: duration (ms) to compress each long gap to.
        seek_step: pydub detection step in ms.

    Returns:
        Numpy array with shape (C, T').
    """
    wave = numpy_to_audiosegment(audio, sampling_rate)
    nonsilent = detect_nonsilent(
        wave, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=seek_step
    )
    if not nonsilent:
        return audio

    target_ms = max(0, int(target_silence_len))
    parts = []
    prev_end = 0
    for start, end in nonsilent:
        gap = start - prev_end
        if gap > target_ms:
            parts.append(wave[prev_end : prev_end + target_ms // 2])
            parts.append(AudioSegment.silent(duration=target_ms))
            parts.append(wave[start - target_ms // 2 : start])
        else:
            if prev_end < start:
                parts.append(wave[prev_end:start])
        parts.append(wave[start:end])
        prev_end = end

    if prev_end < len(wave):
        parts.append(wave[prev_end:])

    out = AudioSegment.silent(duration=0)
    for p in parts:
        out += p
    return audiosegment_to_numpy(out)


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
    mode: str = "compress",
) -> np.ndarray:
    """Remove or compress middle silences and trim edge silences.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        mid_sil: middle-silence threshold in ms (0 to skip).
        lead_sil: kept leading silence in ms.
        trail_sil: kept trailing silence in ms.
        mode: "compress" keeps pauses but caps them; "remove" deletes them.

    Returns:
        Numpy array with shape (C, T').
    """
    wave = numpy_to_audiosegment(audio, sampling_rate)

    if mid_sil > 0:
        if mode == "compress":
            return compress_silence(
                audio,
                sampling_rate,
                min_silence_len=mid_sil,
                target_silence_len=min(200, mid_sil),
            )
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50)

    return audiosegment_to_numpy(wave)


def remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int = 100,
    trail_sil: int = 300,
    silence_threshold: float = -50,
) -> AudioSegment:
    """Remove edge silences, keeping *lead_sil* / *trail_sil* ms."""
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def fade_and_pad_audio(
    audio: np.ndarray,
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Apply fade-in/out and pad with silence to prevent clicks.

    Args:
        audio: numpy array of shape (C, T).
        pad_duration: silence padding duration per side (seconds).
        fade_duration: fade curve duration (seconds).
        sample_rate: audio sampling rate.

    Returns:
        Processed numpy array of shape (C, T_new).
    """
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)

    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            fade_in = np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., :k] *= fade_in

            fade_out = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= fade_out

    if pad_samples > 0:
        silence = np.zeros(
            (processed.shape[0], pad_samples),
            dtype=processed.dtype,
        )
        processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


def suppress_spikes_and_limit(
    audio: np.ndarray,
    peak_limit: float = 0.92,
    spike_threshold: float = 0.45,
    spike_ratio: float = 3.0,
) -> np.ndarray:
    """Suppress isolated impulse spikes and leave headroom before WAV export."""
    if audio.size == 0:
        return audio

    processed = np.asarray(audio, dtype=np.float32).copy()
    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

    if processed.ndim == 1:
        view = processed[np.newaxis, :]
    else:
        view = processed.reshape((-1, processed.shape[-1]))

    for channel in view:
        if channel.size < 3:
            continue
        prev = channel[:-2]
        center = channel[1:-1]
        next_ = channel[2:]
        neighbor_peak = np.maximum(np.abs(prev), np.abs(next_))
        is_spike = (
            (np.abs(center - prev) >= spike_threshold)
            & (np.abs(center - next_) >= spike_threshold)
            & (np.abs(center) >= neighbor_peak * spike_ratio + 0.05)
        )
        if np.any(is_spike):
            center[is_spike] = (prev[is_spike] + next_[is_spike]) * 0.5

    peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if peak > peak_limit > 0:
        processed *= peak_limit / peak

    return processed.astype(np.float32, copy=False)


def trim_long_audio(
    audio: np.ndarray,
    sampling_rate: int,
    max_duration: float = 15.0,
    min_duration: float = 3.0,
    trim_threshold: float = 20.0,
) -> np.ndarray:
    """Trim audio to <= *max_duration* by splitting at the largest silence gap.

    Only trims when the audio exceeds *trim_threshold* seconds.

    Args:
        audio: numpy array of shape (C, T).
        sampling_rate: audio sampling rate.
        max_duration: maximum duration in seconds.
        min_duration: minimum duration in seconds.
        trim_threshold: only trim if audio is longer than this (seconds).

    Returns:
        Trimmed numpy array.
    """
    duration = audio.shape[-1] / sampling_rate
    if duration <= trim_threshold:
        return audio

    seg = numpy_to_audiosegment(audio, sampling_rate)
    nonsilent = detect_nonsilent(
        seg, min_silence_len=100, silence_thresh=-40, seek_step=10
    )
    if not nonsilent:
        return audio

    max_ms = int(max_duration * 1000)
    min_ms = int(min_duration * 1000)

    best_split = 0
    for start, end in nonsilent:
        if start > best_split and start <= max_ms:
            best_split = start
        if end > max_ms:
            break

    if best_split < min_ms:
        best_split = min(max_ms, len(seg))

    trimmed = seg[:best_split]
    return audiosegment_to_numpy(trimmed)


def cross_fade_chunks(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_duration: float = 0.04,
) -> np.ndarray:
    """Concatenate audio chunks with a short true-overlap cross-fade.

    Unlike the previous implementation, this overlaps the tail of the previous
    chunk with the head of the next one and linearly cross-fades them.  This
    removes the audible 0.3 s silence gap while still smoothing chunk seams.

    Args:
        chunks: list of numpy arrays, each (C, T).
        sample_rate: audio sample rate.
        crossfade_duration: overlap region in seconds (default 40 ms).

    Returns:
        Merged numpy array (C, T_total).
    """
    if len(chunks) == 1:
        return chunks[0]

    overlap_n = max(1, int(crossfade_duration * sample_rate))
    merged = chunks[0].copy()

    for chunk in chunks[1:]:
        # Guard against very short chunks; overlap cannot be longer than either
        # the tail of the previous chunk or the head of the next chunk.
        avail = min(overlap_n, merged.shape[-1], chunk.shape[-1])
        if avail <= 0:
            merged = np.concatenate([merged, chunk], axis=-1)
            continue

        # Linear cross-fade weights: previous fades out, next fades in.
        w = np.linspace(0, 1, avail, dtype=np.float32)[np.newaxis, :]
        cross = merged[..., -avail:] * (1.0 - w) + chunk[..., :avail] * w

        merged = np.concatenate(
            [merged[..., :-avail], cross, chunk[..., avail:]],
            axis=-1,
        )

    return merged
