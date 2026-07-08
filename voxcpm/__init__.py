# Vendored from OpenBMB/VoxCPM (https://github.com/OpenBMB/VoxCPM), Apache-2.0.
# Inference-only subset: cli.py, timestamps/, training/ were dropped (not imported by
# `from voxcpm import VoxCPM`). See voxcpm/LICENSE for the upstream license.
# Local patch: core.py forwards `trim_silence_vad` to build_prompt_cache.
from .core import VoxCPM

__all__ = [
    "VoxCPM",
]
