"""Shared validation helpers for VoxCPM nodes.

The helpers in this module deliberately avoid importing ComfyUI so they can be
unit-tested without booting the full server.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


_UNSAFE_PATH_PARTS = {".", ".."}
_NAME_CLEANUP = re.compile(r"[^\w.-]+", re.UNICODE)
_CJK_CHAR = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af]"
)
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


class GeneratedAudioQualityError(ValueError):
    """Raised when generated audio is structurally unusable."""


def validate_path_component(value: object, *, label: str = "name") -> str:
    """Validate an existing filename/directory name without rewriting it."""

    raw = str(value or "").strip()
    if (
        not raw
        or raw in _UNSAFE_PATH_PARTS
        or os.path.isabs(raw)
        or "/" in raw
        or "\\" in raw
        or "\x00" in raw
    ):
        raise ValueError(f"Unsafe {label}: {raw!r}")
    return raw


def safe_name_component(value: object, default: str, max_length: int = 96) -> str:
    """Return a filesystem-safe single path component.

    Separators, absolute paths, and traversal components are rejected instead
    of silently rewritten so an API caller cannot redirect plugin outputs.
    """

    raw = str(value or default).strip()
    if (
        not raw
        or raw in _UNSAFE_PATH_PARTS
        or os.path.isabs(raw)
        or "/" in raw
        or "\\" in raw
        or "\x00" in raw
    ):
        raise ValueError(f"Unsafe path component: {raw!r}")

    cleaned = _NAME_CLEANUP.sub("_", raw).strip("._-")[:max_length]
    if not cleaned or cleaned in _UNSAFE_PATH_PARTS:
        raise ValueError(f"Unsafe path component: {raw!r}")
    return cleaned


def safe_child_directory(base: object, name: object, default: str) -> Path:
    """Resolve a named directory and prove it remains below ``base``."""

    base_path = Path(base).resolve()
    component = safe_name_component(name, default)
    child = (base_path / component).resolve()
    if child.parent != base_path:
        raise ValueError(f"Output directory escapes configured root: {child}")
    return child


def resolve_allowed_file(
    value: object,
    roots: Iterable[object],
    *,
    suffixes: Optional[Sequence[str]] = None,
    label: str = "file",
) -> Path:
    """Resolve an existing file constrained to one of ``roots``."""

    raw = str(value or "").strip()
    if not raw or "\x00" in raw:
        raise ValueError(f"{label} must be a non-empty path")

    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    allowed_roots = [Path(root).resolve() for root in roots if root]
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError(
            f"{label} must be inside a ComfyUI input, output, or temp directory: {path}"
        )

    if suffixes:
        normalized = {suffix.lower() for suffix in suffixes}
        if path.suffix.lower() not in normalized:
            raise ValueError(
                f"{label} must use one of {sorted(normalized)}, got: {path.name}"
            )
    return path


def validate_generated_audio(wav: object, sample_rate: int, target_text: str) -> np.ndarray:
    """Validate basic waveform integrity and reject implausibly short speech.

    VoxCPM's built-in bad-case retry currently catches only overlong feature
    sequences. A premature EOS can therefore return a short but otherwise valid
    ndarray. The language-aware lower bound below is intentionally permissive:
    up to 10 CJK characters/s or 5 Latin words/s is accepted.
    """

    array = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(sample_rate) <= 0:
        raise GeneratedAudioQualityError(f"Invalid sample rate: {sample_rate}")
    if array.size == 0:
        raise GeneratedAudioQualityError("VoxCPM returned an empty waveform")
    if not np.isfinite(array).all():
        raise GeneratedAudioQualityError("VoxCPM returned NaN or infinite samples")

    peak = float(np.max(np.abs(array)))
    rms = float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))
    if peak < 1e-5 or rms < 1e-6:
        raise GeneratedAudioQualityError(
            f"VoxCPM returned silent audio (peak={peak:.2e}, rms={rms:.2e})"
        )

    text = str(target_text or "").strip()
    cjk_count = len(_CJK_CHAR.findall(text))
    latin_words = len(_LATIN_WORD.findall(_CJK_CHAR.sub(" ", text)))
    minimum_duration = max(0.25, cjk_count / 10.0 + latin_words / 5.0)
    duration = array.size / float(sample_rate)
    if duration + 0.05 < minimum_duration:
        raise GeneratedAudioQualityError(
            "VoxCPM returned implausibly short audio: "
            f"duration={duration:.2f}s, required>={minimum_duration:.2f}s, "
            f"cjk_chars={cjk_count}, latin_words={latin_words}"
        )
    return array
