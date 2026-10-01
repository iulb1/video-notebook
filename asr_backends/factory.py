from __future__ import annotations

import logging
import threading

from .base import ASRBackend, ASRBackendError


DEFAULT_MODELS = {
    "faster-whisper": "faster-whisper-small",
    "sensevoice": "iic/SenseVoiceSmall",
}

_ALIASES = {
    "faster_whisper": "faster-whisper",
    "whisper": "faster-whisper",
    "sense-voice": "sensevoice",
}


def normalize_backend_name(name: str | None) -> str:
    normalized = (name or "sensevoice").strip().lower()
    return _ALIASES.get(normalized, normalized)


def resolve_model_name(backend: str, model_name: str | None) -> str:
    normalized_backend = normalize_backend_name(backend)
    if model_name and model_name.strip():
        return model_name.strip()
    try:
        return DEFAULT_MODELS[normalized_backend]
    except KeyError as exc:
        raise ASRBackendError(f"未知 ASR 后端: {backend}") from exc


def create_backend(
    backend: str = "sensevoice",
    model_name: str | None = None,
    device: str = "cuda",
    logger: logging.Logger | None = None,
    stop_event: threading.Event | None = None,
) -> ASRBackend:
    normalized = normalize_backend_name(backend)
    resolved_model = resolve_model_name(normalized, model_name)
    if normalized == "faster-whisper":
        from .faster_whisper import FasterWhisperBackend

        backend_class = FasterWhisperBackend
    elif normalized == "sensevoice":
        from .sensevoice import SenseVoiceBackend

        backend_class = SenseVoiceBackend
    else:
        choices = ", ".join(sorted(DEFAULT_MODELS))
        raise ASRBackendError(f"未知 ASR 后端: {backend}。可选值: {choices}")
    return backend_class(
        model_name=resolved_model,
        device=device,
        logger=logger,
        stop_event=stop_event,
    )
