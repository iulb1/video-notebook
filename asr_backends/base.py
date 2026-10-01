from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


Segment = dict[str, float | str]
Metadata = dict[str, str | float | int | None]


class ASRBackendError(RuntimeError):
    """ASR 后端加载或推理失败。"""


@dataclass(slots=True)
class TranscriptionResult:
    segments: list[Segment]
    metadata: Metadata


class ASRBackend(ABC):
    name: str

    def __init__(
        self,
        model_name: str,
        device: str,
        logger: logging.Logger | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.stop_event = stop_event

    def ensure_not_stopped(self) -> None:
        if self.stop_event and self.stop_event.is_set():
            raise ASRBackendError("Processing stopped by user.")

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray[Any, Any],
        *,
        language: str | None,
        vad_filter: bool,
        media_name: str,
    ) -> TranscriptionResult:
        """将 16kHz 单声道 float32 音频转换为带时间戳的分段。"""
