from __future__ import annotations

from typing import Any

import numpy as np

from .base import ASRBackend, ASRBackendError, TranscriptionResult


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRBackendError(
                "未安装 faster-whisper，请运行 `pip install -r requirements.txt`。"
            ) from exc

        model_name = self.model_name
        if model_name.startswith("faster-whisper-"):
            model_name = model_name.removeprefix("faster-whisper-")
        self.logger.info("Initializing Faster Whisper model (%s)...", model_name)
        try:
            self.model = self._load_model(WhisperModel, model_name, self.device)
        except Exception as exc:
            if self.device != "cuda":
                raise ASRBackendError(f"Faster Whisper 模型初始化失败: {exc}") from exc
            self.logger.warning("GPU 初始化失败 (%s)，自动回退到 CPU...", exc)
            self.device = "cpu"
            try:
                self.model = self._load_model(WhisperModel, model_name, self.device)
            except Exception as cpu_exc:
                raise ASRBackendError(
                    f"Faster Whisper 模型初始化失败（GPU 和 CPU 均失败）: {cpu_exc}"
                ) from cpu_exc

    def _load_model(self, model_class, model_name: str, device: str):
        compute_type = "int8_float16" if device == "cuda" else "int8"
        model = model_class(model_name, device=device, compute_type=compute_type)
        self.logger.info(
            "Faster Whisper loaded (device=%s, compute_type=%s)", device, compute_type
        )
        return model

    def transcribe(
        self,
        audio: np.ndarray[Any, Any],
        *,
        language: str | None,
        vad_filter: bool,
        media_name: str,
    ) -> TranscriptionResult:
        self.ensure_not_stopped()
        duration_seconds = float(len(audio) / 16000.0)
        chunk_size = 30 * 16000
        total_chunks = max(1, (len(audio) + chunk_size - 1) // chunk_size)
        raw_segments: list[dict[str, float | str]] = []
        detected_language: str | None = None
        language_probability = 0.0

        self.logger.info("  Transcribing with Faster Whisper: %s", media_name)
        for index in range(total_chunks):
            self.ensure_not_stopped()
            start_index = index * chunk_size
            end_index = min(start_index + chunk_size, len(audio))
            chunk = audio[start_index:end_index]
            if len(chunk) == 0:
                continue
            offset = start_index / 16000.0
            if total_chunks > 1:
                self.logger.info("  Processing chunk %d/%d", index + 1, total_chunks)
            segments, info = self.model.transcribe(
                chunk,
                language=language,
                beam_size=3,
                word_timestamps=False,
                vad_filter=vad_filter,
            )
            if detected_language is None:
                detected_language = getattr(info, "language", None)
                language_probability = float(
                    getattr(info, "language_probability", 0.0) or 0.0
                )
            for segment in segments:
                raw_segments.append(
                    {
                        "start": float(segment.start) + offset,
                        "end": float(segment.end) + offset,
                        "text": str(segment.text).strip(),
                    }
                )

        return TranscriptionResult(
            segments=raw_segments,
            metadata={
                "language": detected_language or language or "unknown",
                "language_probability": language_probability,
                "duration_seconds": duration_seconds,
                "segment_count": len(raw_segments),
                "raw_segment_count": len(raw_segments),
                "language_mode": language or "auto",
                "backend": self.name,
                "model": self.model_name,
            },
        )
