from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np

from .base import ASRBackend, ASRBackendError, TranscriptionResult


_TAG_PATTERN = re.compile(r"<\|[^|]+\|>")
_LANGUAGE_TAG_PATTERN = re.compile(r"<\|(zh|en|yue|ja|ko)\|>", re.IGNORECASE)


class SenseVoiceBackend(ASRBackend):
    name = "sensevoice"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise ASRBackendError(
                "未安装 SenseVoice 可选依赖，请运行 "
                "`pip install -r requirements-sensevoice.txt`。"
            ) from exc

        self.logger.info("Initializing SenseVoice model (%s)...", self.model_name)
        options: dict[str, Any] = {
            "model": self.model_name,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
            "device": "cuda:0" if self.device == "cuda" else "cpu",
            "disable_update": True,
            "trust_remote_code": True,
        }
        try:
            self.model = AutoModel(**options)
        except Exception as exc:
            if self.device != "cuda":
                raise ASRBackendError(f"SenseVoice 模型初始化失败: {exc}") from exc
            self.logger.warning("SenseVoice GPU 初始化失败 (%s)，自动回退到 CPU...", exc)
            self.device = "cpu"
            options["device"] = "cpu"
            try:
                self.model = AutoModel(**options)
            except Exception as cpu_exc:
                raise ASRBackendError(
                    f"SenseVoice 模型初始化失败（GPU 和 CPU 均失败）: {cpu_exc}"
                ) from cpu_exc

    @staticmethod
    def _clean_text(text: object) -> str:
        return " ".join(_TAG_PATTERN.sub("", str(text or "")).split())

    @staticmethod
    def _seconds(value: object) -> float:
        try:
            return max(0.0, float(value) / 1000.0)
        except (TypeError, ValueError):
            return 0.0

    def _sentence_segments(
        self, items: Iterable[dict[str, Any]]
    ) -> list[dict[str, float | str]]:
        segments: list[dict[str, float | str]] = []
        for item in items:
            text = self._clean_text(item.get("text"))
            if not text:
                continue
            start = self._seconds(item.get("start"))
            end = max(start, self._seconds(item.get("end")))
            segments.append({"start": start, "end": end, "text": text})
        return segments

    def _timestamp_segments(
        self, item: dict[str, Any], duration: float
    ) -> list[dict[str, float | str]]:
        timestamps = item.get("timestamp") or item.get("timestamps") or []
        words = item.get("words") or []
        if timestamps and words and len(timestamps) == len(words):
            segments: list[dict[str, float | str]] = []
            buffer: list[str] = []
            segment_start = 0.0
            segment_end = 0.0
            for word, pair in zip(words, timestamps):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                text = self._clean_text(word)
                if not text:
                    continue
                start = self._seconds(pair[0])
                end = max(start, self._seconds(pair[1]))
                if not buffer:
                    segment_start = start
                buffer.append(text)
                segment_end = end
                joined = "".join(buffer)
                if len(joined) >= 60 or text.endswith(("。", "！", "？", ".", "!", "?")):
                    segments.append(
                        {"start": segment_start, "end": segment_end, "text": joined}
                    )
                    buffer = []
            if buffer:
                segments.append(
                    {"start": segment_start, "end": segment_end, "text": "".join(buffer)}
                )
            if segments:
                return segments

        text = self._clean_text(item.get("text"))
        if not text:
            return []
        return [{"start": 0.0, "end": duration, "text": text}]

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
        self.logger.info("  Transcribing with SenseVoice: %s", media_name)
        options: dict[str, Any] = {
            "input": audio,
            "language": language or "auto",
            "use_itn": True,
            "output_timestamp": True,
            "batch_size_s": 60,
        }
        if vad_filter:
            options.update({"merge_vad": True, "merge_length_s": 15})
        try:
            output = self.model.generate(**options)
        except Exception as exc:
            raise ASRBackendError(f"SenseVoice 转写失败: {media_name}: {exc}") from exc

        self.ensure_not_stopped()
        items = output if isinstance(output, list) else [output]
        segments: list[dict[str, float | str]] = []
        detected_language: str | None = None
        for item in items:
            if not isinstance(item, dict):
                continue
            detected_language = detected_language or item.get("language")
            if not detected_language:
                language_match = _LANGUAGE_TAG_PATTERN.search(str(item.get("text") or ""))
                if language_match:
                    detected_language = language_match.group(1).lower()
            sentence_info = item.get("sentence_info")
            current = (
                self._sentence_segments(sentence_info)
                if isinstance(sentence_info, list)
                else []
            )
            segments.extend(current or self._timestamp_segments(item, duration_seconds))

        return TranscriptionResult(
            segments=segments,
            metadata={
                "language": detected_language or language or "unknown",
                "language_probability": None,
                "duration_seconds": duration_seconds,
                "segment_count": len(segments),
                "raw_segment_count": len(segments),
                "language_mode": language or "auto",
                "backend": self.name,
                "model": self.model_name,
            },
        )
