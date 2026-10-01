from __future__ import annotations

import logging
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from asr_backends.base import ASRBackend, TranscriptionResult
from asr_backends.factory import normalize_backend_name, resolve_model_name
from asr_backends.faster_whisper import FasterWhisperBackend
from asr_backends.sensevoice import SenseVoiceBackend
from video_transcriber import VideoTranscriber


class StubBackend(ASRBackend):
    name = "stub"

    def transcribe(self, audio, *, language, vad_filter, media_name):
        return TranscriptionResult(
            segments=[{"start": 0.0, "end": 1.0, "text": "测试"}],
            metadata={
                "language": language or "zh",
                "language_probability": 1.0,
                "duration_seconds": len(audio) / 16000,
                "segment_count": 1,
                "raw_segment_count": 1,
                "language_mode": language or "auto",
            },
        )


class BackendFactoryTests(unittest.TestCase):
    def test_defaults_and_aliases(self):
        self.assertEqual(normalize_backend_name(None), "sensevoice")
        self.assertEqual(normalize_backend_name("whisper"), "faster-whisper")
        self.assertEqual(resolve_model_name("faster-whisper", None), "faster-whisper-small")
        self.assertEqual(resolve_model_name("sensevoice", None), "iic/SenseVoiceSmall")


class FasterWhisperMappingTests(unittest.TestCase):
    def test_chunk_offsets_are_added(self):
        class FakeModel:
            def __init__(self):
                self.calls = 0

            def transcribe(self, audio, **_kwargs):
                self.calls += 1
                segment = SimpleNamespace(start=0.0, end=len(audio) / 16000, text="片段")
                info = SimpleNamespace(language="zh", language_probability=0.9)
                return [segment], info

        backend = object.__new__(FasterWhisperBackend)
        backend.model_name = "faster-whisper-small"
        backend.device = "cpu"
        backend.logger = logging.getLogger("test")
        backend.stop_event = threading.Event()
        backend.model = FakeModel()
        result = backend.transcribe(
            np.zeros(31 * 16000, dtype=np.float32),
            language=None,
            vad_filter=False,
            media_name="long.wav",
        )
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[1]["start"], 30.0)
        self.assertEqual(result.metadata["backend"], "faster-whisper")


class VideoTranscriberInjectionTests(unittest.TestCase):
    @patch.object(VideoTranscriber, "resolve_ffmpeg_path", return_value="ffmpeg")
    def test_core_uses_injected_backend(self, _resolve_ffmpeg):
        backend = StubBackend("stub-model", "cpu", logging.getLogger("test"))
        transcriber = VideoTranscriber(asr_backend=backend, device="cpu")
        segments, metadata = transcriber.transcribe_audio(
            np.zeros(16000, dtype=np.float32), "sample.wav"
        )
        self.assertEqual(segments[0]["text"], "测试")
        self.assertEqual(metadata["language"], "zh")
        self.assertEqual(transcriber.backend_name, "stub")


class SenseVoiceMappingTests(unittest.TestCase):
    def setUp(self):
        self.backend = object.__new__(SenseVoiceBackend)

    def test_sentence_info_mapping(self):
        segments = self.backend._sentence_segments(
            [{"start": 100, "end": 850, "text": "<|zh|>你好"}]
        )
        self.assertEqual(segments, [{"start": 0.1, "end": 0.85, "text": "你好"}])

    def test_word_timestamps_are_grouped(self):
        segments = self.backend._timestamp_segments(
            {
                "words": ["你", "好", "。", "下", "一", "句"],
                "timestamp": [[0, 100], [100, 300], [300, 400], [500, 600], [600, 700], [700, 900]],
                "text": "你好。下一句",
            },
            1.0,
        )
        self.assertEqual(segments[0], {"start": 0.0, "end": 0.4, "text": "你好。"})
        self.assertEqual(segments[1]["text"], "下一句")


if __name__ == "__main__":
    unittest.main()
