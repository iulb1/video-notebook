from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from video_transcriber import VideoTranscriber, get_keyframe_threshold


class QueueLogHandler(logging.Handler):
    def __init__(self, event_queue: Any) -> None:
        super().__init__()
        self.event_queue = event_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.event_queue.put(
                {
                    "type": "log",
                    "level": record.levelname,
                    "message": self.format(record),
                }
            )
        except Exception:
            pass


def _setup_logger(event_queue: Any) -> logging.Logger:
    logger = logging.getLogger("transcription_worker")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = QueueLogHandler(event_queue)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def _output_formats(config: dict[str, Any]) -> set[str]:
    formats = {
        name
        for enabled, name in (
            (config.get("fmt_md", True), "md"),
            (config.get("fmt_json", False), "json"),
            (config.get("fmt_srt", False), "srt"),
            (config.get("fmt_docx", False), "docx"),
        )
        if enabled
    }
    if "docx" in formats:
        formats.add("md")
    return formats


def _build_exporter(config: dict[str, Any]):
    if not config.get("fmt_docx", False):
        return None
    from document_exporter import DocumentExporter

    return DocumentExporter(pandoc_path=config.get("pandoc_path"))


def _apply_runtime_config(transcriber: VideoTranscriber, config: dict[str, Any]) -> None:
    language = config.get("language")
    transcriber.language = None if language in (None, "", "auto") else language
    transcriber.save_keyframes = config.get("save_keyframes", True)
    transcriber.keyframe_threshold = get_keyframe_threshold(
        config.get("keyframe_frequency", "normal")
    )
    transcriber.ffmpeg_path = VideoTranscriber.resolve_ffmpeg_path_static(
        config.get("ffmpeg_path")
    )
    transcriber.exporter = _build_exporter(config)
    transcriber.output_formats = _output_formats(config)
    transcriber.trash_source = config.get("trash_source", False)
    transcriber.vad_filter = config.get("vad_filter", False)


def run_worker_process(jobs: list[dict[str, Any]], event_queue: Any, stop_event: Any) -> None:
    logger = _setup_logger(event_queue)
    completed = 0
    failed = 0
    try:
        if not jobs:
            event_queue.put({"type": "finished", "completed": 0, "failed": 0})
            return

        first_config = jobs[0]["config"]
        transcriber = VideoTranscriber(
            model_size=first_config.get("model"),
            backend=first_config.get("backend", "sensevoice"),
            device=first_config.get("device", "cuda"),
            language=first_config.get("language"),
            save_keyframes=first_config.get("save_keyframes", True),
            keyframe_threshold=get_keyframe_threshold(
                first_config.get("keyframe_frequency", "normal")
            ),
            ffmpeg_path=first_config.get("ffmpeg_path"),
            exporter=_build_exporter(first_config),
            output_formats=_output_formats(first_config),
            trash_source=first_config.get("trash_source", False),
            vad_filter=first_config.get("vad_filter", False),
            logger=logger,
            stop_event=stop_event,
        )
        event_queue.put(
            {
                "type": "ready",
                "backend": transcriber.backend_name,
                "model": transcriber.model_size,
                "device": transcriber.device,
            }
        )

        for job in jobs:
            task_id = job["task_id"]
            config = job["config"]
            if stop_event.is_set():
                event_queue.put({"type": "task_result", "task_id": task_id, "status": "stopped"})
                continue

            _apply_runtime_config(transcriber, config)
            event_queue.put(
                {
                    "type": "task_started",
                    "task_id": task_id,
                    "file_name": Path(config["input"]).name,
                }
            )

            def report_progress(current: int, total: int, file_name: str) -> None:
                event_queue.put(
                    {
                        "type": "progress",
                        "task_id": task_id,
                        "current": current,
                        "total": total,
                        "file_name": file_name,
                    }
                )

            try:
                success = transcriber.process_path(
                    input_path=config["input"],
                    output_dir=config["output"],
                    skip_processed=config.get("skip_processed", True),
                    progress_callback=report_progress,
                )
                status = "completed" if success else "failed"
            except Exception as exc:
                logger.exception("处理失败: %s", exc)
                status = "failed"

            completed += status == "completed"
            failed += status == "failed"
            event_queue.put({"type": "task_result", "task_id": task_id, "status": status})

        event_queue.put(
            {
                "type": "finished",
                "completed": completed,
                "failed": failed,
                "stopped": stop_event.is_set(),
            }
        )
    except Exception as exc:
        logger.exception("推理子进程异常: %s", exc)
        event_queue.put({"type": "fatal", "message": str(exc)})

