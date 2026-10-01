#!/usr/bin/env python3
"""视频字幕转录工具入口。"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import queue
import sys
from pathlib import Path

from transcription_worker import run_worker_process


LOGGER = logging.getLogger("video_notebook")


def configure_console_encoding() -> None:
    """在 Windows 下尽量统一控制台为 UTF-8，减少中文乱码。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_exporter(export_word: bool, pandoc_path: str | None = None):
    """按需创建导出器，避免无关依赖阻塞普通转写。"""
    if not export_word:
        return None
    try:
        from document_exporter import DocumentExporter
    except ImportError as exc:
        raise RuntimeError(
            "未找到文档导出依赖，请先运行 `pip install -r requirements.txt`。"
        ) from exc
    return DocumentExporter(pandoc_path=pandoc_path)


def parse_output_formats(export_word: bool) -> set[str]:
    """CLI 默认输出格式。"""
    formats = {"md"}
    if export_word:
        formats.add("docx")
    return formats


def configure_logging(verbose: bool) -> None:
    """配置控制台日志输出。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="视频字幕转录工具 - 使用可切换的本地 ASR 模型生成带时间戳字幕",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py --input ./videos --output ./subtitles
  python main.py -i ./videos -o ./subtitles --device cuda
  python main.py -i ./videos -o ./subtitles --ffmpeg C:/tools/ffmpeg/bin/ffmpeg.exe
  python main.py -i ./videos -o ./subtitles --no-skip
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input directory containing video files to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory to store generated subtitle files",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sensevoice",
        choices=["faster-whisper", "sensevoice"],
        help="ASR backend (default: sensevoice)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or local model directory. Uses the backend default when omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computing device (default: cuda)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Specify transcription language, e.g., zh, en; auto-detect if not provided",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default=None,
        help="Optional ffmpeg executable path. Falls back to PATH or VIDEO_NOTEBOOK_FFMPEG.",
    )
    parser.add_argument(
        "--pandoc",
        type=str,
        default=None,
        help="Optional pandoc executable path. Falls back to PATH.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do not skip processed files, force reprocess all videos",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output more detailed debug logs",
    )
    parser.add_argument(
        "--no-keyframes",
        action="store_true",
        help="Do not extract keyframe screenshots or insert Obsidian image links in Markdown",
    )
    parser.add_argument(
        "--no-export-word",
        action="store_true",
        help="Do not export .docx file for each Markdown note",
    )
    parser.add_argument(
        "--keyframe-frequency",
        type=str,
        default="normal",
        choices=["few", "normal", "medium", "many", "all"],
        help="Screenshot frequency (default: normal): few, normal, medium, many, all",
    )
    parser.add_argument(
        "--trash",
        action="store_true",
        help="处理完成后将源视频文件移入回收站",
    )
    return parser


def log_runtime_summary(args: argparse.Namespace) -> None:
    """Print runtime parameter summary."""
    LOGGER.info("=" * 60)
    LOGGER.info("Video Subtitle Transcription Tool")
    LOGGER.info("=" * 60)
    LOGGER.info("Input directory: %s", args.input)
    LOGGER.info("Output directory: %s", args.output)
    LOGGER.info("ASR backend: %s", args.backend)
    LOGGER.info("Model name: %s", args.model or "backend default")
    LOGGER.info("Computing device: %s", args.device)
    LOGGER.info("Transcription language: %s", args.language or "auto-detect")
    LOGGER.info("FFmpeg path: %s", args.ffmpeg or "auto-detect from PATH/env")
    LOGGER.info("Keyframe screenshots: %s (frequency: %s)", not args.no_keyframes, args.keyframe_frequency)
    LOGGER.info("Export Word: %s", not args.no_export_word)
    LOGGER.info("Skip processed: %s", not args.no_skip)
    LOGGER.info("Trash source: %s", args.trash)
    LOGGER.info("=" * 60)


def main() -> int:
    """Program entry point."""
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)

    try:
        # If not a directory, check if file exists
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {args.input}")

        log_runtime_summary(args)

        config = {
            "input": args.input,
            "output": args.output,
            "backend": args.backend,
            "model": args.model,
            "device": args.device,
            "language": args.language,
            "ffmpeg_path": args.ffmpeg,
            "pandoc_path": args.pandoc,
            "save_keyframes": not args.no_keyframes,
            "keyframe_frequency": args.keyframe_frequency,
            "skip_processed": not args.no_skip,
            "fmt_md": True,
            "fmt_docx": not args.no_export_word,
            "trash_source": args.trash,
        }
        context = multiprocessing.get_context("spawn")
        event_queue = context.Queue()
        stop_event = context.Event()
        process = context.Process(
            target=run_worker_process,
            args=([{"task_id": 1, "config": config}], event_queue, stop_event),
            name="video-notebook-worker",
        )
        process.start()
        result_status: str | None = None
        worker_finished = False
        dead_polls = 0
        try:
            while True:
                try:
                    event = event_queue.get(timeout=0.2)
                except queue.Empty:
                    if process.is_alive():
                        dead_polls = 0
                        continue
                    dead_polls += 1
                    if dead_polls >= 5:
                        break
                    continue
                dead_polls = 0
                event_type = event.get("type")
                if event_type == "log":
                    level = getattr(logging, event.get("level", "INFO"), logging.INFO)
                    LOGGER.log(level, event.get("message", ""))
                elif event_type == "progress":
                    LOGGER.info(
                        "Progress %s/%s: %s",
                        event.get("current", 0),
                        event.get("total", 0),
                        event.get("file_name", ""),
                    )
                elif event_type == "task_result":
                    result_status = event.get("status")
                elif event_type == "finished":
                    worker_finished = True
                elif event_type == "fatal":
                    LOGGER.error("Worker failed: %s", event.get("message", "unknown error"))
            process.join(timeout=2)
        except KeyboardInterrupt:
            stop_event.set()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            raise
        finally:
            event_queue.close()

        if result_status == "completed" and worker_finished:
            if process.exitcode not in (0, None):
                LOGGER.warning("Worker exited abnormally after reporting success (code %s).", process.exitcode)
            return 0
        if process.exitcode not in (0, None):
            LOGGER.error("Worker process exited with code %s.", process.exitcode)
        return 130 if result_status == "stopped" else 1
    except KeyboardInterrupt:
        LOGGER.warning("User interrupted, program exited.")
        return 130
    except Exception as exc:
        LOGGER.exception("Processing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
