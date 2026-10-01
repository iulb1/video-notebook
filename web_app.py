from __future__ import annotations

import datetime
import faulthandler
import logging
import multiprocessing
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

faulthandler.enable()

import flask
from flask import Flask, jsonify, render_template, request

from asr_backends import ASRBackendError, normalize_backend_name, resolve_model_name
from transcription_worker import run_worker_process
from video_transcriber import VideoTranscriber, natural_sort_text_key

app = Flask(__name__)

worker_thread: threading.Thread | None = None
worker_process = None
stop_event = None
log_queue: queue.Queue[str] = queue.Queue()
processing = False

# 保护 task_queue / processing / _task_id_counter / progress 的并发访问
# （app.run(threaded=True) 下多个请求线程会同时读写这些全局变量）
_state_lock = threading.Lock()

# 全局复用的转写器实例，避免每次队列处理都重新加载/销毁模型。
# 关键：faster-whisper 模型在析构时会访问 CUDA 资源，可能触发段错误
# 导致整个进程退出（faulthandler 检测到后直接终止）。保留全局引用可避免析构。
_worker_transcribers: dict[tuple[str, str, str], VideoTranscriber] = {}

_task_id_counter = 0
task_queue: list[dict] = []

progress = {
    "current_file": "",
    "total": 0,
    "completed": 0,
    "percentage": 0,
    "status": "idle",
    "queue_total": 0,
    "queue_completed": 0,
}


class WebLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        log_queue.put(self.format(record))


def setup_logger(name: str = "web_transcriber") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, WebLogHandler) for h in logger.handlers):
        handler = WebLogHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger


web_logger = setup_logger()


def _log_exception(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    web_logger.critical("未捕获的异常 (excepthook):\n%s", tb)


def _log_thread_exception(args):
    web_logger.critical("线程未捕获的异常 (threading.excepthook): exc_info=%s",
                        args.exc_info)


sys.excepthook = _log_exception
threading.excepthook = _log_thread_exception


def progress_callback(current: int, total: int, file_name: str):
    with _state_lock:
        progress["current_file"] = file_name
        progress["total"] = total
        progress["completed"] = current
        if progress.get("queue_total", 0) == 0:
            progress["percentage"] = int(current / total * 100) if total > 0 else 0


def build_config(data: dict) -> dict:
    backend = normalize_backend_name(data.get("backend"))
    model = resolve_model_name(backend, data.get("model"))
    return {
        "input": data.get("input", "").strip(),
        "output": data.get("output", "").strip(),
        "device": data.get("device", "cuda"),
        "backend": backend,
        "model": model,
        "language": data.get("language") or None,
        "ffmpeg_path": data.get("ffmpeg_path"),
        "save_keyframes": data.get("save_keyframes", True),
        "keyframe_frequency": data.get("keyframe_frequency", "normal"),
        "skip_processed": data.get("skip_processed", True),
        "fmt_md": data.get("fmt_md", True),
        "fmt_json": data.get("fmt_json", False),
        "fmt_srt": data.get("fmt_srt", False),
        "fmt_docx": data.get("fmt_docx", False),
        "trash_source": data.get("trash_source", False),
        "vad_filter": data.get("vad_filter", False),
    }


def model_key(config: dict) -> tuple[str, str, str]:
    return (
        config.get("backend", "sensevoice"),
        config.get("model", "iic/SenseVoiceSmall"),
        config.get("device", "cuda"),
    )


def queue_model_conflict(config: dict) -> bool:
    requested = model_key(config)
    return any(model_key(item["config"]) != requested for item in task_queue)


def apply_runtime_config(tr: VideoTranscriber, cfg: dict, exporter, formats: set[str]) -> None:
    tr.language = None if cfg.get("language") in (None, "", "auto") else cfg.get("language")
    tr.save_keyframes = cfg.get("save_keyframes", True)
    tr.keyframe_threshold = get_keyframe_threshold(cfg.get("keyframe_frequency", "normal"))
    tr.ffmpeg_path = VideoTranscriber.resolve_ffmpeg_path_static(cfg.get("ffmpeg_path"))
    tr.exporter = exporter
    effective_formats = set(formats)
    if "docx" in effective_formats:
        effective_formats.add("md")
    tr.output_formats = effective_formats
    tr.trash_source = cfg.get("trash_source", False)
    tr.vad_filter = cfg.get("vad_filter", False)


def _run_single(tr: VideoTranscriber, config: dict) -> str:
    try:
        tr.trash_source = config.get("trash_source", False)
        success = tr.process_path(
            input_path=config["input"],
            output_dir=config["output"],
            skip_processed=config.get("skip_processed", True),
            progress_callback=progress_callback,
        )
        return "completed" if success else "failed"
    except Exception as e:
        web_logger.error("处理失败: %s", e)
        return "failed"


_queue_log = Path(__file__).parent / "queue_debug.log"


def _dump_queue_state():
    try:
        state = [{"id": t["id"], "s": t["status"]} for t in task_queue]
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with _queue_log.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] queue={state}\n")
    except Exception:
        pass


def _legacy_queue_worker():
    global processing, progress
    return
    tr: VideoTranscriber | None = None
    try:
        _queue_log.write_text("")
        _dump_queue_state()

        with _state_lock:
            first = next((t for t in task_queue if t["status"] == "pending"), None)
        if first is None:
            return
        cfg = first["config"]

        formats: set[str] = set()
        if cfg.get("fmt_md", True):
            formats.add("md")
        if cfg.get("fmt_json", False):
            formats.add("json")
        if cfg.get("fmt_srt", False):
            formats.add("srt")
        if cfg.get("fmt_docx", True):
            formats.add("docx")

        exporter = DocumentExporter(pandoc_path=cfg.get("pandoc_path")) if cfg.get("fmt_docx") else None

        cache_key = model_key(cfg)

        # 按后端、模型和设备复用实例，保留引用以规避 CUDA 模型析构崩溃。
        if cache_key in _worker_transcribers:
            tr = _worker_transcribers[cache_key]
            apply_runtime_config(tr, cfg, exporter, formats)
            web_logger.info("复用已有 ASR 实例 (%s / %s / %s)", *cache_key)
        else:
            tr = VideoTranscriber(
                model_size=cfg.get("model"),
                backend=cfg.get("backend", "faster-whisper"),
                device=cfg.get("device", "cuda"),
                language=cfg.get("language"),
                save_keyframes=cfg.get("save_keyframes", True),
                keyframe_threshold=get_keyframe_threshold(cfg.get("keyframe_frequency", "normal")),
                ffmpeg_path=cfg.get("ffmpeg_path"),
                exporter=exporter,
                output_formats=formats,
                trash_source=cfg.get("trash_source", False),
                vad_filter=cfg.get("vad_filter", False),
                logger=web_logger,
                stop_event=stop_event,
            )
            actual_key = (tr.backend_name, tr.model_size, tr.device)
            _worker_transcribers[cache_key] = tr
            _worker_transcribers.setdefault(actual_key, tr)
            web_logger.info("创建新的 ASR 实例 (%s / %s / %s)", *actual_key)

        with _state_lock:
            pending_count = sum(1 for t in task_queue if t["status"] == "pending")
            progress["queue_total"] = pending_count
            progress["queue_completed"] = 0
            progress["percentage"] = 0

        while True:
            with _state_lock:
                pending = [t for t in task_queue if t["status"] == "pending"]
                if not pending:
                    break
                item = pending[0]
                item["status"] = "running"
                progress["status"] = "running"
                progress["current_file"] = Path(item["input"]).name
                progress["total"] = 0
                progress["completed"] = 0
            _dump_queue_state()
            web_logger.info("[队列] 开始处理 #%d: %s", item["id"], item["input"])

            item_cfg = item["config"]
            item_formats: set[str] = set()
            for enabled, name in (
                (item_cfg.get("fmt_md", True), "md"),
                (item_cfg.get("fmt_json", False), "json"),
                (item_cfg.get("fmt_srt", False), "srt"),
                (item_cfg.get("fmt_docx", False), "docx"),
            ):
                if enabled:
                    item_formats.add(name)
            item_exporter = (
                DocumentExporter(pandoc_path=item_cfg.get("pandoc_path"))
                if item_cfg.get("fmt_docx")
                else None
            )
            apply_runtime_config(tr, item_cfg, item_exporter, item_formats)

            result = _run_single(tr, item["config"])
            web_logger.info("[队列] #%d 处理完毕: %s", item["id"], result)

            with _state_lock:
                item["status"] = result
                progress["queue_completed"] += 1
                if progress["queue_total"] > 0:
                    progress["percentage"] = int(progress["queue_completed"] / progress["queue_total"] * 100)
            _dump_queue_state()
    except Exception as exc:
        web_logger.error("队列 worker 异常: %s", exc)
        _dump_queue_state()
    finally:
        # 注意：不要在这里 del tr 或调用 torch.cuda.empty_cache()，
        # 这两个操作可能触发 C 扩展级别的段错误（faster-whisper 模型析构
        # 时访问 CUDA 资源），faulthandler 会使整个进程直接退出。
        # 让 Python GC 在后台自然回收即可。
        with _state_lock:
            processing = False
            if progress["status"] != "stopped":
                progress["status"] = "completed"
                progress["percentage"] = 100
        web_logger.info("队列处理完毕")
        _dump_queue_state()


def queue_worker():
    global processing, progress, worker_process, stop_event
    context = multiprocessing.get_context("spawn")
    event_queue = context.Queue()
    local_stop_event = context.Event()
    stop_event = local_stop_event
    finished_message = False
    try:
        with _state_lock:
            pending = [item for item in task_queue if item["status"] == "pending"]
            jobs = [
                {"task_id": item["id"], "config": dict(item["config"])}
                for item in pending
            ]
            progress["queue_total"] = len(jobs)
            progress["queue_completed"] = 0
            progress["percentage"] = 0
        if not jobs:
            return

        worker_process = context.Process(
            target=run_worker_process,
            args=(jobs, event_queue, local_stop_event),
            name="video-notebook-worker",
        )
        worker_process.start()

        dead_polls = 0
        while True:
            try:
                event = event_queue.get(timeout=0.2)
            except queue.Empty:
                if worker_process.is_alive():
                    dead_polls = 0
                    continue
                dead_polls += 1
                if dead_polls >= 5:
                    break
                continue
            dead_polls = 0
            event_type = event.get("type")
            if event_type == "log":
                log_queue.put(event.get("message", ""))
            elif event_type == "ready":
                web_logger.info(
                    "ASR 子进程已就绪 (%s / %s / %s)",
                    event.get("backend"), event.get("model"), event.get("device"),
                )
            elif event_type == "task_started":
                with _state_lock:
                    item = next((item for item in task_queue if item["id"] == event["task_id"]), None)
                    if item:
                        item["status"] = "running"
                    progress["status"] = "running"
                    progress["current_file"] = event.get("file_name", "")
                    progress["total"] = 0
                    progress["completed"] = 0
            elif event_type == "progress":
                with _state_lock:
                    progress["current_file"] = event.get("file_name", "")
                    progress["total"] = event.get("total", 0)
                    progress["completed"] = event.get("current", 0)
            elif event_type == "task_result":
                with _state_lock:
                    item = next((item for item in task_queue if item["id"] == event["task_id"]), None)
                    if item:
                        item["status"] = event.get("status", "failed")
                    progress["queue_completed"] += 1
                    if progress["queue_total"]:
                        progress["percentage"] = int(
                            progress["queue_completed"] / progress["queue_total"] * 100
                        )
            elif event_type == "finished":
                finished_message = True
            elif event_type == "fatal":
                web_logger.error("推理子进程失败: %s", event.get("message", "未知错误"))

        worker_process.join(timeout=2)
        if not finished_message:
            web_logger.error("推理子进程意外退出，退出码: %s", worker_process.exitcode)
            with _state_lock:
                for item in task_queue:
                    if item["status"] == "running":
                        item["status"] = "failed"
    except Exception as exc:
        web_logger.error("队列监督线程异常: %s", exc)
    finally:
        with _state_lock:
            processing = False
            if progress["status"] != "stopped":
                progress["status"] = "completed" if finished_message else "failed"
                if finished_message:
                    progress["percentage"] = 100
        event_queue.close()
        worker_process = None
        stop_event = None
        web_logger.info("队列处理结束")
        _dump_queue_state()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    global _task_id_counter, processing

    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的请求数据"}), 400

    input_path = data.get("input", "").strip()
    output_path = data.get("output", "").strip()

    if not input_path:
        return jsonify({"error": "输入路径不能为空"}), 400

    if not os.path.exists(input_path):
        return jsonify({"error": f"输入路径不存在: {input_path}"}), 400

    if not output_path:
        output_path = input_path
        if os.path.isfile(input_path):
            output_path = str(Path(input_path).parent)

    try:
        config = build_config(data)
    except ASRBackendError as exc:
        return jsonify({"error": str(exc)}), 400
    config["output"] = output_path

    with _state_lock:
        if queue_model_conflict(config):
            return jsonify({"error": "当前队列已锁定其他模型，请先移除队列中的任务"}), 409
        _task_id_counter += 1
        item = {
            "id": _task_id_counter,
            "input": input_path,
            "config": config,
            "status": "pending",
        }
        task_queue.append(item)
        queue_size = len(task_queue)

    web_logger.info("已加入队列 [#%d] %s", item["id"], input_path)
    return jsonify({"status": "queued", "id": item["id"], "queue_size": queue_size})


@app.route("/api/queue/start", methods=["POST"])
def api_queue_start():
    global processing, worker_thread

    with _state_lock:
        pending = [t for t in task_queue if t["status"] == "pending"]
        pending_count = len(pending)
        if not pending:
            return jsonify({"error": "队列为空"}), 400

        if processing:
            return jsonify({"error": "队列已在运行"}), 400

        processing = True
        progress["status"] = "running"
        progress["current_file"] = ""
        progress["total"] = pending_count
        progress["completed"] = 0
        progress["percentage"] = 0

    while not log_queue.empty():
        log_queue.get()

    worker_thread = threading.Thread(target=queue_worker, daemon=True)
    worker_thread.start()

    return jsonify({"status": "started", "queue_size": pending_count})


@app.route("/api/queue")
def api_queue():
    with _state_lock:
        items = []
        for t in task_queue:
            name = Path(t["input"]).name if t["input"] else t["input"]
            items.append({
                "id": t["id"],
                "input": t["input"],
                "name": name,
                "status": t["status"],
            })
        total = len(task_queue)
        completed_count = sum(1 for t in task_queue if t["status"] == "completed")
        running = processing or any(t["status"] == "running" for t in task_queue)
        is_processing = processing

    return jsonify({
        "items": items,
        "total": total,
        "completed": completed_count,
        "running": running,
        "processing": is_processing,
    })


@app.route("/api/queue/remove", methods=["POST"])
def api_queue_remove():
    data = request.get_json()
    item_id = data.get("id")
    if not item_id:
        return jsonify({"error": "缺少 id"}), 400

    with _state_lock:
        target = None
        for t in task_queue:
            if t["id"] == item_id:
                target = t
                break

        if target is None:
            return jsonify({"error": "未找到该任务"}), 404

        if target["status"] == "running":
            return jsonify({"error": "不能移除正在运行的任务"}), 400

        # 原地移除，避免 global task_queue 重建导致的引用竞争
        task_queue[:] = [t for t in task_queue if t["id"] != item_id]

    web_logger.info("已移除队列项 [#%d]", item_id)
    return jsonify({"status": "removed"})


@app.route("/api/scan-subdirs", methods=["POST"])
def api_scan_subdirs():
    data = request.get_json()
    parent = (data.get("path") or "").strip()
    if not parent or not os.path.isdir(parent):
        return jsonify({"error": "无效的目录路径"}), 400

    dirs = []
    try:
        for entry in os.scandir(parent):
            if entry.is_dir() and not entry.name.startswith("."):
                dirs.append({"name": entry.name, "path": entry.path})
    except OSError as e:
        return jsonify({"error": f"扫描目录失败: {e}"}), 500

    # 自然排序，与处理顺序保持一致（01、02、10）
    dirs.sort(key=lambda d: natural_sort_text_key(d["name"]))
    return jsonify({"dirs": dirs})


@app.route("/api/batch-enqueue", methods=["POST"])
def api_batch_enqueue():
    global _task_id_counter, processing

    data = request.get_json()
    items = data.get("items") if data else None
    if not items or not isinstance(items, list):
        return jsonify({"error": "无效的请求数据"}), 400

    added: list[dict] = []
    errors: list[dict] = []
    # 先在锁外做存在性校验和配置构建（涉及 IO，不要持锁）
    prepared: list[tuple[dict, str]] = []
    for entry in items:
        input_path = (entry.get("input") or "").strip()
        if not input_path or not os.path.exists(input_path):
            errors.append({"input": input_path, "error": "路径不存在"})
            continue

        try:
            cfg = build_config(entry)
        except ASRBackendError as exc:
            errors.append({"input": input_path, "error": str(exc)})
            continue
        output_path = (entry.get("output") or "").strip()
        if not output_path:
            # 批量添加默认：在各自的输入路径后追加 output 子目录
            output_path = str(Path(input_path) / "output")
        cfg["output"] = output_path
        prepared.append((cfg, input_path))

    with _state_lock:
        for cfg, input_path in prepared:
            if queue_model_conflict(cfg):
                errors.append({"input": input_path, "error": "当前队列已锁定其他模型"})
                continue
            _task_id_counter += 1
            item = {
                "id": _task_id_counter,
                "input": input_path,
                "config": cfg,
                "status": "pending",
            }
            task_queue.append(item)
            added.append({"id": item["id"], "input": input_path})

    web_logger.info("批量入队: 成功 %d 项, 失败 %d 项", len(added), len(errors))
    return jsonify({"added": added, "errors": errors})


@app.route("/api/ffmpeg-path")
def api_ffmpeg_path():
    try:
        path = VideoTranscriber.resolve_ffmpeg_path_static()
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"path": None, "error": str(e)})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global processing, progress
    with _state_lock:
        if processing:
            if stop_event is not None:
                stop_event.set()
            processing = False
            progress["status"] = "stopped"

        for t in task_queue:
            if t["status"] == "running":
                t["status"] = "stopped"
                break

    def terminate_after_grace():
        process = worker_process
        if process is None:
            return
        process.join(timeout=5)
        if process.is_alive():
            web_logger.warning("子进程未在 5 秒内停止，正在强制终止")
            process.terminate()

    threading.Thread(target=terminate_after_grace, daemon=True).start()

    return jsonify({"status": "stopped"})


@app.route("/api/logs")
def api_logs():
    global processing
    logs = []
    while not log_queue.empty():
        try:
            logs.append(log_queue.get_nowait())
        except queue.Empty:
            break

    with _state_lock:
        if processing and worker_thread and not worker_thread.is_alive():
            processing = False
            if progress["status"] == "running":
                progress["status"] = "completed"
                progress["percentage"] = 100

        total_queue = progress.get("queue_total", 0) or len(task_queue)
        completed_queue = progress.get("queue_completed", 0) or sum(1 for t in task_queue if t["status"] == "completed")
        resp = {
            "logs": logs,
            "processing": processing,
            "status": progress.get("status", "idle"),
            "current_file": progress.get("current_file", ""),
            "total": progress.get("total", 0),
            "completed": progress.get("completed", 0),
            "percentage": progress.get("percentage", 0),
            "queue_total": total_queue,
            "queue_completed": completed_queue,
        }

    return jsonify(resp)


def main():
    print("=" * 50)
    print("  音视频字幕转录工具 - Web UI")
    print("=" * 50)
    print(f"  请在浏览器中访问: http://127.0.0.1:5000")
    print("=" * 50)
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
