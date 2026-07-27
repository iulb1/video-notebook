from __future__ import annotations

import datetime
import faulthandler
import logging
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

faulthandler.enable()

import flask
from flask import Flask, jsonify, render_template, request

from video_transcriber import VideoTranscriber, get_keyframe_threshold
from document_exporter import DocumentExporter

app = Flask(__name__)

worker_thread: threading.Thread | None = None
stop_event = threading.Event()
log_queue: queue.Queue[str] = queue.Queue()
processing = False

# 全局复用的转写器实例，避免每次队列处理都重新加载/销毁模型。
# 关键：faster-whisper 模型在析构时会访问 CUDA 资源，可能触发段错误
# 导致整个进程退出（faulthandler 检测到后直接终止）。保留全局引用可避免析构。
_worker_transcriber: VideoTranscriber | None = None
_worker_device: str = ""

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
    global progress
    progress["current_file"] = file_name
    progress["total"] = total
    progress["completed"] = current
    if progress.get("queue_total", 0) == 0:
        progress["percentage"] = int(current / total * 100) if total > 0 else 0


def build_config(data: dict) -> dict:
    return {
        "input": data.get("input", "").strip(),
        "output": data.get("output", "").strip(),
        "device": data.get("device", "cuda"),
        "language": data.get("language") or None,
        "ffmpeg_path": data.get("ffmpeg_path"),
        "save_keyframes": data.get("save_keyframes", True),
        "keyframe_frequency": data.get("keyframe_frequency", "normal"),
        "skip_processed": data.get("skip_processed", True),
        "fmt_md": data.get("fmt_md", True),
        "fmt_json": data.get("fmt_json", False),
        "fmt_srt": data.get("fmt_srt", False),
        "fmt_docx": data.get("fmt_docx", True),
        "trash_source": data.get("trash_source", True),
    }


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


def queue_worker():
    global processing, progress, _worker_transcriber, _worker_device
    tr: VideoTranscriber | None = None
    try:
        _queue_log.write_text("")
        _dump_queue_state()

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

        requested_device = cfg.get("device", "cuda")

        # 复用已有模型实例（device 相同），避免反复加载/销毁模型导致段错误退出
        if _worker_transcriber is not None and _worker_device == requested_device:
            tr = _worker_transcriber
            # 运行时更新可变配置
            tr.language = None if cfg.get("language") in (None, "", "auto") else cfg.get("language")
            tr.save_keyframes = cfg.get("save_keyframes", True)
            tr.keyframe_threshold = get_keyframe_threshold(cfg.get("keyframe_frequency", "normal"))
            try:
                tr.ffmpeg_path = VideoTranscriber.resolve_ffmpeg_path_static(cfg.get("ffmpeg_path"))
            except Exception:
                pass
            tr.exporter = exporter
            effective_formats = set(formats)
            if "docx" in effective_formats:
                effective_formats.add("md")
            tr.output_formats = effective_formats
            tr.trash_source = cfg.get("trash_source", False)
            web_logger.info("复用已有的转写模型实例 (device=%s)", _worker_device)
        else:
            tr = VideoTranscriber(
                model_size="faster-whisper-small",
                device=requested_device,
                language=cfg.get("language"),
                save_keyframes=cfg.get("save_keyframes", True),
                keyframe_threshold=get_keyframe_threshold(cfg.get("keyframe_frequency", "normal")),
                ffmpeg_path=cfg.get("ffmpeg_path"),
                exporter=exporter,
                output_formats=formats,
                trash_source=cfg.get("trash_source", False),
                logger=web_logger,
                stop_event=stop_event,
            )
            _worker_transcriber = tr
            _worker_device = tr.device
            web_logger.info("创建新的转写模型实例 (device=%s)", _worker_device)

        pending_count = sum(1 for t in task_queue if t["status"] == "pending")
        progress["queue_total"] = pending_count
        progress["queue_completed"] = 0
        progress["percentage"] = 0

        while True:
            pending = [t for t in task_queue if t["status"] == "pending"]
            if not pending:
                break
            item = pending[0]
            item["status"] = "running"
            progress["status"] = "running"
            _dump_queue_state()
            web_logger.info("[队列] 开始处理 #%d: %s", item["id"], item["input"])

            progress["current_file"] = Path(item["input"]).name
            progress["total"] = 0
            progress["completed"] = 0

            result = _run_single(tr, item["config"])
            web_logger.info("[队列] #%d 处理完毕: %s", item["id"], result)
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
        processing = False
        if progress["status"] != "stopped":
            progress["status"] = "completed"
            progress["percentage"] = 100
        web_logger.info("队列处理完毕")
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

    config = build_config(data)
    config["output"] = output_path

    _task_id_counter += 1
    item = {
        "id": _task_id_counter,
        "input": input_path,
        "config": config,
        "status": "pending",
    }
    task_queue.append(item)

    web_logger.info("已加入队列 [#%d] %s", item["id"], input_path)
    return jsonify({"status": "queued", "id": item["id"], "queue_size": len(task_queue)})


@app.route("/api/queue/start", methods=["POST"])
def api_queue_start():
    global processing, worker_thread, stop_event

    pending = [t for t in task_queue if t["status"] == "pending"]
    if not pending:
        return jsonify({"error": "队列为空"}), 400

    if processing:
        return jsonify({"error": "队列已在运行"}), 400

    processing = True
    stop_event.clear()
    progress["status"] = "running"

    progress["current_file"] = ""
    progress["total"] = len(pending)
    progress["completed"] = 0
    progress["percentage"] = 0

    while not log_queue.empty():
        log_queue.get()

    worker_thread = threading.Thread(target=queue_worker, daemon=True)
    worker_thread.start()

    return jsonify({"status": "started", "queue_size": len(pending)})


@app.route("/api/queue")
def api_queue():
    total = len(task_queue)
    completed_count = sum(1 for t in task_queue if t["status"] == "completed")
    running = processing or any(t["status"] == "running" for t in task_queue)

    items = []
    for t in task_queue:
        name = Path(t["input"]).name if t["input"] else t["input"]
        items.append({
            "id": t["id"],
            "input": t["input"],
            "name": name,
            "status": t["status"],
        })

    return jsonify({
        "items": items,
        "total": total,
        "completed": completed_count,
        "running": running,
        "processing": processing,
    })


@app.route("/api/queue/remove", methods=["POST"])
def api_queue_remove():
    data = request.get_json()
    item_id = data.get("id")
    if not item_id:
        return jsonify({"error": "缺少 id"}), 400

    global task_queue
    target = None
    for t in task_queue:
        if t["id"] == item_id:
            target = t
            break

    if target is None:
        return jsonify({"error": "未找到该任务"}), 404

    if target["status"] == "running":
        return jsonify({"error": "不能移除正在运行的任务"}), 400

    task_queue = [t for t in task_queue if t["id"] != item_id]
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

    dirs.sort(key=lambda d: d["name"])
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
    for entry in items:
        input_path = (entry.get("input") or "").strip()
        if not input_path or not os.path.exists(input_path):
            errors.append({"input": input_path, "error": "路径不存在"})
            continue

        cfg = build_config(entry)
        output_path = (entry.get("output") or "").strip()
        if not output_path:
            output_path = input_path
        cfg["output"] = output_path

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
    if processing:
        stop_event.set()
        processing = False
        progress["status"] = "stopped"

    for t in task_queue:
        if t["status"] == "running":
            t["status"] = "stopped"
            break

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

    if processing and worker_thread and not worker_thread.is_alive():
        processing = False
        if progress["status"] == "running":
            progress["status"] = "completed"
            progress["percentage"] = 100

    total_queue = progress.get("queue_total", 0) or len(task_queue)
    completed_queue = progress.get("queue_completed", 0) or sum(1 for t in task_queue if t["status"] == "completed")

    return jsonify({
        "logs": logs,
        "processing": processing,
        "status": progress.get("status", "idle"),
        "current_file": progress.get("current_file", ""),
        "total": progress.get("total", 0),
        "completed": progress.get("completed", 0),
        "percentage": progress.get("percentage", 0),
        "queue_total": total_queue,
        "queue_completed": completed_queue,
    })


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