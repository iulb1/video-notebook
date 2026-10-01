from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from typing import Any


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def collect_status(backend: str, device: str) -> dict[str, Any]:
    required_modules = ["flask", "numpy", "PIL"]
    if backend == "sensevoice":
        required_modules.extend(["torch", "funasr", "modelscope", "kaldi_native_fbank"])
    else:
        required_modules.append("faster_whisper")

    modules = {name: module_available(name) for name in required_modules}
    status: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "backend": backend,
        "device": device,
        "modules": modules,
        "ffmpeg": shutil.which("ffmpeg"),
        "pandoc": shutil.which("pandoc"),
        "cuda_available": None,
        "gpu": None,
        "ok": all(modules.values()) and shutil.which("ffmpeg") is not None,
    }

    if device == "cuda":
        if not modules.get("torch", module_available("torch")):
            status["cuda_available"] = False
            status["ok"] = False
        else:
            try:
                import torch

                status["torch"] = torch.__version__
                status["torch_cuda"] = torch.version.cuda
                status["cuda_available"] = torch.cuda.is_available()
                if status["cuda_available"]:
                    status["gpu"] = torch.cuda.get_device_name(0)
                else:
                    status["ok"] = False
            except Exception as exc:
                status["cuda_available"] = False
                status["cuda_error"] = str(exc)
                status["ok"] = False
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="检查视频字幕工具运行环境")
    parser.add_argument("--backend", choices=["sensevoice", "faster-whisper"], default="sensevoice")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    status = collect_status(args.backend, args.device)

    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(f"Python: {status['python']} ({status['executable']})")
        for name, available in status["modules"].items():
            print(f"{'[OK]' if available else '[MISSING]'} Python module: {name}")
        print(f"{'[OK]' if status['ffmpeg'] else '[MISSING]'} FFmpeg: {status['ffmpeg'] or 'not found on PATH'}")
        print(f"{'[OK]' if status['pandoc'] else '[OPTIONAL]'} Pandoc: {status['pandoc'] or 'not installed (Word export only)'}")
        if args.device == "cuda":
            print(f"{'[OK]' if status['cuda_available'] else '[FAILED]'} CUDA: {status['gpu'] or 'unavailable'}")
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
