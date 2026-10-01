from __future__ import annotations

import ctypes
import functools
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from ctypes import wintypes
from pathlib import Path
from typing import Any

# 自动处理 Windows 下的 cuDNN/CUDA DLL 路径
if sys.platform == "win32":
    _checked_dll_dirs: set[str] = set()

    def _add_nvidia_dll_bin(bin_path: str) -> None:
        if not os.path.exists(bin_path) or bin_path in _checked_dll_dirs:
            return
        _checked_dll_dirs.add(bin_path)
        if bin_path not in os.environ["PATH"]:
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(bin_path)
            except Exception:
                pass

    # 扫描 site-packages/nvidia/*
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if os.path.exists(site_packages):
        for subfolder in os.listdir(site_packages):
            _add_nvidia_dll_bin(os.path.join(site_packages, subfolder, "bin"))

    # 扫描常见 conda 环境
    for _prefix in [r"C:\ProgramData\miniconda3", r"C:\ProgramData\anaconda3", os.path.expandvars(r"%USERPROFILE%\miniconda3")]:
        _conda_nvidia = os.path.join(_prefix, "Lib", "site-packages", "nvidia")
        if os.path.exists(_conda_nvidia):
            for subfolder in os.listdir(_conda_nvidia):
                _add_nvidia_dll_bin(os.path.join(_conda_nvidia, subfolder, "bin"))

# ── 常量 ──────────────────────────────────────────────
NO_WINDOW = subprocess.CREATE_NO_WINDOW

# ── Windows 回收站工具 ─────────────────────────────────
if sys.platform == "win32":
    _SHFileOperationW = ctypes.windll.shell32.SHFileOperationW
    _SHFileOperationW.restype = wintypes.BOOL
    _SHFileOperationW.argtypes = [ctypes.c_void_p]

    FO_DELETE = 3
    FOF_ALLOWUNDO = 0x40
    FOF_NOCONFIRMATION = 0x10
    FOF_SILENT = 0x04

    def move_to_trash(path: str) -> bool:
        """将文件或目录送回收站。"""
        buf = ctypes.c_wchar_p(path + "\0\0")

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", ctypes.c_uint),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", ctypes.c_ushort),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        fo = SHFILEOPSTRUCTW(
            hwnd=None,
            wFunc=FO_DELETE,
            pFrom=buf,
            pTo=None,
            fFlags=FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT,
            fAnyOperationsAborted=False,
            hNameMappings=None,
            lpszProgressTitle=None,
        )
        result = _SHFileOperationW(ctypes.byref(fo))
        return result == 0
else:
    def move_to_trash(path: str) -> bool:
        return False


import numpy as np
from PIL import Image

from asr_backends import ASRBackend, ASRBackendError, create_backend


class VideoProcessingError(Exception):
    """视频处理阶段的通用异常。"""


class AudioExtractionError(VideoProcessingError):
    """音频提取失败。"""


class TranscriptionError(VideoProcessingError):
    """音频转写失败。"""


class OutputWriteError(VideoProcessingError):
    """输出文件写入失败。"""


SegmentDict = dict[str, float | str]
TranscriptionMetadata = dict[str, str | float | int | None]


def safe_name(path: Path) -> str:
    """安全获取文件名，避免 Windows 中文路径下的编码异常。"""
    try:
        return os.fsdecode(path.name)
    except Exception:
        return str(path.name)


def get_keyframe_threshold(frequency: str) -> float:
    """根据截图频率名称返回对应的帧差阈值。"""
    mapping = {
        "few": 0.35,
        "normal": 0.15,
        "medium": 0.25,
        "many": 0.05,
        "all": 0.0,
    }
    return mapping.get(frequency, 0.15)


# Windows 文件名非法字符（含控制字符）——统一替换为下划线，
# 另见 Naming Conventions: https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file
_INVALID_WIN_CHARS = '<>:"/\\|?*\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f' \
    '\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'


def sanitize_stem(stem: str) -> str:
    """清理文件名/目录名 stem，确保在 Windows 上与磁盘实际路径一致。

    修复场景：
    - 末尾空格/点号：Windows 创建文件/目录时静默去除，
      导致 Python 侧的 `Path("a  ").exists()` 与磁盘真实状态脱节。
    - Windows 非法字符：<> : " / \\ | ? * 以及控制字符。
    - 清理后为空时回退为 md5 短哈希。
    """
    if stem is None:
        stem = ""
    # 1) 替换非法字符为下划线
    cleaned = stem.translate({ord(c): "_" for c in _INVALID_WIN_CHARS})
    # 2) 去掉末尾空格和点号（Windows 会静默去除，必须与磁盘一致）
    cleaned = cleaned.rstrip(" .\t\r\n")
    if cleaned:
        return cleaned
    # 3) 清理后为空，使用哈希兜底
    return hashlib.md5(stem.encode("utf-8", errors="ignore")).hexdigest()[:12]


def natural_key(text: str) -> list[object]:
    """生成自然排序键：数字段按数值比较，其余按不区分大小写比较。

    例如: "2" < "10"、"" < "01" < "02" < "10" < "a1" < "a2"。
    re.split 的捕获分组保证同一索引位置上两个键的元素类型一致
    （奇数位必为数字串、偶数位必为非数字串），因此列表比较不会
    出现 int 与 str 混比抛 TypeError 的情况。
    """
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", text)
    ]


def natural_sort_text_key(text: str) -> tuple[list[object], str]:
    """带原文回退的自然排序键。

    数字段相同时（如 "01" 与 "1"）按原文小写比较，保证排序结果
    与输入顺序无关（确定性强，对顺序命名 + 跳过判断的一致性很重要）。
    返回元组 (数字键, 原文)，两层比较类型各自一致，不会混比抛错。
    """
    return natural_key(text), text.casefold()


def natural_sort_key(path: Path) -> list[tuple[list[object], str]]:
    """按路径的每一段做自然排序，使 01/、02/、10/ 文件夹按数字序排列。

    逐段比较能保证相同前缀的子路径聚集在一起（文件按所在目录分组），
    与用户预期的"先按文件夹、再按文件夹内文件"的处理顺序一致。
    """
    return [natural_sort_text_key(part) for part in path.parts]


class VideoTranscriber:
    # 支持的媒体扩展名（类常量，避免每次实例化都重建集合）
    VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".ts"})
    AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"})

    def __init__(
        self,
        model_size: str | None = None,
        backend: str = "sensevoice",
        device: str = "cuda",
        language: str | None = "zh",
        save_keyframes: bool = True,
        keyframe_threshold: float = 0.25,
        ffmpeg_path: str | None = None,
        exporter: object | None = None,
        output_formats: set[str] | None = None,
        trash_source: bool = False,
        vad_filter: bool = False,
        logger: logging.Logger | None = None,
        stop_event: threading.Event | None = None,
        asr_backend: ASRBackend | None = None,
    ) -> None:
        """
        初始化视频转写器
        
        Args:
            model_size: ASR 模型名称或本地目录；为空时使用后端默认模型
            backend: ASR 后端名称（faster-whisper 或 sensevoice）
            device: 计算设备 (cuda 或 cpu)
            vad_filter: 是否开启 VAD 静音过滤（长静音视频可加速，但可能误删低音量语音）
            stop_event: 停止信号事件
            asr_backend: 可选的已构造后端，主要用于依赖注入和测试
        """
        self.logger = logger or logging.getLogger(__name__)
        self.stop_event = stop_event
        self.vad_filter = vad_filter
        self.language = None if language in (None, "", "auto") else language
        self.save_keyframes = save_keyframes
        self.keyframe_threshold = keyframe_threshold
        self.ffmpeg_path = self.resolve_ffmpeg_path(ffmpeg_path)
        self.exporter = exporter
        effective_formats = set(output_formats or {"md"})
        if "docx" in effective_formats:
            effective_formats.add("md")
        self.output_formats = effective_formats
        self.trash_source = trash_source
        self.backend_name = backend
        self.model_size = model_size
        self.device = device
        
        # 记录关键路径以方便调试
        self.logger.info("环境检查: Python=%s, Device=%s", sys.version.split()[0], device)
        
        try:
            self.asr_backend = asr_backend or create_backend(
                backend=backend,
                model_name=model_size,
                device=device,
                logger=self.logger,
                stop_event=stop_event,
            )
        except ASRBackendError as exc:
            raise VideoProcessingError(str(exc)) from exc
        self.backend_name = self.asr_backend.name
        self.model_size = self.asr_backend.model_name
        self.device = self.asr_backend.device
        self.engine = self.backend_name
        
        self.logger.info("模型加载完成")
        # video_extensions / audio_extensions 已提为类常量 VIDEO_EXTENSIONS / AUDIO_EXTENSIONS

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def resolve_ffmpeg_path_static(ffmpeg_path: str | None = None) -> str:
        candidates = [
            ffmpeg_path,
            os.environ.get("VIDEO_NOTEBOOK_FFMPEG"),
            "ffmpeg",
            "ffmpeg.exe",
        ]
        # 尝试扫描常见 ffmpeg 安装路径
        common_ffmpeg_dirs = [
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\ProgramData\chocolatey\bin",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages"),
            os.path.expandvars(r"%USERPROFILE%\scoop\apps\ffmpeg\current\bin"),
        ]
        for d in common_ffmpeg_dirs:
            expanded = os.path.expandvars(d)
            if not os.path.isdir(expanded):
                continue
            for root, dirs, files in os.walk(expanded):
                found = False
                for f in files:
                    if f.lower() == "ffmpeg.exe":
                        candidates.append(os.path.join(root, f))
                        found = True
                        break
                if found:
                    break

        for candidate in candidates:
            if not candidate:
                continue
            resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
            if resolved and Path(resolved).exists():
                return resolved
        raise VideoProcessingError(
            "ffmpeg executable not found. Install ffmpeg or set VIDEO_NOTEBOOK_FFMPEG / --ffmpeg."
        )

    def resolve_ffmpeg_path(self, ffmpeg_path: str | None) -> str:
        return self.resolve_ffmpeg_path_static(ffmpeg_path)

    def requested_language(self) -> str | None:
        return self.language

    def ensure_not_stopped(self) -> None:
        if self.stop_event and self.stop_event.is_set():
            raise VideoProcessingError("Processing stopped by user.")

    def deduplicate_repeated_text(self, text: str) -> str:
        """清理连续重复的短语，降低口吃式重复输出。"""
        normalized = " ".join(text.split())
        if not normalized:
            return normalized

        words = normalized.split(" ")
        cleaned: list[str] = []
        index = 0
        max_phrase_len = min(8, len(words) // 2) if len(words) > 1 else 1

        while index < len(words):
            removed = False
            for phrase_len in range(max_phrase_len, 0, -1):
                if index + phrase_len * 2 > len(words):
                    continue
                phrase = words[index : index + phrase_len]
                repeat_count = 1
                cursor = index + phrase_len
                while cursor + phrase_len <= len(words) and words[cursor : cursor + phrase_len] == phrase:
                    repeat_count += 1
                    cursor += phrase_len
                if repeat_count > 1:
                    cleaned.extend(phrase)
                    index = cursor
                    removed = True
                    break
            if not removed:
                cleaned.append(words[index])
                index += 1

        return " ".join(cleaned)

    def deduplicate_segments(self, results: list[SegmentDict]) -> list[SegmentDict]:
        """清理相邻重复片段，并对每段文本做轻量去重。"""
        deduplicated: list[SegmentDict] = []
        for item in results:
            cleaned_text = self.deduplicate_repeated_text(str(item["text"]))
            if not cleaned_text:
                continue

            current: SegmentDict = {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": cleaned_text,
            }
            if deduplicated and cleaned_text == str(deduplicated[-1]["text"]):
                deduplicated[-1]["end"] = float(current["end"])
                continue
            deduplicated.append(current)
        return deduplicated
    
    def is_video_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.VIDEO_EXTENSIONS

    def is_audio_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.AUDIO_EXTENSIONS

    def scan_media(self, input_dir: str) -> list[Path]:
        """
        递归扫描目录中的所有音视频文件
        
        Args:
            input_dir: 输入目录路径
            
        Returns:
            音视频文件路径列表
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        media_files: list[Path] = []
        all_extensions: set[str] = set()
        total_files = 0
        
        try:
            for file_path in input_path.rglob('*'):
                if not file_path.is_file():
                    continue
                total_files += 1
                ext = file_path.suffix.lower()
                all_extensions.add(ext)
                if self.is_video_file(file_path) or self.is_audio_file(file_path):
                    self.logger.info(f"  发现媒体文件: {safe_name(file_path)} ({ext})")
                    media_files.append(file_path)
        except Exception as e:
            self.logger.error(f"  目录扫描异常: {e}")
        
        self.logger.info(f"  扫描完成: 总文件数={total_files}, 媒体文件数={len(media_files)}, 所有扩展名={all_extensions}")
        
        # 自然排序（01/、02/、10/ 文件夹按数字序；文件夹内 1、2、10 同理）
        return sorted(media_files, key=natural_sort_key)
    
    def _run_ffmpeg_audio(self, cmd: list[str], temp_path: Path, video_name: str) -> bytes:
        """执行 ffmpeg 命令提取音频，返回读取到的 wav 数据（调用方负责删除临时文件）。"""
        self.logger.info("  执行 ffmpeg 命令: %s", " ".join(cmd))
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            text=False,
            timeout=3600,
            check=False,
            creationflags=NO_WINDOW,
        )
        if process.returncode != 0:
            error_msg = process.stderr.decode("utf-8", errors="ignore").strip()
            # 截断 ffmpeg 的巨量 AAC 解码错误，避免刷屏日志
            if len(error_msg) > 600:
                error_msg = error_msg[:600] + f"...（完整错误共 {len(error_msg)} 字符）"
            raise AudioExtractionError(
                f"ffmpeg 提取音频失败: {video_name}, 错误: {error_msg}"
            )
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise AudioExtractionError(f"音频提取结果为空: {video_name}")
        with wave.open(str(temp_path), "rb") as wf:
            return wf.readframes(wf.getnframes())

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        """静默删除临时文件，忽略文件不存在/被占用等错误。"""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _standard_audio_cmd(self, video_path_str: str, out_path: Path) -> list[str]:
        """标准音频提取命令（对正常文件无任何额外开销）。"""
        return [
            self.ffmpeg_path,
            "-y",
            "-i",
            video_path_str,
            "-f",
            "wav",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-loglevel",
            "error",
            str(out_path),
        ]

    def _tolerant_audio_cmd(self, video_path_str: str, out_path: Path) -> list[str]:
        """宽容模式：跳过损坏帧 + 单线程解码 + pan 强制单声道输出。"""
        return [
            self.ffmpeg_path,
            "-y",
            "-err_detect",
            "ignore_err",
            "-threads",
            "1",
            "-i",
            video_path_str,
            "-af",
            "pan=mono|c0=c0",
            "-ar",
            "16000",
            "-f",
            "wav",
            "-loglevel",
            "error",
            str(out_path),
        ]

    def _extract_audio_via_adts(self, video_path_str: str, temp_path: Path, video_name: str) -> bytes:
        """最后补救：极端损坏的 AAC 流（常见于被改名成 .mp4 的 MPEG-TS 文件）直接解码时，
        ffmpeg 8.x 解码器会把损坏帧误判为超大通道数（如 47/52 通道），导致自动重采样混流失败。
        先将音频流原样复制为 ADTS（不解码，必然成功），再解码 ADTS 即可正常跳过损坏帧。
        """
        adts_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as adts_handle:
                adts_path = Path(adts_handle.name)

            remux_cmd = [
                self.ffmpeg_path,
                "-y",
                "-i",
                video_path_str,
                "-vn",
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                "-f",
                "adts",
                "-loglevel",
                "error",
                str(adts_path),
            ]
            self.logger.info("  执行 ffmpeg 重封装命令: %s", " ".join(remux_cmd))
            process = subprocess.run(
                remux_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                text=False,
                timeout=3600,
                check=False,
                creationflags=NO_WINDOW,
            )
            if process.returncode != 0 or not adts_path.exists() or adts_path.stat().st_size == 0:
                raise AudioExtractionError(f"ADTS 重封装失败: {video_name}")

            # 删除可能残留的 wav，再解码 ADTS 写入目标文件
            self._safe_unlink(temp_path)
            decode_cmd = [
                self.ffmpeg_path,
                "-y",
                "-err_detect",
                "ignore_err",
                "-i",
                str(adts_path),
                "-f",
                "wav",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-loglevel",
                "error",
                str(temp_path),
            ]
            return self._run_ffmpeg_audio(decode_cmd, temp_path, video_name)
        finally:
            if adts_path is not None:
                self._safe_unlink(adts_path)

    def extract_audio(self, video_path: Path) -> np.ndarray[Any, Any]:
        """
        使用 ffmpeg 提取音频到内存，避免 PyAV 的编码问题。
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            16kHz 单声道 float32 音频数组
        """
        try:
            self.ensure_not_stopped()
            time.sleep(0.1)
            video_path_str = str(video_path)
            video_name = safe_name(video_path)
            self.logger.info(f"  使用 ffmpeg 提取音频: {video_name}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = Path(handle.name)

            try:
                # 降级策略链：标准 → 宽容 → ADTS 重封装，全部失败时上报首次（最可读）错误
                first_exc: AudioExtractionError | None = None
                strategies = [
                    ("标准参数", self._standard_audio_cmd),
                    ("宽容模式", self._tolerant_audio_cmd),
                ]
                for label, build_cmd in strategies:
                    self._safe_unlink(temp_path)
                    try:
                        audio_data = self._run_ffmpeg_audio(
                            build_cmd(video_path_str, temp_path), temp_path, video_name
                        )
                        if label != "标准参数":
                            self.logger.warning(
                                "  %s提取成功（可能丢失少量损坏片段）: %s", label, video_name
                            )
                        break
                    except AudioExtractionError as exc:
                        if first_exc is None:
                            first_exc = exc
                        self.logger.warning("  %s解码失败，尝试更宽容的策略...", label)
                else:
                    # 最后补救：ADTS 重封装后再解码
                    try:
                        audio_data = self._extract_audio_via_adts(
                            video_path_str, temp_path, video_name
                        )
                        self.logger.warning(
                            "  通过 ADTS 重封装成功提取（可能丢失少量损坏片段）: %s", video_name
                        )
                    except AudioExtractionError:
                        # 保留首次（更可读）的错误信息，避免多条超长日志叠加
                        if first_exc is None:
                            raise AudioExtractionError(
                                f"音频提取失败: {video_name}（无可用解码策略）"
                            )
                        raise first_exc
            finally:
                self._safe_unlink(temp_path)

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except AudioExtractionError:
            raise
        except Exception as exc:
            # 简化错误消息，避免编码问题
            video_name = safe_name(video_path)
            raise AudioExtractionError(f"音频提取失败: {video_name}") from exc

    def transcribe_audio(
        self, audio: np.ndarray[Any, Any], video_name: str
    ) -> tuple[list[SegmentDict], TranscriptionMetadata]:
        """通过可插拔 ASR 后端将音频转换为带时间戳文本。"""
        try:
            self.logger.info("  Transcribing: %s", video_name)
            self.ensure_not_stopped()
            result = self.asr_backend.transcribe(
                audio,
                language=self.requested_language(),
                vad_filter=self.vad_filter,
                media_name=video_name,
            )
            raw_results = result.segments
            metadata = result.metadata
        except Exception as exc:
            raise TranscriptionError(f"Transcription failed: {video_name}: {exc}") from exc

        results = self.deduplicate_segments(raw_results)
        metadata["segment_count"] = len(results)
        metadata["raw_segment_count"] = len(raw_results)
        return results, metadata



    def merge_segments(
        self,
        results: list[SegmentDict],
        max_gap: float = 1.2,
        max_chars_per_block: int = 120,
    ) -> list[SegmentDict]:
        """将短片段合并为更适合阅读的笔记块。"""
        merged: list[SegmentDict] = []
        for item in results:
            text = str(item["text"]).strip()
            if not text:
                continue

            current: SegmentDict = {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": text,
            }
            if not merged:
                merged.append(current)
                continue

            previous = merged[-1]
            gap = float(current["start"]) - float(previous["end"])
            combined_length = len(str(previous["text"])) + len(text)
            should_merge = gap <= max_gap and combined_length <= max_chars_per_block
            if should_merge:
                previous["end"] = float(current["end"])
                previous["text"] = f"{previous['text']} {text}".strip()
                continue
            merged.append(current)
        return merged

    def _extract_frame_at(self, video_path: Path, timestamp: float, output_path: Path) -> bool:
        """用 ffmpeg -ss 精准 seek 抽取指定时间点的一帧，返回是否成功。

        相比全量按 fps 抽帧，这种按需 seek 的方式只生成真正需要的少量帧，
        长视频下可从分钟级降到秒级，且磁盘 IO 大幅减少。
        -ss 放在 -i 之前使用 input seek（快速跳转到最近关键帧）。
        """
        cmd = [
            self.ffmpeg_path, "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=60, creationflags=NO_WINDOW
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0 and output_path.exists()

    def extract_note_keyframes(
        self,
        video_path: Path,
        note_blocks: list[SegmentDict],
        output_path: Path,
    ) -> list[SegmentDict]:
        if not self.save_keyframes or not note_blocks:
            return note_blocks

        try:
            video_stem = sanitize_stem(video_path.stem)
        except Exception:
            video_stem = hashlib.md5(str(video_path).encode('utf-8')).hexdigest()

        image_dir = output_path.parent / "assets" / video_stem
        image_dir_created = False
        targets = [(float(b["start"]) + float(b["end"])) / 2 for b in note_blocks]
        enriched_blocks = [dict(b, screenshot=None) for b in note_blocks]
        video_duration = float(note_blocks[-1]["end"]) if note_blocks else 0
        if video_duration <= 0:
            return enriched_blocks

        temp_dir: Path | None = None
        try:
            # ── 第 1 步：按候选时间戳用 ffmpeg -ss 精准 seek 抽帧 ──
            # （取代原来 fps=2 全量抽帧，长视频下大幅减少 IO 和耗时）
            temp_dir = Path(tempfile.gettempdir()) / f"kf_{os.getpid()}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            seg_duration = 600
            num_segs = max(1, int(video_duration) // seg_duration + 1)
            target_per_seg = 8

            candidates: list[tuple[float, Path]] = []
            frame_idx = 0
            for seg_idx in range(num_segs):
                start = seg_idx * seg_duration
                end = min(start + seg_duration, video_duration)
                if end <= start:
                    continue
                for i in range(target_per_seg):
                    # 在每段内均匀取 target_per_seg 个时间点（中点采样）
                    ts = start + (end - start) * (i + 0.5) / target_per_seg
                    out_frame = temp_dir / f"frame_{frame_idx:06d}.jpg"
                    frame_idx += 1
                    if self._extract_frame_at(video_path, ts, out_frame):
                        candidates.append((ts, out_frame))

            total_extracted = len(candidates)
            if not candidates:
                return enriched_blocks

            # ── 第 3 步：帧差去重 ──
            self.logger.info("  帧差去重前: candidates=%d, threshold=%.3f", len(candidates), self.keyframe_threshold)
            prev_arr = None
            deduped: list[tuple[float, Path]] = []
            for ts, fp in candidates:
                try:
                    with Image.open(fp) as img:
                        arr = np.array(img.convert('L').resize((100, 75)))
                    if prev_arr is not None:
                        diff = np.mean(np.abs(arr - prev_arr)) / 255.0
                        self.logger.debug("    帧差 @ %.1fs: %.4f (阈值 %.3f)", ts, diff, self.keyframe_threshold)
                        if diff < self.keyframe_threshold:
                            continue
                    deduped.append((ts, fp))
                    prev_arr = arr
                except Exception:
                    continue
            self.logger.info("  帧差去重后: deduped=%d", len(deduped))

            if not deduped:
                deduped = candidates[:1]

            # 总量上限
            max_frames = 10 * num_segs
            if len(deduped) > max_frames:
                deduped = deduped[::max(1, len(deduped) // max_frames)][:max_frames]

            # ── 第 4 步：创建目录、分配到笔记块 ──
            try:
                image_dir.mkdir(parents=True, exist_ok=True)
                image_dir_created = True
            except OSError as e:
                self.logger.warning("  无法创建截图目录 %s: %s", image_dir, e)
                return enriched_blocks

            assigned = 0
            for ts, src in deduped:
                best = min(
                    range(len(targets)),
                    key=lambda j: abs(targets[j] - ts) if enriched_blocks[j].get("screenshot") is None else float('inf'),
                )
                if enriched_blocks[best].get("screenshot") is not None:
                    continue
                dst = image_dir / f"{best + 1:03d}_{int(ts * 1000):09d}.jpg"
                try:
                    shutil.copy(src, dst)
                except OSError as e:
                    self.logger.warning("  截图拷贝失败 ts=%.1fs dst=%s: %s", ts, dst, e)
                    continue
                # 显式校验文件已落盘：避免写入失败/目录实际不存在时，
                # screenshot 字段变为悬空引用而后续 md 渲染阶段才报错
                if not dst.exists():
                    self.logger.warning("  截图未落盘 ts=%.1fs dst=%s，跳过分配", ts, dst)
                    continue
                enriched_blocks[best]["screenshot"] = os.path.relpath(
                    str(dst), str(output_path.parent)
                ).replace("\\", "/")
                assigned += 1

            self.logger.info("  关键帧: seek 抽 %d 帧 → 去重 %d → 分配到 %d 个笔记块",
                             total_extracted, len(deduped), assigned)

        except Exception as exc:
            self.logger.warning("关键帧提取失败: %s", exc)
        finally:
            if temp_dir and temp_dir.exists():
                for f in temp_dir.glob("frame_*.jpg"):
                    try:
                        f.unlink()
                    except OSError:
                        pass
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass
            if image_dir_created and image_dir.exists() and not any(image_dir.iterdir()):
                try:
                    image_dir.rmdir()
                except OSError:
                    pass

        return enriched_blocks

    def format_timestamp(self, seconds: float) -> str:
        """
        格式化时间戳为 HH:MM:SS.mmm 格式
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间戳字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def save_markdown(
        self,
        results: list[SegmentDict],
        output_path: Path,
        video_path: Path,
        metadata: TranscriptionMetadata,
    ) -> None:
        """
        保存为 Markdown 格式（优化版）
        
        Args:
            results: 转写结果
            output_path: 输出文件路径
            video_path: 视频文件路径
            metadata: 转写元数据
        """
        note_blocks = self.merge_segments(results)
        if self.is_video_file(video_path):
            note_blocks = self.extract_note_keyframes(video_path, note_blocks, output_path)
        else:
            note_blocks = [dict(block, screenshot=None) for block in note_blocks]
        detected_language = metadata.get("language") or "unknown"
        confidence = float(metadata.get("language_probability") or 0.0)
        duration = float(metadata.get("duration_seconds") or 0.0)
        raw_segment_count = int(metadata.get("raw_segment_count") or len(results))
        try:
            with output_path.open("w", encoding="utf-8") as file:
                file.write(f"# {video_path.stem} 字幕笔记\n\n")
                file.write("## 概览\n\n")
                file.write(f"- 原始片段数: {raw_segment_count}\n")
                file.write(f"- 去重后片段数: {len(results)}\n")
                file.write(f"- 笔记段落数: {len(note_blocks)}\n")
                file.write(
                    f"- 插图段落数: {sum(1 for item in note_blocks if item.get('screenshot'))}\n"
                )
                file.write(f"- 转写语言: {detected_language}\n")
                file.write(f"- 语言置信度: {confidence:.2%}\n")
                file.write(f"- 视频时长: {self.format_timestamp(duration)}\n\n")
                file.write("## 笔记内容\n\n")
                for idx, item in enumerate(note_blocks, 1):
                    timestamp = (
                        f"{self.format_timestamp(float(item['start']))} - "
                        f"{self.format_timestamp(float(item['end']))}"
                    )
                    file.write(f"### [{idx}] {timestamp}\n\n")
                    screenshot = item.get("screenshot")
                    if screenshot:
                        # 确保路径使用正斜杠
                        screenshot_str = str(screenshot).replace("\\", "/")
                        # 修复编码问题，使用正确的路径
                        file.write(f"![{video_path.stem}]({screenshot_str})\n\n")
                    file.write(f"{str(item['text']).strip()}\n\n")
                    file.write("---\n\n")
        except OSError as exc:
            if exc.errno == 13 or exc.errno == 5:
                self.logger.error("权限错误: 无法写入 Markdown 文件 %s。请检查该文件夹是否被百度网盘等程序占用，或者尝试以管理员身份运行程序。", output_path.name)
            raise OutputWriteError(f"Markdown 写入失败: {output_path.name}: {exc}") from exc

    def save_json(self, results: list[SegmentDict], output_path: Path) -> None:
        """
        保存为 JSON 格式
        
        Args:
            results: 转写结果
            output_path: 输出文件路径
        """
        try:
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, ensure_ascii=False, indent=2)
        except OSError as exc:
            if exc.errno == 13 or exc.errno == 5:
                self.logger.error("权限错误: 无法写入 JSON 文件 %s。请检查文件夹权限。", output_path.name)
            raise OutputWriteError(f"JSON 写入失败: {output_path.name}: {exc}") from exc

    def save_srt(self, results: list[SegmentDict], output_path: Path) -> None:
        """
        保存为 SRT 格式字幕文件
        
        Args:
            results: 转写结果列表，每个元素包含 start, end, text
            output_path: 输出文件路径
        """
        def format_srt_time(seconds: float) -> str:
            """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            whole_seconds = int(secs)
            milliseconds = int((secs - whole_seconds) * 1000)
            return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
        
        try:
            with output_path.open("w", encoding="utf-8") as file:
                for idx, item in enumerate(results, 1):
                    file.write(f"{idx}\n")
                    start_time = format_srt_time(float(item["start"]))
                    end_time = format_srt_time(float(item["end"]))
                    file.write(f"{start_time} --> {end_time}\n")
                    file.write(f"{item['text']}\n\n")
        except OSError as exc:
            if exc.errno == 13 or exc.errno == 5:
                self.logger.error("权限错误: 无法写入 SRT 文件 %s。请检查文件夹权限。", output_path.name)
            raise OutputWriteError(f"SRT 写入失败: {output_path.name}: {exc}") from exc

    def build_output_stem(self, media_path: Path, seq_index: int | None = None, batch_total: int = 0) -> str:
        """计算输出文件的基础名（md/json/srt/docx 共用，保证跳过判断一致）。

        Args:
            media_path: 音视频文件路径
            seq_index: 批次内序号（1 起始）。目录批处理时传入，
                       生成 "{固定宽度序号}_{原名}"；单文件处理传 None 保持原名。
            batch_total: 批次总文件数，用于动态决定序号位数
                         （保证 0999 < 1000 的排序正确）。

        Returns:
            清理后的输出基础名（不含扩展名）
        """
        try:
            original = sanitize_stem(os.fsdecode(media_path.stem))
        except Exception:
            original = hashlib.md5(str(media_path).encode("utf-8", errors="ignore")).hexdigest()
        if seq_index is None:
            return original
        width = max(3, len(str(batch_total)))
        return f"{seq_index:0{width}d}_{original}"

    def is_already_processed(self, media_path: Path, output_dir: Path, output_stem: str | None = None) -> bool:
        """
        检查视频是否已处理
        
        Args:
            media_path: 音视频文件路径
            output_dir: 输出目录
            output_stem: 预期输出基础名（含序号前缀）。为 None 时按原名推导，
                         与 process_media 的计算方式保持一致。
            
        Returns:
            如果已处理返回 True，否则返回 False
        """
        try:
            stem = output_stem if output_stem is not None else sanitize_stem(os.fsdecode(media_path.stem))
        except Exception:
            stem = hashlib.md5(str(media_path).encode('utf-8', errors='ignore')).hexdigest()
        md_path = output_dir / f"{stem}.md"
        json_path = output_dir / f"{stem}.json"
        srt_path = output_dir / f"{stem}.srt"
        docx_path = output_dir / f"{stem}.docx"

        required_paths: list[Path] = []
        if "md" in self.output_formats:
            required_paths.append(md_path)
        if "json" in self.output_formats:
            required_paths.append(json_path)
        if "srt" in self.output_formats:
            required_paths.append(srt_path)
        core_outputs_exist = all(path.exists() for path in required_paths)
        if not core_outputs_exist:
            return False
        if "docx" in self.output_formats and self.exporter is not None and not docx_path.exists():
            return False
        return True
    
    def process_media(
        self,
        media_path: Path,
        output_dir: Path,
        skip_processed: bool = True,
        seq_index: int | None = None,
        batch_total: int = 0,
    ) -> bool:
        """
        处理单个视频文件
        
        Args:
            media_path: 音视频文件路径
            output_dir: 输出目录
            skip_processed: 是否跳过已处理的文件
            seq_index: 批次内序号（1 起始）。目录批处理时传入，输出文件名为
                       "{固定宽度序号}_{原名}.md"，保证按名称排序即处理顺序。
                       单文件处理传 None，输出文件名保持原名。
            batch_total: 批次总文件数，用于决定序号位数
            
        Returns:
            处理成功返回 True，失败返回 False
        """
        # 安全处理媒体文件名，避免编码错误
        media_name = safe_name(media_path)

        # 输出基础名只计算一次，跳过判断与各格式保存共用，避免命名不一致
        output_stem = self.build_output_stem(media_path, seq_index, batch_total)

        # 检查是否已处理
        if skip_processed and self.is_already_processed(media_path, output_dir, output_stem):
            self.logger.info("[跳过] %s (已处理)", media_name)
            return True

        self.logger.info("[处理] %s", media_name)
        # 提前记录原始值，确保 finally 块中一定能访问到（避免 NameError）
        original_save_keyframes = self.save_keyframes
        try:
            if self.is_audio_file(media_path):
                self.save_keyframes = False

            audio = self.extract_audio(media_path)
            results, metadata = self.transcribe_audio(audio, media_name)
            if metadata["language"] != "unknown":
                self.logger.info(
                    "  语言: %s (置信度 %.2f%%)",
                    metadata["language"],
                    float(metadata["language_probability"] or 0.0) * 100,
                )
            if not results:
                self.logger.warning("[警告] 转写结果为空: %s", media_name)

            md_path = output_dir / f"{output_stem}.md"
            json_path = output_dir / f"{output_stem}.json"
            srt_path = output_dir / f"{output_stem}.srt"
            if "md" in self.output_formats:
                self.save_markdown(results, md_path, media_path, metadata)
            if "json" in self.output_formats:
                self.save_json(results, json_path)
            if "srt" in self.output_formats:
                self.save_srt(results, srt_path)
            if "docx" in self.output_formats and self.exporter is not None:
                docx_path = output_dir / f"{output_stem}.docx"
                self.exporter.export_docx(md_path, docx_path)
            self.logger.info("[完成] %s", media_name)

            if self.trash_source and sys.platform == "win32":
                try:
                    move_to_trash(str(media_path))
                    self.logger.info("  已移入回收站: %s", media_name)
                except Exception as e:
                    self.logger.warning("  移入回收站失败: %s", e)

            return True
        except Exception as exc:
            self.logger.error("[失败] %s: %s", media_name, exc)
            return False
        finally:
            self.save_keyframes = original_save_keyframes

    def process_directory(self, input_dir: str, output_dir: str, skip_processed: bool = True, progress_callback=None) -> bool:
        """处理目录中的所有视频。"""
        if self.stop_event and self.stop_event.is_set():
            self.logger.info("收到停止信号，取消处理目录")
            return False

        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f"无法创建输出目录 {output_path}: {e}")
            return False

        media_files = self.scan_media(input_dir)
        if not media_files:
            self.logger.warning("在目录 %s 中未找到音视频文件", input_dir)
            return False

        self.logger.info("找到 %s 个音视频文件", len(media_files))
        success_count = 0
        batch_total = len(media_files)

        for i, media_path in enumerate(media_files, 1):
            if self.stop_event and self.stop_event.is_set():
                self.logger.info("收到停止信号，停止后续文件处理")
                break
            if self.process_media(
                media_path, output_path, skip_processed,
                seq_index=i, batch_total=batch_total,
            ):
                success_count += 1
            if progress_callback:
                progress_callback(i, media_path.name)

        self.logger.info("处理完成，成功: %s/%s", success_count, len(media_files))
        return success_count == len(media_files)

    def process_path(self, input_path: str, output_dir: str, skip_processed: bool = True, progress_callback=None) -> bool:
        """处理单个文件或目录。

        Args:
            input_path: 输入文件或目录路径
            output_dir: 输出目录
            skip_processed: 是否跳过已处理文件
            progress_callback: 进度回调函数，签名 (current, total, file_name)
        """
        path = Path(input_path)
        output_path = Path(output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f"无法创建输出目录 {output_path}: {e}")
            return False

        if path.is_file():
            result = self.process_media(path, output_path, skip_processed)
            if progress_callback:
                progress_callback(1, 1, path.name)
            return result
        if path.is_dir():
            if progress_callback:
                media_files = self.scan_media(str(path))
                total = len(media_files)
                wrapped_cb = lambda i, f: progress_callback(i, total, f)
                return self.process_directory(str(path), str(output_path), skip_processed, wrapped_cb)
            return self.process_directory(str(path), str(output_path), skip_processed)
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
