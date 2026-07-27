from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import logging
import os
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
        op = (ctypes.c_ushort * 7)(
            ctypes.c_ushort(0),          # hwnd
            ctypes.c_ushort(FO_DELETE),   # wFunc
            ctypes.sizeof(buf) // 2,      # pFrom 是 buf 地址低16位
            # 实际上我们应该用 SHFILEOPSTRUCTW
        )
        # 直接用 SHFILEOPSTRUCTW 更可靠
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

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from tqdm import tqdm


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


class VideoTranscriber:
    def __init__(
        self,
        model_size: str = "faster-whisper-small",
        device: str = "cuda",
        language: str | None = "zh",
        save_keyframes: bool = True,
        keyframe_threshold: float = 0.25,
        ffmpeg_path: str | None = None,
        exporter: object | None = None,
        output_formats: set[str] | None = None,
        trash_source: bool = True,
        logger: logging.Logger | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        初始化视频转写器
        
        Args:
            model_size: ASR 模型名称 (默认为 ParakeetTDT-0.6B-v3)
            device: 计算设备 (cuda 或 cpu)
            stop_event: 停止信号事件
        """
        self.logger = logger or logging.getLogger(__name__)
        self.stop_event = stop_event
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
        self.model_size = model_size
        self.device = device
        self.engine = "faster-whisper"
        
        # 记录关键路径以方便调试
        self.logger.info("环境检查: Python=%s, Device=%s", sys.version.split()[0], device)
        
        # Only Faster Whisper model is supported
        self.logger.info("Initializing Faster Whisper model (%s)...", model_size)
        try:
            if WhisperModel is None:
                raise VideoProcessingError(
                    "faster-whisper not found. Please run `pip install -r requirements.txt` or `pip install -U faster-whisper`"
                )

            model_size_name = model_size.split("-")[-1] if "-" in model_size else "small"
            
            self._init_model(model_size_name)
        except Exception as exc:
            # GPU 初始化失败时自动回退到 CPU
            if self.device == "cuda":
                self.logger.warning("GPU 初始化失败 (%s)，自动回退到 CPU...", exc)
                self.device = "cpu"
                try:
                    self._init_model(model_size_name)
                except Exception as cpu_exc:
                    raise VideoProcessingError(
                        f"模型初始化失败 (GPU 和 CPU 均失败): {cpu_exc}"
                    ) from cpu_exc
            else:
                raise VideoProcessingError(f"模型初始化失败: {exc}") from exc
        
        self.logger.info("模型加载完成")
        self.video_extensions = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".ts"}
        self.audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

    @staticmethod
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

    def _init_model(self, model_size_name: str) -> None:
        requested_device = "cuda" if self.device == "cuda" else "cpu"
        compute_type = "int8_float16" if requested_device == "cuda" else "int8"
        self.model = WhisperModel(
            model_size_name,
            device=requested_device,
            compute_type=compute_type,
        )
        self.logger.info(
            "  Faster Whisper model loaded successfully (device: %s, compute_type: %s)",
            requested_device,
            compute_type,
        )

    def requested_language(self) -> str | None:
        return self.language

    def ensure_not_stopped(self) -> None:
        if self.stop_event and self.stop_event.is_set():
            raise VideoProcessingError("Processing stopped by user.")

    def should_use_tqdm(self) -> bool:
        """仅在可写控制台存在时启用 tqdm。"""
        stream = sys.stderr
        return bool(stream and hasattr(stream, "write") and hasattr(stream, "isatty") and stream.isatty())

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
        return path.suffix.lower() in self.video_extensions

    def is_audio_file(self, path: Path) -> bool:
        return path.suffix.lower() in self.audio_extensions

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
                    try:
                        file_name = os.fsdecode(file_path.name)
                    except:
                        file_name = str(file_path.name)
                    self.logger.info(f"  发现媒体文件: {file_name} ({ext})")
                    media_files.append(file_path)
        except Exception as e:
            self.logger.error(f"  目录扫描异常: {e}")
        
        self.logger.info(f"  扫描完成: 总文件数={total_files}, 媒体文件数={len(media_files)}, 所有扩展名={all_extensions}")
        
        return sorted(media_files)
    
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
            video_name = "未知文件"
            try:
                video_name = os.fsdecode(video_path.name)
            except:
                pass
            self.logger.info(f"  使用 ffmpeg 提取音频: {video_name}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                temp_path = Path(handle.name)

            cmd = [
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
                str(temp_path),
            ]
            self.logger.info("  执行 ffmpeg 命令: %s", " ".join(cmd))
            try:
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
                    raise AudioExtractionError(f"ffmpeg 提取音频失败: {video_name}, 错误: {error_msg}")

                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    raise AudioExtractionError(f"音频提取结果为空: {video_name}")
                with wave.open(str(temp_path), "rb") as wf:
                    audio_data = wf.readframes(wf.getnframes())
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            return audio_array
        except AudioExtractionError:
            raise
        except Exception as exc:
            # 简化错误消息，避免编码问题
            video_name = "未知文件"
            try:
                video_name = os.fsdecode(video_path.name)
            except:
                pass
            raise AudioExtractionError(f"音频提取失败: {video_name}") from exc

    def transcribe_audio(
        self, audio: np.ndarray[Any, Any], video_name: str
    ) -> tuple[list[SegmentDict], TranscriptionMetadata]:
        """
        Transcribe audio to text with timestamps.
        """
        try:
            self.logger.info("  Transcribing: %s", video_name)
            self.ensure_not_stopped()

            # Calculate audio duration
            duration_seconds = float(len(audio) / 16000.0)
            self.logger.info("  Audio duration: %.2f seconds", duration_seconds)

            # Only Faster Whisper model is supported
            # Faster Whisper expects audio as a numpy array with shape (n,) and sample rate 16000
            
            # Split long audio into chunks to reduce memory usage
            chunk_duration = 30
            chunk_size = chunk_duration * 16000
            total_chunks = max(1, (len(audio) + chunk_size - 1) // chunk_size)
            
            if total_chunks > 1:
                self.logger.info("  Splitting audio into %d chunks to reduce memory usage", total_chunks)
                raw_results = []
                language = None
                language_probability = 0.0
                
                for i in range(total_chunks):
                    self.ensure_not_stopped()
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(audio))
                    chunk_audio = audio[start_idx:end_idx]
                    if len(chunk_audio) == 0:
                        continue
                    chunk_start_time = start_idx / 16000.0
                    
                    self.logger.info("  Processing chunk %d/%d (%.2f-%.2f seconds)", 
                                  i + 1, total_chunks, chunk_start_time, chunk_start_time + len(chunk_audio)/16000.0)
                    
                    segments, info = self.model.transcribe(
                        chunk_audio,
                        language=self.requested_language(),
                        beam_size=3,
                        word_timestamps=False,
                    )
                    
                    if i == 0:
                        language = info.language
                        language_probability = info.language_probability
                    
                    for segment in segments:
                        segment_dict = {
                            "text": segment.text,
                            "start": segment.start + chunk_start_time,
                            "end": segment.end + chunk_start_time,
                        }
                        raw_results.append(segment_dict)
            else:
                # Process as a single chunk
                segments, info = self.model.transcribe(
                    audio,
                    language=self.requested_language(),
                    beam_size=3,
                    word_timestamps=False,
                )
                raw_results = []
                for segment in segments:
                    raw_results.append({
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": str(segment.text).strip(),
                    })
                language = info.language
                language_probability = info.language_probability
            
            del audio
            
            metadata = {
                "language": language,
                "language_probability": language_probability,
                "duration_seconds": duration_seconds,
                "segment_count": len(raw_results),
                "raw_segment_count": len(raw_results),
                "language_mode": self.language or "auto",
            }
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

    def _extract_all_frames(
        self, video_path: Path, fps: float, output_dir: Path
    ) -> list[tuple[float, Path]]:
        """单次 ffmpeg 解码，按 fps 抽帧到 output_dir，返回 [(时间戳, 路径)]。"""
        cmd = [
            self.ffmpeg_path, "-y", "-i", str(video_path),
            "-vf", f"fps={fps}", "-q:v", "2",
            str(output_dir / "frame_%09d.jpg"),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=3600,
                                creationflags=NO_WINDOW)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="ignore").strip()
            raise VideoProcessingError(f"ffmpeg 抽帧失败: {err}")

        frames: list[tuple[float, Path]] = []
        for fp in sorted(output_dir.glob("frame_*.jpg")):
            idx = int(fp.stem.split("_")[1])
            frames.append((idx / fps, fp))
        return frames

    def extract_note_keyframes(
        self,
        video_path: Path,
        note_blocks: list[SegmentDict],
        output_path: Path,
    ) -> list[SegmentDict]:
        if not self.save_keyframes or not note_blocks:
            return note_blocks

        try:
            video_stem = video_path.stem
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
            # ── 第 1 步：单次 ffmpeg 抽帧 ──
            temp_dir = Path(tempfile.gettempdir()) / f"kf_{os.getpid()}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            fps = 2
            all_frames = self._extract_all_frames(video_path, fps, temp_dir)
            if not all_frames:
                return enriched_blocks

            # ── 第 2 步：每 10 分钟段均匀选 8 张候选 ──
            seg_duration = 600
            num_segs = max(1, int(video_duration) // seg_duration + 1)
            target_per_seg = 8

            candidates: list[tuple[float, Path]] = []
            for seg_idx in range(num_segs):
                start = seg_idx * seg_duration
                end = min(start + seg_duration, video_duration)
                seg_frames = [(t, p) for t, p in all_frames if start <= t < end]
                if not seg_frames:
                    continue
                step = max(1, len(seg_frames) / target_per_seg)
                for i in range(target_per_seg):
                    idx = min(int(i * step), len(seg_frames) - 1)
                    candidates.append(seg_frames[idx])

            if not candidates:
                candidates = all_frames[:1]

            # ── 第 3 步：帧差去重 ──
            prev_arr = None
            deduped: list[tuple[float, Path]] = []
            for ts, fp in candidates:
                try:
                    with Image.open(fp) as img:
                        arr = np.array(img.convert('L').resize((100, 75)))
                    if prev_arr is not None:
                        diff = np.mean(np.abs(arr - prev_arr)) / 255.0
                        if diff < self.keyframe_threshold:
                            continue
                    deduped.append((ts, fp))
                    prev_arr = arr
                except Exception:
                    continue

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
                shutil.copy(src, dst)
                enriched_blocks[best]["screenshot"] = os.path.relpath(
                    str(dst), str(output_path.parent)
                ).replace("\\", "/")
                assigned += 1

            self.logger.info("  关键帧: 抽 %d 帧 → 候选 %d → 去重 %d → 分配到 %d 个笔记块",
                             len(all_frames), len(candidates), len(deduped), assigned)

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

    def is_already_processed(self, media_path: Path, output_dir: Path) -> bool:
        """
        检查视频是否已处理
        
        Args:
            media_path: 音视频文件路径
            output_dir: 输出目录
            
        Returns:
            如果已处理返回 True，否则返回 False
        """
        md_path = output_dir / f"{media_path.stem}.md"
        json_path = output_dir / f"{media_path.stem}.json"
        srt_path = output_dir / f"{media_path.stem}.srt"
        docx_path = output_dir / f"{media_path.stem}.docx"

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
    
    def process_media(self, media_path: Path, output_dir: Path, skip_processed: bool = True) -> bool:
        """
        处理单个视频文件
        
        Args:
            media_path: 音视频文件路径
            output_dir: 输出目录
            skip_processed: 是否跳过已处理的文件
            
        Returns:
            处理成功返回 True，失败返回 False
        """
        # 安全处理媒体文件名，避免编码错误
        media_name = "未知文件"
        try:
            media_name = os.fsdecode(media_path.name)
        except:
            pass
        
        # 检查是否已处理
        if skip_processed and self.is_already_processed(media_path, output_dir):
            self.logger.info("[跳过] %s (已处理)", media_name)
            return True

        self.logger.info("[处理] %s", media_name)
        try:
            if self.is_audio_file(media_path):
                original_save_keyframes = self.save_keyframes
                self.save_keyframes = False
            else:
                original_save_keyframes = self.save_keyframes

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

            # 安全处理输出文件名，避免编码错误
            output_stem = ""
            try:
                output_stem = os.fsdecode(media_path.stem)
            except:
                import hashlib
                output_stem = hashlib.md5(str(media_path).encode('utf-8', errors='ignore')).hexdigest()
            
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

        for i, media_path in enumerate(media_files, 1):
            if self.stop_event and self.stop_event.is_set():
                self.logger.info("收到停止信号，停止后续文件处理")
                break
            if self.process_media(media_path, output_path, skip_processed):
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
