# 视频字幕转录工具

支持 **Faster Whisper** 与 **SenseVoice** 的本地音视频字幕转录工具，为音视频生成带时间戳的字幕，支持多格式输出和智能关键帧提取。模型层与视频处理层相互独立，后续可以继续添加其他本地 ASR 后端。

## 主要特性

- 🚀 **模型可切换**：默认使用中文优化的 SenseVoiceSmall，也可切换到 Faster Whisper Small
- 📝 **多格式输出**：Markdown、JSON、SRT、Word (.docx)
- 🖼️ **智能关键帧**：基于帧差检测，每10分钟均匀提取5-10张关键帧
- ⏭️ **智能跳过**：自动检测已处理文件，避免重复处理
- 📁 **批量处理**：递归扫描目录，批量处理多个音视频文件
- 🧹 **智能清理**：自动清理连续重复文本，合并短片段为笔记块
- 🌐 **Web UI**：现代化浏览器界面，无黑框闪烁

## 项目结构

```
vedionotebook/
├── web_app.py                 # Web UI 服务（Flask）
├── templates/index.html       # 前端界面
├── asr_backends/              # 可插拔 ASR 后端与模型工厂
├── video_transcriber.py       # 音视频处理与输出编排
├── document_exporter.py       # Word 文档导出模块
├── main.py                    # 命令行主程序入口
├── requirements.txt           # Python 依赖
├── AGENTS.md                  # 项目说明文档
└── .gitignore                 # Git 忽略文件
```

## 快速开始

### 1. 安装环境（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
```

脚本会创建独立的 `video-notebook` Conda 环境，固定 Python 3.12，并安装 CUDA 12.8 版 PyTorch、SenseVoice 与 Faster Whisper 依赖。不会再使用系统 `PATH` 中不确定的 Python。

如只使用 CPU 或需要手动维护环境，也可以分别安装基础依赖与 SenseVoice 依赖：

```bash
pip install -r requirements-sensevoice.txt
```

首次使用时会从 ModelScope 下载 `iic/SenseVoiceSmall` 和长音频 VAD 模型。桌面版 RTX 4060 8GB 推荐使用 CUDA；SenseVoiceSmall 对中文、粤语课程和会议录音通常更合适。

> 还需安装 [Pandoc](https://pandoc.org/installing.html)（Word 导出用）和 [ffmpeg](https://ffmpeg.org/)（音视频处理用）。

### 2. 启动 Web UI

双击 `视频字幕转录工具.bat`。启动器会先运行 `environment_check.py`，确认 Python 包、CUDA 和 FFmpeg 可用，再启动 Web 服务。

浏览器访问 **http://127.0.0.1:5000**

### 3. 命令行模式

```bash
python main.py -i "输入目录" -o "输出目录" --device cuda
```

切换到 SenseVoice：

```bash
python main.py -i "输入目录" -o "输出目录" --backend sensevoice --device cuda
```

也可以通过 `--model` 指定模型名称或本地模型目录：

```bash
python main.py -i "输入目录" -o "输出目录" --backend sensevoice --model "D:\\models\\SenseVoiceSmall"
```

## Web UI 界面

| 功能 | 说明 |
|------|------|
| 📂 文件路径 | 输入/输出路径设置，支持路径存在性校验 |
| ⚙️ 工具配置 | FFmpeg / Pandoc 路径预设 |
| 🔧 转写参数 | 模型、设备、语言、截图频率、输出格式勾选 |
| 🚀 一键启停 | 启动/停止任务 |
| 📋 实时日志 | 带颜色标识的运行日志 |

## 关键帧算法

1. 按 **10 分钟分段** 均匀提取候选帧
2. **帧差检测**（灰度图差异）去掉相似帧（阈值 0.15）
3. 确保每段 **5-10 张**，不足时补充均匀采样帧
4. 关键帧按时间均匀分配给笔记块

## 输出格式

- **Markdown (.md)** - 完整时间戳、文本、关键帧图片
- **JSON (.json)** - 结构化数据
- **SRT (.srt)** - 标准字幕格式
- **Word (.docx)** - Pandoc 导出，图片正确嵌入

## 常用参数（CLI）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend` | `sensevoice` | sensevoice / faster-whisper |
| `--model` | 后端默认值 | 模型名称或本地模型目录 |
| `--device` | `cuda` | cuda / cpu |
| `--language` | `auto` | zh / en / auto |
| `--keyframe-frequency` | `normal` | few/normal/medium/many/all |
| `--no-skip` | `false` | 不跳过已处理文件 |
| `--no-keyframes` | `false` | 不提取关键帧 |
| `--no-export-word` | `false` | 不导出 Word |

## 故障排除

- **CUDA 不可用**：`python -c "import torch; print(torch.cuda.is_available())"`
- **SenseVoice 无法加载**：先执行 `pip install -r requirements-sensevoice.txt`
- **首次启动较慢**：新模型需要先下载到本地缓存，后续运行会直接复用
- **Word 无图片**：确保 Pandoc 路径正确，Markdown 和 assets 在同一目录
- **权限错误**：关闭百度网盘同步，或将输出目录设在非同步文件夹

## 添加新的本地模型

在 `asr_backends` 中实现 `ASRBackend.transcribe()`，将模型结果转换成统一的 `start`、`end`、`text` 分段，然后在工厂中注册后端名称和默认模型即可。视频扫描、FFmpeg、关键帧和各输出格式不需要随模型修改。
