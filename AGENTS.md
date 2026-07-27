# 视频字幕转录工具 - 项目说明

## 项目概述

这是一个基于 Python 的音视频字幕转录工具，使用 **Faster Whisper small** 模型在本地为音视频生成带时间戳的字幕。提供 Web UI 和命令行两种使用方式。

### 主要特性

- 🚀 **高效转写**：Faster Whisper 架构，GPU 加速，快速准确的语音识别
- 🎯 **多语言支持**：自动识别中文、英文等多种语言
- 📝 **多种输出格式**：Markdown、JSON、SRT、Word (.docx)
- 🖼️ **智能关键帧**：基于帧差检测算法，每10分钟均匀提取5-10张关键帧
- ⏭️ **智能跳过**：自动检测已处理文件，避免重复处理
- 📁 **批量处理**：递归扫描目录，批量处理多个音视频文件
- 🧹 **智能清理**：自动清理连续重复文本，合并短片段为笔记块
- 🌐 **Web UI**：Flask 现代化浏览器界面，无黑框闪烁
- 🔧 **中文优化**：完善的 Windows 中文路径支持

### 技术栈

- **Python** 3.8+
- **Flask** 3.0+ - Web 服务框架
- **faster-whisper** - 高性能语音识别模型
- **onnxruntime-gpu** - CUDA 加速推理引擎
- **PyTorch** 2.0+ - 基础深度学习框架
- **ffmpeg** - 音视频提取和关键帧提取
- **Pillow** 10.0.0+ - 图像处理和帧差计算
- **numpy** - 帧差计算
- **Pandoc** - Markdown 转 Word 文档

## 项目结构

```
vedionotebook/
├── web_app.py                 # Web UI 服务 (Flask)
├── templates/
│   └── index.html             # 前端界面
├── video_transcriber.py       # 核心转写模块 (Faster Whisper 引擎)
├── document_exporter.py       # Word 文档导出模块
├── main.py                    # 命令行主程序入口
├── requirements.txt           # Python 依赖
├── README.md                  # 快速开始文档
├── AGENTS.md                  # 项目说明文档
└── .gitignore                 # Git 忽略文件配置
```

## 安装依赖

```bash
pip install -r requirements.txt
pip install onnxruntime-gpu  # GPU 加速（可选）
```

还需安装：
- **Pandoc**：https://pandoc.org/installing.html（Word 导出用）
- **ffmpeg**：https://ffmpeg.org/（音视频处理用）

## 使用方法

### Web UI 模式（推荐）

```bash
python web_app.py
```

浏览器访问 **http://127.0.0.1:5000**

### 命令行模式 (CLI)

```bash
python main.py -i "输入目录" -o "输出目录" --device cuda
```

完整参数：

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必需 | 输入路径（文件或目录） |
| `--output` | `-o` | 必需 | 输出目录 |
| `--model` | | `faster-whisper-small` | 模型名称 |
| `--device` | | `cuda` | 计算设备 (cuda/cpu) |
| `--language` | `-l` | `auto` | 语言设置（zh/en/auto） |
| `--ffmpeg` | | `ffmpeg` | ffmpeg 程序路径 |
| `--pandoc` | | `pandoc` | Pandoc 程序路径 |
| `--keyframe-frequency` | | `normal` | 截图频率 (few/normal/medium/many/all) |
| `--no-skip` | | `false` | 不跳过已处理文件 |
| `--no-keyframes` | | `false` | 不提取关键帧截图 |
| `--no-export-word` | | `false` | 不导出 Word 文档 |

## 关键帧提取算法

1. **10分钟分段**：按600秒为一个时间段划分视频
2. **均匀采样**：每段提取20张候选帧
3. **帧差去重**：将帧转为灰度图(100x75)，计算帧差，阈值0.15以上保留
4. **数量控制**：每段确保5-10张，总数不足5张时补充均匀采样
5. **全局去重**：最后对所有关键帧做一次帧差检测，去掉相似图片

## 输出格式

### Markdown (.md)
包含完整的时间戳、文本内容和关键帧图片

### JSON (.json)
结构化的转写数据，便于程序处理

### SRT (.srt)
标准的字幕格式，可用于视频播放器

### Word (.docx)
使用 Pandoc 生成的 Word 文档，图片正确嵌入

## 故障排除

### CUDA 不可用
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
确保安装了正确的 `onnxruntime-gpu` 和 NVIDIA 驱动。

### Word 文档没有图片
Pandoc 需要相对路径的图片，确保 Markdown 文件和 assets 目录在同一目录，使用 `--pandoc` 参数指定正确的 Pandoc 路径。

### 权限错误
处理百度网盘同步目录时可能遇到，暂时关闭网盘同步或将输出目录设在非同步文件夹。

### 中文路径乱码
代码已优化中文路径处理，确保使用 UTF-8 编码。

## Git 配置

本项目包含 `.gitignore` 文件，用于排除不需要版本控制的文件和目录：

- **虚拟环境**：`venv/`, `.venv*/`
- **临时文件**：`*.tmp`, `.tmp/`, `*.temp`
- **音视频文件**：`*.mp4`, `*.mp3` 等
- **输出目录**：`subtitles/`, `local_output/`, `CLI_Output/`
- **构建输出**：`dist/`, `build/`, `*.spec`
- **IDE文件**：`.vscode/`, `.idea/`
- **系统文件**：`Thumbs.db`, `.DS_Store`

## 打包说明

项目使用 Flask Web UI 作为主要交互方式，无需打包成 EXE。直接通过 `python web_app.py` 启动，浏览器访问即可使用。

如需打包，可使用 PyInstaller，但需注意包含 torch/onnxruntime 等大型依赖，打包体积较大（约 100MB+）。