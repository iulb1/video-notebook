# 视频字幕转录工具

基于 **Faster Whisper small** 的音视频字幕转录工具，为音视频生成带时间戳的字幕，支持多格式输出和智能关键帧提取。

## 主要特性

- 🚀 **高效转写**：Faster Whisper 架构，GPU 加速，支持中文/英文等多语言自动检测
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
├── video_transcriber.py       # 核心转写模块 (Faster Whisper 引擎)
├── document_exporter.py       # Word 文档导出模块
├── main.py                    # 命令行主程序入口
├── requirements.txt           # Python 依赖
├── AGENTS.md                  # 项目说明文档
└── .gitignore                 # Git 忽略文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install onnxruntime-gpu  # GPU 加速
```

> 还需安装 [Pandoc](https://pandoc.org/installing.html)（Word 导出用）和 [ffmpeg](https://ffmpeg.org/)（音视频处理用）。

### 2. 启动 Web UI

```bash
python web_app.py
```

浏览器访问 **http://127.0.0.1:5000**

### 3. 命令行模式

```bash
python main.py -i "输入目录" -o "输出目录" --device cuda
```

## Web UI 界面

| 功能 | 说明 |
|------|------|
| 📂 文件路径 | 输入/输出路径设置，支持路径存在性校验 |
| ⚙️ 工具配置 | FFmpeg / Pandoc 路径预设 |
| 🔧 转写参数 | 设备、语言、截图频率、输出格式勾选 |
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
| `--device` | `cuda` | cuda / cpu |
| `--language` | `auto` | zh / en / auto |
| `--keyframe-frequency` | `normal` | few/normal/medium/many/all |
| `--no-skip` | `false` | 不跳过已处理文件 |
| `--no-keyframes` | `false` | 不提取关键帧 |
| `--no-export-word` | `false` | 不导出 Word |

## 故障排除

- **CUDA 不可用**：`python -c "import torch; print(torch.cuda.is_available())"`
- **Word 无图片**：确保 Pandoc 路径正确，Markdown 和 assets 在同一目录
- **权限错误**：关闭百度网盘同步，或将输出目录设在非同步文件夹