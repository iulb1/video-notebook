# 视频字幕转录工具需求

## 1. 目标

将音视频文件转录为带时间戳的字幕笔记，并生成适合整理和阅读的 Markdown / JSON / SRT / Word 文档。

## 2. 核心原则

- 稳定优先
- 可用优先
- 功能完整优先于花哨实现

## 3. 必须满足的需求

### 3.1 稳定性

- 能处理常见可播放的视频和音频文件
- 处理失败时给出明确错误信息，不能无提示崩溃
- 支持中文路径、中文文件名和常见特殊字符路径
- 长音频处理时不能因为分块逻辑产生空块或明显的边界错误
- CLI、GUI 的处理结果和参数语义应保持一致

### 3.2 转录

- 使用 SenseVoiceSmall 作为默认模型，Faster Whisper small 作为兼容选项
- 支持在 Faster Whisper 与 SenseVoiceSmall 本地后端之间切换
- 模型实现通过统一 ASR 接口与视频处理、字幕导出解耦
- 支持自动语言检测
- 允许手动指定语言，如 `zh`、`en`
- 输出包含时间戳
- 对连续重复文本做去重清理
- 对短片段做适度合并，生成更适合阅读的笔记块

### 3.3 输出

- 默认输出 Markdown、JSON、SRT
- 默认同时导出 Word 文档
- Word 导出基于 Markdown 结果生成
- 已处理文件可跳过，避免重复处理

### 3.4 关键帧截图

- 仅对视频文件提取截图
- 截图应能反映内容变化，避免大量重复图片
- 提供截图频率控制选项：`few`、`normal`、`medium`、`many`、`all`
- Markdown 中插入截图引用，Word 导出时带入图片

### 3.5 GUI

- 提供桌面 GUI
- 支持选择单个文件或目录
- 支持选择输出目录
- 支持设置设备、语言、截图开关、截图频率、输出格式
- 支持指定 `ffmpeg` 路径
- 支持停止任务
- 在界面中显示日志

## 4. 技术约束

- Python 3.10+
- `faster-whisper`
- `torch`
- `numpy`
- `Pillow`
- `python-docx`
- 系统中可用的 `ffmpeg`

## 5. 命令行参数

- `-i, --input`：输入文件或目录
- `-o, --output`：输出目录
- `--device`：`cuda` 或 `cpu`
- `--backend`：`faster-whisper` 或 `sensevoice`
- `--model`：可选模型名称或本地模型目录
- `--language`：手动指定语言，不填时自动检测
- `--ffmpeg`：可选，指定 ffmpeg 可执行文件路径
- `--no-skip`：不跳过已处理文件
- `--no-keyframes`：不提取关键帧截图
- `--no-export-word`：不导出 Word
- `--keyframe-frequency`：`few` / `normal` / `medium` / `many` / `all`

## 6. 验收标准

- 给定一批音视频文件，程序能够批量处理并输出对应文件
- 自动语言检测对中英文内容可正常工作
- 长音频在 30 秒分块策略下不会因为空块报错
- `ffmpeg` 未安装或路径错误时，报错明确
- GUI 能发起任务、显示日志并请求停止
- CLI 在处理失败时返回非 0 退出码

## 7. 已知环境风险

- 百度网盘、同步目录或只读目录可能导致 `WinError 5` / `Errno 13`
- 当前本地 Python 环境若缺少 `numpy`、`python-docx` 等依赖，将无法运行
- 若系统 PATH 中没有 `ffmpeg`，需要显式设置 `--ffmpeg` 或环境变量 `VIDEO_NOTEBOOK_FFMPEG`
y
y
