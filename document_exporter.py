from __future__ import annotations

import re
import subprocess
from pathlib import Path


class DocumentExporter:
    """将 Markdown 笔记导出为 Word 文档。"""

    def __init__(self, pandoc_path: str | None = None):
        """
        初始化文档导出器
        
        Args:
            pandoc_path: pandoc 可执行文件路径
        """
        self.pandoc_path = pandoc_path or "pandoc"

    def extract_image_path(self, line: str) -> str | None:
        """从 Markdown 图片语法中提取路径，兼容标题里包含 [] 的情况。"""
        if not line.startswith("!") or not line.endswith(")"):
            return None
        marker = "] ("
        if marker in line:
            split_index = line.rfind(marker)
            if split_index == -1:
                return None
            return line[split_index + len(marker) : -1].strip()
        marker = "]"
        split_index = line.rfind(marker)
        if split_index == -1:
            return None
        return line[split_index + 1 : -1].strip()

    def export_docx(self, markdown_path: Path, output_path: Path) -> Path:
        """使用 pandoc 将 Markdown 导出为 Word 文档。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.pandoc_path,
            str(markdown_path),
            "-o",
            str(output_path),
            "--from", "markdown",
            "--to", "docx",
            "--standalone"
        ]
        
        try:
            # 在Markdown文件所在目录运行Pandoc，确保相对路径的图片能被正确找到
            cwd = markdown_path.parent
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Pandoc 导出失败: {e.stderr}")
        except FileNotFoundError:
            raise Exception(f"Pandoc 未找到: {self.pandoc_path}")