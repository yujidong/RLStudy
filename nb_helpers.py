"""全部 build_chXX.py 共享的 notebook 构建工具（统一写出格式与 metadata）。

历史背景：ch06-09（dict 风格）、ch10（tuple 风格）、ch11-14（nbformat 风格）、
ch16（dict 风格变体）、ch17/18（content.txt 解析）各自复制了一份
"md/code helper + metadata + 写盘"样板，且 metadata 有三种漂移。
本模块把它们收敛为一处；各章脚本只保留内容定义。

两种主要用法：

**A. dict 追加风格（ch06-09 / ch11-14）**::

    from nb_helpers import NotebookBuilder
    _nb = NotebookBuilder("ch06")
    cells = _nb.cells            # 模块级 cells 列表（build_notebooks.py 适配层读它）
    md, code = _nb.md, _nb.code  # 正文里 md(...)/code(...) 调用不变
    ...
    if __name__ == "__main__":
        _nb.write("ch06_xxx.ipynb")

**B. tuple 风格（build_notebooks.py 内联章 / ch10）**::

    from nb_helpers import md, code, build_notebook, save
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"

# 统一的 notebook metadata（此前 ch06-09 / ch11-14 / ch16 三种漂移，现统一为一种）
NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.13",
        "mimetype": "text/x-python",
        "file_extension": ".py",
    },
}

Cell = Tuple[str, str]


# =============================================================================
# B. tuple 风格（与 build_notebooks.py 原实现逐字节一致，保证重建产物不变）
# =============================================================================
def md(text: str) -> Cell:
    return ("md", text)


def code(text: str) -> Cell:
    return ("code", text)


def build_notebook(cells: List[Cell]) -> dict:
    """把 cells 列表转成 nbformat 4 的 dict。"""
    nb_cells = []
    for ctype, src in cells:
        # 按 \\n 切，但保留每行结尾的 \\n（最后一行不加）
        lines = src.splitlines(keepends=False)
        source = [s + "\n" for s in lines[:-1]] + ([lines[-1]] if lines else [])
        cid = uuid.uuid4().hex[:8]
        if ctype == "md":
            nb_cells.append({
                "id": cid,
                "cell_type": "markdown",
                "metadata": {},
                "source": source or [""],
            })
        else:
            nb_cells.append({
                "id": cid,
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source or [""],
            })
    return {
        "cells": nb_cells,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save(nb: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  -> {path}  ({len(nb['cells'])} cells)")


# =============================================================================
# A. dict 追加风格
# =============================================================================
class NotebookBuilder:
    """章节 cell 收集器 + 统一写出（ch06-09 / ch11-14 用）。

    Parameters
    ----------
    id_prefix : str
        cell id 前缀，如 "ch06"（生成的 id 形如 ch06c001）。
    """

    def __init__(self, id_prefix: str):
        self.id_prefix = id_prefix
        self.cells = []
        self._n = 0

    def _next_id(self) -> str:
        self._n += 1
        return f"{self.id_prefix}c{self._n:03d}"

    def md(self, source: str) -> None:
        self.cells.append({
            "cell_type": "markdown",
            "id": self._next_id(),
            "metadata": {},
            "source": source.splitlines(keepends=True),
        })

    def code(self, source: str) -> None:
        self.cells.append({
            "cell_type": "code",
            "id": self._next_id(),
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        })

    def write(self, nb_name: str) -> Path:
        out = NOTEBOOKS_DIR / nb_name
        notebook = {
            "cells": self.cells,
            "metadata": NOTEBOOK_METADATA,
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        save(notebook, out)
        return out


# =============================================================================
# ch17/18：content.txt 标记解析
# =============================================================================
def parse_content_file(path) -> List[dict]:
    """解析 ``###%% MD`` / ``###%% CODE`` 标记分隔的内容文件为 cell dict 列表。

    每个标记到下一个标记之间的所有行是一个 cell 的内容
    （首尾空行剥掉、内部结构保留）。
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    cells = []
    current_type = None
    current_lines = []
    counter = [0]

    def _flush(ctype, lines):
        src = "".join(lines)
        while src.startswith("\n"):
            src = src[1:]
        while src.endswith("\n\n"):
            src = src[:-1]
        if not src:
            return
        counter[0] += 1
        cell = {
            "cell_type": "markdown" if ctype == "MD" else "code",
            "id": f"cell-{counter[0]:03d}",
            "metadata": {},
            "source": src.splitlines(keepends=True),
        }
        if ctype != "MD":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    for line in text.splitlines(keepends=True):
        if line.startswith("###%% "):
            if current_type is not None:
                _flush(current_type, current_lines)
            marker = line.strip().split(maxsplit=1)[1]
            if marker not in ("MD", "CODE"):
                raise ValueError(f"Unknown marker: {marker}")
            current_type = marker
            current_lines = []
        else:
            current_lines.append(line)
    if current_type is not None:
        _flush(current_type, current_lines)
    return cells


def write_notebook_dict(cells: List[dict], nb_name: str) -> Path:
    """从 cell dict 列表直接写出 notebook（ch16 / ch17 / ch18 用）。"""
    out = NOTEBOOKS_DIR / nb_name
    notebook = {
        "cells": cells,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    save(notebook, out)
    return out
