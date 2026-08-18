"""Build notebooks/ch17_self_play_cai.ipynb from ch17_content.txt.

The content file uses simple cell markers:
    ###%% MD    -> markdown cell
    ###%% CODE  -> code cell

Cell content is everything between markers. 解析器与写出在 nb_helpers.py
（与 build_ch18.py 共享同一实现）。
"""
from pathlib import Path

from nb_helpers import parse_content_file, write_notebook_dict


def main():
    content = Path(__file__).parent / "ch17_content.txt"
    cells = parse_content_file(content)
    write_notebook_dict(cells, "ch17_self_play_cai.ipynb")


if __name__ == "__main__":
    main()
