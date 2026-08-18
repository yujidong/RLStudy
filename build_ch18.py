"""Build notebooks/ch18_offline_rl.ipynb from ch18_content.txt.

Same parser as build_ch17.py: cell markers
    ###%% MD    -> markdown cell
    ###%% CODE  -> code cell

解析器与写出在 nb_helpers.py（与 build_ch17.py 共享同一实现）。
"""
from pathlib import Path

from nb_helpers import parse_content_file, write_notebook_dict


def main():
    content = Path(__file__).parent / "ch18_content.txt"
    cells = parse_content_file(content)
    write_notebook_dict(cells, "ch18_offline_rl.ipynb")


if __name__ == "__main__":
    main()
