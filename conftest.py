# 这个文件的存在让 pytest 把它所在目录（项目根）加到 sys.path[0]，
# 让 `pytest tests/` 直接调用时也能找到项目内的 rlenvs/utils 包。
# 历史教训：原先包名叫 envs，会和 conda 的 miniconda3/envs/ 目录
# （namespace package）冲突，所以已重命名为 rlenvs。
