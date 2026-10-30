import sys
from pathlib import Path

# 让 pytest 在未安装包的情况下也能导入 src 布局
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
