"""配置：路径与少量环境变量。

刻意不引入 pydantic / python-dotenv：这个仓库的依赖应当尽可能少，
让 clone 之后的第一条命令就能跑起来。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.environ.get("AVATAR_WEIGHTS_DIR", ROOT / "weights"))
ASSETS_DIR = ROOT / "assets"
GARMENT_CATALOG = ASSETS_DIR / "garments.json"
DEFAULT_OUTPUT = ROOT / "outputs"


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 读取：KEY=VALUE，忽略注释与空行。"""
    env_file = path or (ROOT / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
