"""试穿冒烟验证：一条命令判断「本机能不能真跑出换装图」。

它做四件事：
    1. 检查前置（权重 / 密钥），缺什么直接说清楚；
    2. 真跑一次试穿（本地权重或托管接口）；
    3. 校验产出：尺寸、非空、与原图的差异——全等说明没换装，是假成功；
    4. 报告耗时，便于判断 CPU 是否够用。

用法：
    python scripts/smoke_vton.py                          # 本地权重
    python scripts/smoke_vton.py --backend api            # 托管接口（需 FASHN_API_KEY）
    python scripts/smoke_vton.py --person p.jpg --garment g.jpg --category tops

退出码：0 = 通过；1 = 产出不合格；2 = 前置缺失（缺权重 / 缺 key）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avatar_tryon.config import load_dotenv  # noqa: E402
from avatar_tryon.models.errors import ModelNotWired  # noqa: E402
from avatar_tryon.models.generation_adapters import FashnVton  # noqa: E402

DEFAULT_PERSON = ROOT / "outputs" / "demo" / "person_front.png"
DEFAULT_GARMENT = ROOT / "outputs" / "demo" / "garment" / "tee.jpg"
WEIGHTS = ROOT / "weights"

# 与原图的平均像素差低于这个值，说明衣服根本没换上去
MIN_CHANGE = 2.0


def _rel(path: Path) -> str:
    """尽量显示仓库内相对路径；外部路径或相对路径则原样返回。"""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def image_diff(a: Path, b: Path) -> float:
    """平均绝对像素差（0-255）。只用于判断「有没有变化」，不作为质量指标。"""
    from PIL import Image, ImageChops

    with Image.open(a) as ia, Image.open(b) as ib:
        left = ia.convert("RGB")
        right = ib.convert("RGB").resize(left.size)
        diff = ImageChops.difference(left, right)
        stat = diff.convert("L").resize((64, 64)).getdata()
    return sum(stat) / len(stat)


def main() -> int:
    parser = argparse.ArgumentParser(description="试穿冒烟验证")
    parser.add_argument("--person", type=Path, default=DEFAULT_PERSON)
    parser.add_argument("--garment", type=Path, default=DEFAULT_GARMENT)
    parser.add_argument("--category", default="tops", choices=FashnVton.CATEGORIES)
    parser.add_argument("--backend", default="local", choices=("local", "api"))
    args = parser.parse_args()

    load_dotenv()
    for label, path in (("人物图", args.person), ("服装图", args.garment)):
        if not path.exists():
            print(f"缺少{label}：{path}", file=sys.stderr)
            return 2

    model = FashnVton(WEIGHTS, backend=args.backend)
    print(f"后端 {args.backend}  category {args.category}")
    print(f"  人物 {_rel(args.person)}")
    print(f"  服装 {_rel(args.garment)}")

    t0 = time.monotonic()
    try:
        result = model.apply(
            person=args.person,
            garment=args.garment,
            garment_id=args.garment.stem,
            category=args.category,
        )
    except ModelNotWired as exc:
        print(f"\n前置缺失：\n{exc}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - t0

    out = Path(result.image_path)
    if not out.exists() or out.stat().st_size == 0:
        print(f"产出为空：{out}", file=sys.stderr)
        return 1

    from PIL import Image

    with Image.open(out) as img:
        size = img.size
    change = image_diff(args.person, out)

    print(f"\n耗时 {elapsed:.1f}s  产出 {_rel(out)}  {size[0]}x{size[1]}")
    print(f"与原图平均差 {change:.1f}/255（阈值 {MIN_CHANGE}）")
    if change < MIN_CHANGE:
        print("差异过小：衣服没换上去，属于假成功。", file=sys.stderr)
        return 1
    print("通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
