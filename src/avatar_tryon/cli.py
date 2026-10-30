"""命令行入口。

用法：
    python -m avatar_tryon.cli --image photo.jpg --height 170 --mock
    python -m avatar_tryon.cli --image photo.jpg --height 170 --garment tee_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .pipeline import orchestrator
from .utils.image import write_blank_png


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="avatar-tryon", description="单图驱动的 3D 数字人试穿试发")
    p.add_argument("--image", type=Path, required=True, help="单张全身照")
    p.add_argument("--height", type=float, required=True, help="身高，厘米")
    p.add_argument("--out", type=Path, default=config.DEFAULT_OUTPUT, help="输出目录")
    p.add_argument("--weights-dir", type=Path, default=config.WEIGHTS_DIR, help="权重目录")
    p.add_argument("--mock", action="store_true", help="不加载模型，用占位实现跑通流水线")
    p.add_argument(
        "--engine",
        choices=("auto", "mock", "proxy", "real"),
        default="auto",
        help=(
            "执行档位：auto=按 --mock 推断；mock=占位；"
            "proxy=参数化几何 + 软件渲染 + 基线补全（纯 CPU，无需权重）；real=加载真实模型"
        ),
    )
    p.add_argument("--garment", type=str, default=None, help="要试穿的单品 ID")
    p.add_argument("--garment-image", type=Path, default=None, help="单品图（平铺或模特图）")
    p.add_argument(
        "--vton-backend",
        choices=("api", "local"),
        default="api",
        help="试衣后端：api 用托管接口快速验证，local 用本地权重",
    )
    p.add_argument("--category", choices=("tops", "bottoms", "one-pieces"), default="tops")
    p.add_argument(
        "--views-backend",
        choices=("api", "local"),
        default="local",
        help="??????api ????????/????? DASHSCOPE_API_KEY??local ? LHM",
    )
    p.add_argument(
        "--edit-backend",
        choices=("api", "local"),
        default="api",
        help="UV 纹理补全后端：api 用 DashScope 托管接口（需 DASHSCOPE_API_KEY），local 用本地权重",
    )
    p.add_argument("--catalog", type=Path, default=config.GARMENT_CATALOG)
    p.add_argument("--hairstyle", type=Path, default=None, help="参考发型图")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.load_dotenv()
    engine = None if args.engine == "auto" else args.engine

    if not args.image.exists():
        if not (args.mock or args.engine == "mock"):
            print(f"图片不存在：{args.image}", file=sys.stderr)
            return 2
        write_blank_png(args.image, 1024, 1024)
        print(f"[mock] 未提供图片，已生成占位图：{args.image}")

    result = orchestrator.run(
        image=args.image,
        height_cm=args.height,
        out_dir=args.out,
        weights_dir=args.weights_dir,
        mock=args.mock,
        garment_catalog=args.catalog if args.catalog.exists() else None,
        garment_id=args.garment,
        garment_image=args.garment_image,
        vton_backend=args.vton_backend,
        vton_category=args.category,
        hairstyle=args.hairstyle,
        engine=engine,
        edit_backend=args.edit_backend,
        views_backend=args.views_backend,
    )

    print(json.dumps({
        "ok": result.ok,
        "stages": result.stages,
        "artifacts": result.artifacts,
        "warnings": result.warnings,
        "errors": result.errors,
        "result_json": str(result.out_dir / "result.json"),
    }, ensure_ascii=False, indent=2))

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
