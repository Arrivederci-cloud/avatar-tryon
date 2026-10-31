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
from .models import registry
from .pipeline import compare, orchestrator
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
        help="三视图后端：api 用托管接口（需 DASHSCOPE_API_KEY），local 用 LHM",
    )
    p.add_argument(
        "--edit-backend",
        choices=("api", "local"),
        default="api",
        help="UV 纹理补全后端：api 用 DashScope 托管接口（需 DASHSCOPE_API_KEY），local 用本地权重",
    )
    p.add_argument("--catalog", type=Path, default=config.GARMENT_CATALOG)
    p.add_argument("--hairstyle", type=Path, default=None, help="参考发型图")
    p.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "候选对比：传入多个文件或目录，逐个渲染后拼成一张对比图。"
            "类型由文件名前缀决定——hair_*/hairstyle_* 作发型，"
            "worn_*/ref_*/look_* 作模特上身参照图，其余当衣服"
        ),
    )
    p.add_argument(
        "--compare-backend",
        choices=("local", "api"),
        default="local",
        help="对比时试穿用的后端：local 用本地权重（GPU 上约 3 分钟/件），api 用托管接口",
    )
    p.add_argument("--compare-cols", type=int, default=4, help="对比图每行几个候选")
    p.add_argument(
        "--compare-category",
        choices=(*compare.CATEGORIES, compare.CATEGORY_AUTO),
        default=compare.CATEGORY_AUTO,
        help="对比时每件衣服的试穿类别；auto 按文件名关键词逐件判断（默认）",
    )
    p.add_argument("--hair-color", type=str, default=None, help="发型迁移的目标发色（可选）")
    return p


def _run_compare(args, engine: str | None, out_dir: Path) -> compare.CompareReport:
    """按档位挑后端，然后把候选逐个渲染并拼对比图。

    后端只在「构造」阶段出错时才降级为 None——那样每个候选会各自记一条
    not_wired，用户仍能看到其余候选的结果，而不是整轮失败。
    """
    engine_name = engine or ("mock" if args.mock else "real")
    vton = hair = None
    try:
        vton = registry.build_vton(
            "fashn_vton", args.weights_dir, engine_name, backend=args.compare_backend
        )
    except (KeyError, ValueError):
        vton = None
    try:
        hair = registry.build_hair("hairfastgan", args.weights_dir, engine_name)
    except (KeyError, ValueError):
        hair = None

    candidates = compare.discover(args.compare)
    return compare.run(
        args.image,
        candidates,
        out_dir / "compare",
        vton=vton,
        hair=hair,
        category=args.compare_category,
        hair_color=args.hair_color,
        cols=args.compare_cols,
    )


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

    report = _run_compare(args, engine, result.out_dir) if args.compare else None

    print(json.dumps({
        "ok": result.ok,
        "stages": result.stages,
        "artifacts": result.artifacts,
        "warnings": result.warnings,
        "errors": result.errors,
        "result_json": str(result.out_dir / "result.json"),
        "compare": report.as_dict() if report else None,
    }, ensure_ascii=False, indent=2))

    if report is not None:
        print("\n候选对比（排序交给你，工具只把候选摆齐）：")
        print(report.table())
        if report.sheet:
            print(f"对比图：{report.sheet}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
