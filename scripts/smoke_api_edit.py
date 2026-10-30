"""真实 API 冒烟验证：一条命令判断 key 可用性、出图质量、掩码约束是否守住。

它做三件事：
    1. 用现有 UV 纹理 + 掩码真调一次托管补全（消耗一次接口额度）；
    2. 校验产出尺寸、非空、是否真的填上了待补区；
    3. 计算「掩码黑色区域」的像素变化——这是最关键的约束：
       已观测区域被改动说明模型没守住指令，应改用支持真 mask 的后端。

用法：
    python scripts/smoke_api_edit.py                      # 用默认的 proxy 产物
    python scripts/smoke_api_edit.py --uv a.png --mask b.png --prompt "..."

退出码：0 = 通过；1 = 接口/约束不达标；2 = 缺 key 或输入缺失。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avatar_tryon.config import load_dotenv  # noqa: E402
from avatar_tryon.models.errors import ModelNotWired  # noqa: E402
from avatar_tryon.models.generation_adapters import (  # noqa: E402
    DashscopeEdit,
    DashscopeWanxEdit,
)

DEFAULT_UV = ROOT / "outputs" / "demo" / "final_api" / "uv" / "uv_multiview.png"
DEFAULT_MASK = ROOT / "outputs" / "demo" / "final_api" / "uv" / "uv_mask.png"
DEFAULT_PROMPT = "深蓝色棉质 T 恤面料的连续纹理，中性棚拍光照，无 logo 无文字"

# 已知区域允许的平均像素变化（0-255）；超过这个值说明模型改了不该改的地方
KNOWN_TOLERANCE = 8.0


def main() -> int:
    parser = argparse.ArgumentParser(description="托管纹理补全的冒烟验证")
    parser.add_argument("--uv", type=Path, default=DEFAULT_UV)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--backend",
        choices=("wanx", "conversational"),
        default="wanx",
        help="wanx=真 mask 异步接口（默认）；conversational=对话式端点（实测守不住掩码）",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("缺少 DASHSCOPE_API_KEY。", file=sys.stderr)
        print("到 https://bailian.console.aliyun.com 开通百炼并创建 API-KEY，", file=sys.stderr)
        print("写入 .env：DASHSCOPE_API_KEY=sk-xxxx", file=sys.stderr)
        return 2

    for p in (args.uv, args.mask):
        if not p.exists():
            print(f"输入不存在：{p}\n先跑一次 --engine proxy 生成 UV 产物。", file=sys.stderr)
            return 2

    print(f"[1/3] 调用托管补全（{args.backend}）\n      uv   = {args.uv}\n      mask = {args.mask}")
    editor: DashscopeEdit | DashscopeWanxEdit
    if args.backend == "wanx":
        editor = DashscopeWanxEdit(ROOT / "weights")
    else:
        editor = DashscopeEdit(ROOT / "weights")
    try:
        out = editor.inpaint(args.uv, args.mask, args.prompt)
    except ModelNotWired as exc:
        print(f"调用失败：{exc}", file=sys.stderr)
        return 1
    print(f"      产出 = {out}")

    print("[2/3] 校验产出")
    try:
        from PIL import Image
    except ImportError:
        print("需要 Pillow 才能校验：pip install pillow", file=sys.stderr)
        return 1

    src = Image.open(args.uv).convert("RGB")
    res = Image.open(out).convert("RGB")
    if res.size != src.size:
        res = res.resize(src.size)
        print(f"      注意：返回尺寸与输入不同，已缩放对齐（原始 {Image.open(out).size}）")
    mask = Image.open(args.mask).convert("L").resize(src.size, Image.NEAREST)

    known_mae, fill_ratio, fill_var = _metrics(src, res, mask)
    print(f"      已观测区平均变化 = {known_mae:.2f}/255（阈值 {KNOWN_TOLERANCE}）")
    print(f"      待补区均值 = {fill_ratio:.1f}，方差 = {fill_var:.1f}（方差过低说明糊成一片）")

    print("[3/3] 结论")
    ok = True
    if known_mae > KNOWN_TOLERANCE:
        ok = False
        print("      ✗ 已观测区域被明显改动：模型把 UV 图集当成一张图片重新合成了。")
        print("        这是实测已知的结论——通用图像编辑模型的先验与 UV 布局冲突，")
        print("        不应在 UV 空间调用它们。正确路径是先生成侧/背视图，")
        print("        再由 three_view._backproject_views 做几何投影烘焙（见 docs/pipeline.md）。")
    if fill_var < 5.0:
        ok = False
        print("      ✗ 待补区过于平滑，生成质量不足（可能只是被模糊填充）。")
    if ok:
        print("      ✓ 通过：接口可用、尺寸正确、掩码约束守住、补全区有真实纹理。")
    return 0 if ok else 1


def _metrics(src, res, mask) -> tuple[float, float, float]:
    """返回 (已观测区 MAE, 待补区均值, 待补区方差)。"""
    try:
        import numpy as np

        a = np.asarray(src, dtype=np.float32)
        b = np.asarray(res, dtype=np.float32)
        m = np.asarray(mask, dtype=np.uint8) >= 128  # True = 待补
        diff = np.abs(a - b).mean(axis=2)
        known_mae = float(diff[~m].mean()) if (~m).any() else 0.0
        if not m.any():
            return known_mae, 0.0, 0.0
        fill = b[m].mean(axis=1)
        return known_mae, float(fill.mean()), float(fill.var())
    except ImportError:
        from PIL import Image, ImageChops, ImageStat

        diff = ImageChops.difference(src, res).convert("L")
        inv = mask.point(lambda v: 255 if v < 128 else 0)
        covered = ImageStat.Stat(inv).mean[0] / 255.0
        total = ImageStat.Stat(Image.composite(diff, Image.new("L", src.size, 0), inv)).mean[0]
        known_mae = total / covered if covered > 0 else 0.0
        fill_img = Image.composite(res, Image.new("RGB", src.size, (0, 0, 0)), mask)
        stat = ImageStat.Stat(fill_img)
        return known_mae, sum(stat.mean) / 3.0, sum(stat.stddev) / 3.0


if __name__ == "__main__":
    raise SystemExit(main())
