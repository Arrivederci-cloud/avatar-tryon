"""联网取样验证：抓一张真实图片，验证解析与流水线。

这不是单元测试（CI 不联网、也不该依赖外网），而是给开发者的一次性自检：

    python scripts/verify_network.py
    python scripts/verify_network.py --height 172 --garment tee_001

它会做三件事：
1. 从候选源抓一张真实 JPEG（有多个源，按顺序尝试）；
2. 用本仓库的纯标准库解析器读尺寸，并与 Pillow 的结果比对；
3. 用这张真图跑一遍 mock 流水线，确认端到端仍然通畅。

抓下来的图片只落在 outputs/verify/（已 gitignore），不会进仓库，也不进 assets/。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avatar_tryon.pipeline import orchestrator  # noqa: E402
from avatar_tryon.utils.image import image_size, write_blank_png  # noqa: E402

# 都是公开可访问的静态资源；picsum / pexels / wikimedia 在部分网络环境下会被拦，
# 所以列表按可用性排序，失败即跳过。
SOURCES = [
    "https://www.gstatic.com/webp/gallery/1.jpg",
    "https://www.gstatic.com/webp/gallery/2.jpg",
    "https://picsum.photos/1024/1536",
]


def fetch(timeout: int = 20) -> tuple[Path, str]:
    out_dir = ROOT / "outputs" / "verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "sample.jpg"

    errors: list[str] = []
    for url in SOURCES:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
                data = resp.read()
            if len(data) < 1024:
                errors.append(f"{url}: 返回内容过小（{len(data)} B）")
                continue
            dest.write_bytes(data)
            return dest, url
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {type(exc).__name__}")

    raise SystemExit("所有候选源都不可达：\n  " + "\n  ".join(errors))


def check_parser(path: Path) -> tuple[tuple[int, int], tuple[int, int] | None]:
    ours = image_size(path)
    try:
        from PIL import Image
    except ImportError:
        return ours, None
    with Image.open(path) as img:
        return ours, img.size


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="联网取样并验证")
    p.add_argument("--height", type=float, default=172.0)
    p.add_argument("--garment", default="tee_001")
    args = p.parse_args(argv)

    path, url = fetch()
    print(f"[1/4] 已抓取：{url}\n      -> {path.relative_to(ROOT)}（{path.stat().st_size} B）")

    ours, ref = check_parser(path)
    print(f"[2/4] 解析尺寸：本仓库={ours}  Pillow={ref}")
    if ref is not None and ours != ref:
        print("      结果不一致，JPEG/PNG 头解析有 bug。")
        return 1
    if ref is None:
        print("      未安装 Pillow，跳过比对（这是允许的，解析器不依赖它）")

    def _run(img: Path):
        return orchestrator.run(
            image=img,
            height_cm=args.height,
            out_dir=ROOT / "outputs" / "verify",
            weights_dir=ROOT / "weights",
            mock=True,
            garment_catalog=ROOT / "assets" / "garments.json",
            garment_id=args.garment,
        )

    res = _run(path)

    # 抓来的图未必满足拍摄规范，被质量控制拦下是预期行为，不算失败。
    # 这时换一张合成图继续验证「流水线是否连通」，两件事分开报告。
    if not res.ok and any("分辨率" in e for e in res.errors):
        print("[3/4] 真实图被质量控制正确拦截（分辨率不足），这本身说明闸门生效")
        synth = write_blank_png(ROOT / "outputs" / "verify" / "synthetic.png", 1024, 1536)
        print("[4/4] 改用合成图验证流水线连通性……")
        res = _run(synth)
    else:
        print("[3/4] 流水线在真实图片上完成")
        print("[4/4] 跳过合成图回退（真实图已通过质量控制）")

    print(f"      ok={res.ok}  stages={res.stages}")
    if res.errors:
        for e in res.errors:
            print("      error:", e)
    print(f"      结果：{(res.out_dir / 'result.json').relative_to(ROOT)}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
