"""烘焙质量基准：同一组输入下，对比「全局线性映射」与「逐带轮廓对齐」。

用法（在仓库根目录）：
    python scripts/bench_bake.py outputs/demo/full

输出每一项的 fill_ratio（UV 图集内、正面取不到颜色的比例）。
这个数字直接决定最终纹理有多依赖生成补全，越低越好。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avatar_tryon.models.base import ThreeViews  # noqa: E402
from avatar_tryon.pipeline import three_view  # noqa: E402
from avatar_tryon.utils.mesh import load_obj  # noqa: E402


def run(out_dir: Path, banded: bool) -> dict:
    mesh = load_obj(out_dir / "geometry" / "body.obj")
    if mesh is None:
        raise SystemExit("找不到 geometry/body.obj")
    verts, uvs, faces = mesh
    views = ThreeViews(
        front=out_dir / "person_front.png",
        side=out_dir / "views" / "side.png",
        back=out_dir / "views" / "back.png",
    )
    original = three_view.width_profile
    if not banded:
        three_view.width_profile = lambda *_a, **_k: None
    try:
        _p, _m, stats = three_view._backproject_views(
            views.front, views, verts, uvs, faces, three_view.UV_SIZE, out_dir / "uv"
        )
    finally:
        three_view.width_profile = original
    return stats


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "demo" / "full"
    for banded in (False, True):
        stats = run(target, banded)
        tag = "逐带对齐" if banded else "全局映射"
        print(
            f"{tag}: fill_ratio={stats['fill_ratio']:.1%} "
            f"atlas_texels={stats['atlas_texels']:.0f} "
            f"banded_views={stats.get('banded_views', 0):.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
