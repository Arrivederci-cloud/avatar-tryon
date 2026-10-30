"""三视图软件渲染器：CPU 上把 UV 纹理贴回固定机位。

定位：这是「不依赖 GPU 也能出图」的实现，用于链路验证与 CI 回归。
它**不是产品效果**——几何来自参数化先验、纹理侧面背面靠补全外推，
真实效果需要 LHM/Qwen-Image-Edit 这条 real 档路线。

两条渲染路径：
    1. numpy 存在时：逐像素 z-buffer + 顶点法线 Gouraud 插值 + 主光，
       遮挡关系正确、表面平滑，是给人看的那条；
    2. 没有 numpy 时：画家算法逐面填色（按重心深度排序），只保证能出图。

为什么值得留一个 CPU 渲染器：
  1. 三视图是流水线核心中间产物，必须能在没有显卡的机器上回归测试；
  2. 它把「UV 补全是否真的补对了」变成肉眼可判的事——补错了一眼就看出来。
"""

from __future__ import annotations

import math
from pathlib import Path

from ..camera import VIEW_YAW
from ..utils.mesh import load_obj, triangles
from .base import HumanGeometry, ThreeViews, ViewModel
from .errors import ModelNotWired

BACKGROUND = (245, 245, 245)
# 主光方向（相机空间，指向光源）。跟着相机走，三个机位曝光一致。
LIGHT_DIR = (-0.35, 0.32, 0.88)
AMBIENT = 0.32


class SoftwareViews(ViewModel):
    name = "software-render"
    license = "Apache-2.0（本仓库原创）"
    # 提示 three_view.build：先出 UV 纹理，再渲染三视图
    requires_texture = True

    def __init__(self, size: tuple[int, int] = (720, 1080)) -> None:
        self.size = size
        self._texture: Path | None = None

    def attach_texture(self, path: Path | None) -> None:
        self._texture = path

    def render_views(self, image: Path, geometry: HumanGeometry) -> ThreeViews:
        mesh = load_obj(geometry.mesh_path)
        if mesh is None:
            raise ModelNotWired(f"软件渲染器需要可解析的 OBJ 网格：{geometry.mesh_path}")
        verts, uvs, faces = mesh
        if uvs is None:
            raise ModelNotWired("网格缺少 vt 坐标，软件渲染器无法贴图")

        tris = list(triangles(verts, uvs, faces))
        if not tris:
            raise ModelNotWired("网格没有可用三角形")

        texture = self._load_texture()
        scale, cx, cy = _frame(verts, self.size)
        out_dir = image.parent / "views"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for name, yaw in VIEW_YAW.items():
            img = self._render(tris, texture, yaw, scale, cx, cy)
            p = out_dir / f"{name}.png"
            img.save(p, "PNG")
            paths[name] = p

        return ThreeViews(
            front=paths["front"],
            side=paths["side"],
            back=paths["back"],
            confidence={"front": 0.98, "side": 0.55, "back": 0.40},
            notes=[
                "三视图由代理几何 + UV 纹理软件渲染得到（CPU 预览，非产品效果）",
                "side/back 的置信度反映纹理里生成区域的比例，不是测量精度",
            ],
        )

    def _load_texture(self):
        if self._texture is None:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        return Image.open(self._texture).convert("RGB")

    def _render(self, tris, texture, yaw: float, scale: float, cx: float, cy: float):
        try:
            return _render_zbuffer(tris, texture, yaw, scale, cx, cy, self.size)
        except ImportError:
            return _render_painter(tris, texture, yaw, scale, cx, cy, self.size)


# ---------------------------------------------------------------------------
# 路径一：numpy z-buffer + Gouraud
# ---------------------------------------------------------------------------


def _render_zbuffer(tris, texture, yaw: float, scale: float, cx: float, cy: float, size):
    import numpy as np
    from PIL import Image

    w, h = size
    r = math.radians(yaw)
    cos_r, sin_r = math.cos(r), math.sin(r)

    n = len(tris)
    pos = np.zeros((n, 3, 3), dtype=np.float32)
    uvs = np.zeros((n, 3, 2), dtype=np.float32)
    fnrm = np.zeros((n, 3), dtype=np.float32)
    for i, (p0, p1, p2, a, b, c, nrm) in enumerate(tris):
        pos[i] = (a, b, c)
        uvs[i] = (p0, p1, p2)
        fnrm[i] = nrm

    # 顶点法线：把同位置的面法线累加后再归一化，得到平滑着色所需的法线
    vn = _vertex_normals(pos, fnrm)

    # 相机变换：绕 y 轴旋转 yaw，正交投影
    x = pos[..., 0] * cos_r + pos[..., 2] * sin_r
    z = -pos[..., 0] * sin_r + pos[..., 2] * cos_r
    sx = w / 2.0 + (x - cx) * scale
    sy = h * 0.95 - (pos[..., 1] - cy) * scale

    light = np.array(LIGHT_DIR, dtype=np.float32)
    light /= np.linalg.norm(light)
    # 法线跟着相机一起转
    nrm_cam = np.stack(
        [
            vn[..., 0] * cos_r + vn[..., 2] * sin_r,
            vn[..., 1],
            -vn[..., 0] * sin_r + vn[..., 2] * cos_r,
        ],
        axis=-1,
    )
    shade = AMBIENT + (1.0 - AMBIENT) * np.clip(nrm_cam @ light, 0.0, 1.0)

    colors = _sample_vertices(texture, uvs, w, h) * shade[..., None]

    color_buf = np.empty((h, w, 3), dtype=np.float32)
    color_buf[:] = BACKGROUND
    zbuf = np.full((h, w), -1e9, dtype=np.float32)

    for i in range(n):
        _raster_tri(
            sx[i], sy[i], z[i], colors[i],
            color_buf, zbuf, w, h,
        )

    return Image.fromarray(np.clip(color_buf, 0, 255).astype(np.uint8), "RGB")


def _vertex_normals(pos, fnrm):
    """把相同位置的面法线累加，得到顶点法线（平滑着色）。"""
    import numpy as np

    n = pos.shape[0]
    flat = pos.reshape(-1, 3)
    quant = np.round(flat * 1000.0).astype(np.int64)
    _, inverse = np.unique(quant, axis=0, return_inverse=True)
    acc = np.zeros((inverse.max() + 1, 3), dtype=np.float32)
    for k in range(3):
        np.add.at(acc, inverse[k::3], fnrm)
    norm = np.linalg.norm(acc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    acc /= norm
    out = acc[inverse].reshape(n, 3, 3)
    return out


def _sample_vertices(texture, uvs, w, h):
    import numpy as np

    if texture is None:
        return np.full(uvs.shape[:2] + (3,), 200.0, dtype=np.float32)
    tw, th = texture.size
    xs = np.clip((uvs[..., 0] * tw).astype(np.int32), 0, tw - 1)
    ys = np.clip(((1.0 - uvs[..., 1]) * th).astype(np.int32), 0, th - 1)
    px = np.asarray(texture, dtype=np.uint8)
    return px[ys, xs].astype(np.float32)


def _raster_tri(sx, sy, z, col, color_buf, zbuf, w, h) -> None:
    import numpy as np

    x0, x1 = float(sx.min()), float(sx.max())
    y0, y1 = float(sy.min()), float(sy.max())
    if x1 < 0 or y1 < 0 or x0 > w - 1 or y0 > h - 1:
        return
    xa = max(0, int(math.floor(x0)))
    xb = min(w - 1, int(math.ceil(x1)))
    ya = max(0, int(math.floor(y0)))
    yb = min(h - 1, int(math.ceil(y1)))
    if xb < xa or yb < ya:
        return

    denom = (sy[1] - sy[2]) * (sx[0] - sx[2]) + (sx[2] - sx[1]) * (sy[0] - sy[2])
    if abs(denom) < 1e-9:
        return

    ys_i, xs_i = np.meshgrid(
        np.arange(ya, yb + 1), np.arange(xa, xb + 1), indexing="ij"
    )
    pxf = xs_i.astype(np.float32) + 0.5
    pyf = ys_i.astype(np.float32) + 0.5
    w0 = ((sy[1] - sy[2]) * (pxf - sx[2]) + (sx[2] - sx[1]) * (pyf - sy[2])) / denom
    w1 = ((sy[2] - sy[0]) * (pxf - sx[2]) + (sx[0] - sx[2]) * (pyf - sy[2])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
    sub_z = zbuf[ya : yb + 1, xa : xb + 1]
    update = inside & (depth > sub_z)
    if not update.any():
        return

    sub_c = color_buf[ya : yb + 1, xa : xb + 1]
    for k in range(3):
        sub_c[..., k] = np.where(
            update, w0 * col[0][k] + w1 * col[1][k] + w2 * col[2][k], sub_c[..., k]
        )
    sub_z[update] = depth[update]


# ---------------------------------------------------------------------------
# 路径二：画家算法（无 numpy 时的退路）
# ---------------------------------------------------------------------------


def _render_painter(tris, texture, yaw: float, scale: float, cx: float, cy: float, size):
    from PIL import Image, ImageDraw

    w, h = size
    r = math.radians(yaw)
    cos_r, sin_r = math.cos(r), math.sin(r)
    img = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(img)
    light = _normalize(LIGHT_DIR)

    items = []
    for p0, p1, p2, a, b, c, nrm in tris:
        pts = []
        depth = 0.0
        for p in (a, b, c):
            xr = p[0] * cos_r + p[2] * sin_r
            zr = -p[0] * sin_r + p[2] * cos_r
            pts.append((w / 2 + (xr - cx) * scale, h * 0.95 - (p[1] - cy) * scale))
            depth += zr
        nc = (nrm[0] * cos_r + nrm[2] * sin_r, nrm[1], -nrm[0] * sin_r + nrm[2] * cos_r)
        shade = AMBIENT + (1.0 - AMBIENT) * max(0.0, _dot(nc, light))
        items.append((depth / 3.0, pts, _sample_flat(texture, (p0, p1, p2), shade)))

    items.sort(key=lambda t: t[0])
    for _, pts, color in items:
        draw.polygon(pts, fill=color, outline=color)
    return img


def _sample_flat(texture, uvtri, shade: float):
    if texture is None:
        return (int(200 * shade),) * 3
    tw, th = texture.size
    r = g = b = 0
    for u, v in uvtri:
        x = min(tw - 1, max(0, int(u * tw)))
        y = min(th - 1, max(0, int((1.0 - v) * th)))
        px = texture.getpixel((x, y))
        r += px[0]
        g += px[1]
        b += px[2]
    return (
        int(r / 3 * shade),
        int(g / 3 * shade),
        int(b / 3 * shade),
    )


# ---------------------------------------------------------------------------
# 取景与向量工具
# ---------------------------------------------------------------------------


def _frame(verts, size: tuple[int, int]) -> tuple[float, float, float]:
    """统一取景：三视图共用同一比例尺，否则三张图对不上。"""
    w, h = size
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    height = (max(ys) - min(ys)) or 1.0
    width = max(max(xs) - min(xs), max(zs) - min(zs)) or 1.0
    scale = min(w * 0.86 / width, h * 0.90 / height)
    return scale, (max(xs) + min(xs)) / 2.0, min(ys)


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / n for c in v)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


__all__ = ["SoftwareViews"]
