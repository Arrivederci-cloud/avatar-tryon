"""单图生成三视图。

这是整个项目最容易做错的一环。核心约束只有一条：

    几何来自参数化先验，纹理才来自生成模型。

如果把三视图整张交给扩散模型去「想」，正面和背面会各自脑补出一张脸
（Janus 问题），而且三张图的肩宽腰线互相对不上，后面的重建必然失败。

所以这里的做法是：
    1. 用 geometry（MHR/SMPL-X/LHM）在 yaw = 0 / 90 / 180 三个固定机位渲染轮廓与法线
    2. 正面纹理由原图反投影得到，几乎无损
    3. 侧面与背面需要补的面积，落在 UV 空间而不是新视角空间
    4. UV 空间只有一个，不存在「多视角各自生成」的可能，天然规避 Janus

本模块的 UV 链路是可运行的真实现（纯 stdlib，Pillow 可选）：
    a. 解析 OBJ 网格（不可解析时退化为程序化圆柱代理网格）
    b. 网格没有 vt 时按圆柱展开参数化（正式部署应由参数化模型自带 layout）
    c. 对每个 UV texel 计算三个固定机位下的最优入射余弦：
       三个机位都只能掠射（cos < GRAZE_COS）的 texel 即待补区，写入掩码
    d. 有 Pillow 时把原图按正交投影反投影进 UV，得到正面纹理
    e. 待补区交给 edit_model.inpaint 补全
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from ..camera import VIEW_YAW, camera_dirs
from ..models.base import EditModel, HumanGeometry, ThreeViews, ViewModel
from ..utils.image import write_gray_png
from ..utils.matte import erode, person_matte
from ..utils.mesh import load_obj, proxy_cylinder, to_obj, triangles
from ..utils.silhouette import band_of, mesh_ranges, width_profile

# 固定机位定义在 ..camera（渲染器与 pipeline 共用），这里只做转出以免旧引用失效
_camera_dirs = camera_dirs

# UV 分辨率。掩码与纹理同尺寸；生产环境可调到 1024。
UV_SIZE = 256

# 三个机位最优入射余弦低于此值视为掠射（入射角约 75.5 度以上），
# 该 texel 在任何一张视图里都看不到可靠颜色，必须生成补全。
GRAZE_COS = 0.25

INPAINT_PROMPT = "接缝处的皮肤与面料纹理，与周围一致的中性光照，无 logo"

# 兼容旧引用（历史上这些工具长在本模块里）
_load_obj = load_obj
_to_obj = to_obj
_triangles = triangles
_proxy_mesh = proxy_cylinder


@dataclass(slots=True)
class ViewBundle:
    views: ThreeViews
    uv_texture: Path | None
    uv_mask: Path | None = None
    coverage: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def build(
    image,
    geometry: HumanGeometry,
    view_model: ViewModel,
    edit_model: EditModel,
    fill_uv: bool = True,
) -> ViewBundle:
    notes: list[str] = []

    # 软件渲染器需要先有 UV 纹理才能把纹理贴到三视图上；
    # 生成模型（LHM）自己会出视图，走「先渲染后补 UV」的顺序。
    texture_first = getattr(view_model, "requires_texture", False)

    uv_texture: Path | None = None
    uv_mask: Path | None = None
    coverage: dict[str, float] = {}
    if fill_uv and texture_first:
        uv_texture, uv_mask, coverage = _complete_uv(image, geometry, edit_model, notes)
        view_model.attach_texture(uv_texture)  # type: ignore[attr-defined]

    # step 1：几何投影出三张轮廓
    views = view_model.render_views(image, geometry)
    notes.append("front 纹理来自原图反投影；side/back 的可见部分由几何约束投影得到")

    # step 2：UV 空间投影烘焙（三张视图都已就绪，逐机位投影进 UV）
    if fill_uv and not texture_first:
        uv_texture, uv_mask, coverage = _complete_uv(
            image, geometry, edit_model, notes, views=views
        )
    # 注意：texture_first（软件渲染）引擎不要做「第二遍烘焙」。
    # 它渲染出的侧/背视图颜色来自同一张纹理，没有新信息；
    # 实测第二遍会把第一遍 inpaint 补出的衣服颜色覆盖成渲染图的灰色。

    notes.append("side 与 back 为生成结果，UI 上必须标注，不能当成观测")
    return ViewBundle(
        views=views,
        uv_texture=uv_texture,
        uv_mask=uv_mask,
        coverage=coverage,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# UV 补全主流程
# ---------------------------------------------------------------------------


def _complete_uv(
    image,
    geometry: HumanGeometry,
    edit_model: EditModel,
    notes: list[str],
    views: ThreeViews | None = None,
):
    """把三视图投影烘焙到 UV，再对残留的掠射区做 inpaint。

    关键设计：**颜色来源是「视图」而不是「生成模型对 UV 的直接重绘」**。
    实测通用图像编辑 API 不认识 UV 图集，会把贴图当成商品图整张重生
    （已观测区域改动 58/255）；而把侧/背视图按机位投影进 UV 是确定的几何运算，
    模型只在「三张视图都看不见」的窄缝上兜底。

    返回 (补全后的 UV 纹理路径, 掩码路径, 覆盖统计)。
    """
    out_dir = image.parent / "uv"
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_obj(geometry.mesh_path)
    if mesh is None:
        notes.append("mesh 不可解析为 OBJ，使用程序化圆柱代理网格保证链路可跑通")
        mesh = proxy_cylinder()
    verts, uvs, faces = mesh
    if uvs is None:
        notes.append("OBJ 缺少 vt 坐标，按圆柱展开参数化（正式实现应由模型自带 layout）")
        uvs = _cylindrical_uvs(verts)

    try:
        tex_raw, mask_path, stats = _backproject_views(
            image, views, verts, uvs, faces, UV_SIZE, out_dir
        )
    except ImportError:
        notes.append("未安装 Pillow，跳过投影烘焙；pip install pillow 后可生成 UV 纹理")
        return None, None, {}

    notes.append(
        f"UV 烘焙：{stats['views']} 张视图参与投影"
        f"（{stats['banded_views']:.0f} 张按轮廓剖面逐带对齐），"
        f"残留待补 {stats['fill_ratio']:.1%}"
    )
    filled = edit_model.inpaint(image=tex_raw, mask=mask_path, prompt=INPAINT_PROMPT)
    return filled, mask_path, stats


def _fill_pixels(front_cos, bg_flags) -> bytes:
    return bytes(
        255 if (f < GRAZE_COS or (bg_flags is not None and bg_flags[i])) else 0
        for i, f in enumerate(front_cos)
    )


# ---------------------------------------------------------------------------
# 观测掩码：三个固定机位下每个 UV texel 的可见性
# ---------------------------------------------------------------------------


def _observation(verts, uvs, faces, size: int):
    """对每个 UV texel 记录三机位最优入射余弦、来源机位与正面入射余弦。

    best:      长度 size*size，三机位最优余弦，-1.0 表示未被任何三角形覆盖
    cam_of:    0=front 1=side 2=back，-1 表示无
    front_cos: 正面机位的入射余弦。UV 纹理只有正面颜色，
               所以「待补区」必须按它判定，而不是按三机位最优。
    """
    cams = list(camera_dirs().values())
    best = [-1.0] * (size * size)
    cam_of = [-1] * (size * size)
    front_cos = [-1.0] * (size * size)

    def visit(x: int, y: int, w0: float, w1: float, w2: float, cs) -> None:
        # 平面三角形法线统一：机位余弦取最优即可，无需按顶点插值
        k = 0
        obs = cs[0]
        for i in (1, 2):
            if cs[i] > obs:
                obs, k = cs[i], i
        idx = y * size + x
        if obs > best[idx]:
            best[idx] = obs
            cam_of[idx] = k if obs > 0.0 else -1
        if cs[0] > front_cos[idx]:
            front_cos[idx] = cs[0]

    for p0, p1, p2, _a, _b, _c, n in triangles(verts, uvs, faces):
        payload = tuple(n[0] * d[0] + n[1] * d[1] + n[2] * d[2] for d in cams)
        _raster_uv(
            p0, p1, p2, size,
            lambda x, y, w0, w1, w2, c=payload: visit(x, y, w0, w1, w2, c),
        )
    return best, cam_of, front_cos


def _stats(best, cam_of, front_cos) -> dict[str, float]:
    covered = sum(1 for b in best if b >= 0.0)
    visible = sum(1 for b in best if b >= GRAZE_COS)
    front_vis = sum(1 for f in front_cos if f >= GRAZE_COS)
    cams = [0, 0, 0]
    for i, b in enumerate(best):
        if b >= GRAZE_COS and cam_of[i] >= 0:
            cams[cam_of[i]] += 1
    vis = sum(cams) or 1
    return {
        "texels": float(covered),
        # fill_ratio：正面纹理里没有颜色来源、必须补全的比例
        "fill_ratio": round((covered - front_vis) / covered, 4) if covered else 1.0,
        # 三机位联合观测仍补不上的比例，用于「几何质量不足」告警
        "observed_fill_ratio": round((covered - visible) / covered, 4) if covered else 1.0,
        "front_ratio": round(cams[0] / vis, 4),
        "side_ratio": round(cams[1] / vis, 4),
        "back_ratio": round(cams[2] / vis, 4),
    }


def _dilate(pixels: bytes, size: int, radius: int = 1) -> bytes:
    """8 邻域膨胀（纯 stdlib，供无 cv2 环境使用）。"""
    if radius <= 0:
        return pixels
    out = bytearray(pixels)
    for y in range(size):
        row = y * size
        for x in range(size):
            if pixels[row + x]:
                continue
            hit = False
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= size:
                    continue
                base = ny * size
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if 0 <= nx < size and pixels[base + nx]:
                        out[row + x] = 255
                        hit = True
                        break
                if hit:
                    break
    return bytes(out)


def _write_mask(path: Path, pixels) -> Path:
    """掩码约定：白色(255) = 待 inpaint 区，黑色(0) = 保留已观测颜色。"""
    size = int(len(pixels) ** 0.5)
    return write_gray_png(path, size, size, bytes(pixels))


# ---------------------------------------------------------------------------
# 正面颜色反投影（需要 Pillow）
# ---------------------------------------------------------------------------


# 背景色容差：采样色与估计背景的欧氏距离低于此值视为背景。
# 正式方案应换成分割/抠图模型；这里只处理棚拍类纯色背景。
BG_TOLERANCE = 36.0

# 贴图采样时向内收缩的边缘保护像素数：轮廓边缘混了背景的抗锯齿色
EDGE_GUARD = 1


def _background_stats(src) -> tuple[tuple[int, int, int], float]:
    """用整圈边框的中位数估背景色，用边框自身离散度定容差。

    只取四角会栽在渐变/暗角上（实测把浅灰背景估成 162,163,157，导致近半背景
    未被剔除、包围盒退回全图）。边框采样能拿到主要背景色，且离散度直接反映
    渐变幅度，容差跟着放大即可。
    """
    w, h = src.size
    step = max(1, w // 64)
    samples: list[tuple[int, int, int]] = []
    for x in range(0, w, step):
        samples.append(src.getpixel((x, 0)))
        samples.append(src.getpixel((x, h - 1)))
    step_y = max(1, h // 64)
    for y in range(0, h, step_y):
        samples.append(src.getpixel((0, y)))
        samples.append(src.getpixel((w - 1, y)))

    def median(vals: list[int]) -> int:
        vals = sorted(vals)
        return vals[len(vals) // 2]

    med = (
        median([s[0] for s in samples]),
        median([s[1] for s in samples]),
        median([s[2] for s in samples]),
    )

    def spread(vals: list[int]) -> int:
        vals = sorted(vals)
        return vals[int(len(vals) * 0.9)] - vals[int(len(vals) * 0.1)]

    worst = max(
        spread([s[0] for s in samples]),
        spread([s[1] for s in samples]),
        spread([s[2] for s in samples]),
    )
    # 上界 48：容差再大就会把浅肤色当成背景吃掉
    tol = min(48.0, max(30.0, worst * 1.5 + 12.0))
    return med, tol


def _background_color(src) -> tuple[int, int, int]:
    return _background_stats(src)[0]


def _person_bbox(src, bg, thresh: float = BG_TOLERANCE):
    """估计人物在照片中的包围盒：与背景色差异明显的行列。

    对纯色/渐变背景的棚拍照片有效；复杂背景下退化回整张图。
    正式方案应换成分割模型，这里只做零依赖的快速对齐。
    """
    factor = 8
    small = src.resize((max(1, src.width // factor), max(1, src.height // factor)))
    w, h = small.size
    px = small.load()
    cols = [0] * w
    rows = [0] * h
    for y in range(h):
        for x in range(w):
            c = px[x, y]
            if math.sqrt(sum((c[k] - bg[k]) ** 2 for k in range(3))) > thresh:
                cols[x] += 1
                rows[y] += 1

    def bounds(arr, limit: int) -> tuple[int, int]:
        need = max(1, int(limit * 0.02))
        idx = [i for i, v in enumerate(arr) if v >= need]
        if not idx:
            return 0, len(arr) - 1
        return idx[0], idx[-1]

    x0, x1 = bounds(cols, h)
    y0, y1 = bounds(rows, w)
    sx = src.width / w
    sy = src.height / h
    box = (int(x0 * sx), int(y0 * sy), int((x1 + 1) * sx), int((y1 + 1) * sy))
    # 包围盒过小说明背景估计失败，退回整图
    if (box[2] - box[0]) < src.width * 0.25 or (box[3] - box[1]) < src.height * 0.25:
        return (0, 0, src.width, src.height)
    return box


# 各机位图像的水平轴对应的世界方向，必须与软件渲染器一致，否则纹理左右翻转
VIEW_RIGHT = {
    "front": (1.0, 0.0, 0.0),
    "side": (0.0, 0.0, 1.0),
    "back": (-1.0, 0.0, 0.0),
}


def _is_bg(color, bg, thresh: float = BG_TOLERANCE) -> bool:
    return math.sqrt(sum((color[k] - bg[k]) ** 2 for k in range(3))) < thresh


def _backproject_views(image, views, verts, uvs, faces, size: int, out_dir: Path):
    """把三视图按固定机位投影烘焙进 UV。

    每个 texel 只采信「法线与该机位夹角小于掠射阈值」的视图，
    并在多张视图都可见时取入射角最小（最正对）的那张，避免接缝处出现重影。

    返回 (原始纹理路径, 掩码路径, 统计)。掩码白色 = 三张视图都补不上的窄缝。
    """
    from PIL import Image  # 可选依赖，缺失时上层降级

    cams = camera_dirs()
    ys = [v[1] for v in verts]
    ymin, ymax = min(ys), max(ys)
    y_span = (ymax - ymin) or 1.0

    best = [-1.0] * (size * size)
    covered = bytearray(size * size)  # 本组视图采样到的 texel
    colors: list[tuple[int, int, int]] = [(255, 255, 255)] * (size * size)
    tris = list(triangles(verts, uvs, faces))

    # 图集总面积：所有三角形（不管哪个机位可见）覆盖的 texel。
    # 分母必须用它——若用「本组视图可见三角形的覆盖」，视图越少分母越小，
    # fill_ratio 会虚低（单视图时后半个身体全空也不计入）。
    atlas_all = bytearray(size * size)

    def _mark(x: int, y: int, w0: float, w1: float, w2: float) -> None:
        atlas_all[y * size + x] = 1

    for p0, p1, p2, *_rest in tris:
        _raster_uv(p0, p1, p2, size, _mark)

    sources = [
        ("front", views.front if views is not None else image),
        ("side", getattr(views, "side", None)),
        ("back", getattr(views, "back", None)),
    ]
    used = 0
    banded = 0
    for name, path in sources:
        if path is None or not Path(path).exists():
            continue
        src = Image.open(path).convert("RGB")
        sw, sh = src.size
        bg, tol = _background_stats(src)
        bx0, by0, bx1, by1 = _person_bbox(src, bg, tol)

        # 优先用 GrabCut 抠图（对渐变背景稳健）；cv2 缺失时退回颜色启发式
        matte, mbox = person_matte(src, rect=(bx0, by0, bx1, by1))
        if matte is not None:
            bx0, by0, bx1, by1 = mbox
            # 背景判定用收缩后的掩码：轮廓边缘 1-2px 混了背景的抗锯齿色，
            # 直接采样会在模型边缘形成灰边；剖面统计仍用原始掩码
            mpx = erode(matte, EDGE_GUARD).load()
        else:
            mpx = None
        cam = cams[name]
        right = VIEW_RIGHT[name]
        proj = [v[0] * right[0] + v[2] * right[2] for v in verts]
        p_min, p_max = min(proj), max(proj)
        p_span = (p_max - p_min) or 1.0
        used += 1

        # 逐带宽度剖面：网格在高度 y 的横向范围映射到照片在同高度的横向范围。
        # 只用「总宽度对包围盒」的单一线性映射时，网格比真人宽的部位（外展手臂）
        # 会落到轮廓外的背景上被丢弃，这是代理几何残留待补的最大来源。
        prof = width_profile(matte) if matte is not None else None
        if prof is not None:
            m_lo, m_hi = mesh_ranges(verts, right, ymin, y_span, prof.bands)
            prof_bands, plo, phi = prof.bands, prof.lo, prof.hi
            global_scale = (bx1 - bx0) / p_span
            banded += 1
        else:
            m_lo = m_hi = None
            prof_bands, plo, phi = 0, [], []
            global_scale = 0.0

        def visit(x, y, w0, w1, w2, tri, cos, *, src=src, sw=sw, sh=sh, bg=bg, tol=tol,
                  mpx=mpx, right=right, p_min=p_min, p_span=p_span,
                  box=(bx0, by0, bx1, by1), prof_bands=prof_bands, plo=plo, phi=phi,
                  m_lo=m_lo, m_hi=m_hi, gs=global_scale) -> None:
            idx = y * size + x
            covered[idx] = 1
            a, b, c = tri
            wx = w0 * a[0] + w1 * b[0] + w2 * c[0]
            wy = w0 * a[1] + w1 * b[1] + w2 * c[1]
            wz = w0 * a[2] + w1 * b[2] + w2 * c[2]
            p = wx * right[0] + wz * right[2]
            u = (p - p_min) / p_span
            v = 1.0 - (wy - ymin) / y_span
            sx = min(sw - 1, max(0, int(box[0] + u * (box[2] - box[0]))))
            sy = min(sh - 1, max(0, int(box[1] + v * (box[3] - box[1]))))
            if prof_bands:
                bb = band_of(v, prof_bands)
                mw = m_hi[bb] - m_lo[bb]
                bw = phi[bb] - plo[bb]
                if mw > 1e-6 and bw > 2.0:
                    local = bw / mw
                    # 局部尺度离全局太远说明该带退化（如头顶点环），退回全局映射
                    if gs <= 0.0 or 0.4 * gs <= local <= 2.5 * gs:
                        sx = min(sw - 1, max(0, int(plo[bb] + (p - m_lo[bb]) * local)))
            if mpx is not None:
                if mpx[sx, sy] < 128:  # 抠图判定为背景
                    return
            elif _is_bg(src.getpixel((sx, sy)), bg, tol):
                return
            color = src.getpixel((sx, sy))
            if cos > best[idx]:
                best[idx] = cos
                colors[idx] = color

        for p0, p1, p2, a, b, c, n in tris:
            cos = n[0] * cam[0] + n[1] * cam[1] + n[2] * cam[2]
            if cos <= GRAZE_COS:
                continue
            _raster_uv(
                p0, p1, p2, size,
                lambda x, y, w0, w1, w2, t=(a, b, c), k=cos: visit(
                    x, y, w0, w1, w2, t, k
                ),
            )

    out = Image.new("RGB", (size, size), (255, 255, 255))
    px = out.load()
    filled = 0
    for i, col in enumerate(colors):
        if best[i] > 0.0:
            px[i % size, i // size] = col
        elif atlas_all[i]:
            filled += 1
    path = out_dir / "uv_multiview.png"
    out.save(path, "PNG")

    # 掩码只标记「图集内且三视图都没采到」的窄缝，图集外的空白不参与补全。
    # 覆盖区先膨胀 1px：图集矩形边缘的 texel 常落在三角形光栅化之外
    # （u=0.02*256=5.12，整数栅格从 x=6 开始），这 1px 从未被采样却会被
    # 渲染器采到，不标记成待补的话就是一条贯穿正面的白缝。
    atlas_edge = _dilate(bytes(atlas_all), size)
    mask_path = write_gray_png(
        out_dir / "uv_mask.png",
        size,
        size,
        bytes(
            255 if (atlas_edge[i] and best[i] <= 0.0) else 0
            for i in range(size * size)
        ),
    )
    atlas = sum(atlas_all) or 1
    return path, mask_path, {
        "views": float(used),
        "banded_views": float(banded),
        "atlas_texels": float(atlas),
        "sampled_texels": float(sum(covered)),
        "fill_ratio": round(filled / atlas, 4),
        "texels": float(atlas - filled),
    }


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------


def _cylindrical_uvs(verts):
    ys = [v[1] for v in verts]
    ymin, ymax = min(ys), max(ys)
    span = (ymax - ymin) or 1.0
    return [
        ((math.atan2(x, z) + math.pi) / (2.0 * math.pi), 1.0 - (y - ymin) / span)
        for x, y, z in verts
    ]


# ---------------------------------------------------------------------------
# UV 光栅化
# ---------------------------------------------------------------------------


def _raster_uv(p0, p1, p2, size: int, visit) -> None:
    """UV 三角形光栅化。uv 的 v 轴翻转后扫描，命中像素回调 visit(x, y, w0, w1, w2)。"""
    us = (p0[0], p1[0], p2[0])
    if max(us) - min(us) <= 0.5:
        _raster_uv_core(p0, p1, p2, size, visit)
        return
    # UV 接缝环绕：u 从 ~1 跳回 ~0，朴素光栅会横扫整张纹理。
    # 把两侧顶点各平移一个周期后分别光栅，命中按 x mod size 回调，重复命中由上层取 max。
    a = tuple((u + 1.0, v) if u < 0.5 else (u, v) for u, v in (p0, p1, p2))
    b = tuple((u - 1.0, v) if u >= 0.5 else (u, v) for u, v in (p0, p1, p2))
    for tri in (a, b):
        _raster_uv_core(
            tri[0], tri[1], tri[2], size,
            lambda x, y, w0, w1, w2: visit(x % size, y, w0, w1, w2),
        )


def _raster_uv_core(p0, p1, p2, size: int, visit) -> None:
    x0, y0 = p0[0] * size, (1.0 - p0[1]) * size
    x1, y1 = p1[0] * size, (1.0 - p1[1]) * size
    x2, y2 = p2[0] * size, (1.0 - p2[1]) * size
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(det) < 1e-12:
        return
    minx = max(0, int(math.floor(min(x0, x1, x2))))
    maxx = min(size - 1, int(math.ceil(max(x0, x1, x2))))
    miny = max(0, int(math.floor(min(y0, y1, y2))))
    maxy = min(size - 1, int(math.ceil(max(y0, y1, y2))))
    for y in range(miny, maxy + 1):
        py = y + 0.5  # 像素中心约定：否则贴着矩形边缘的三角形会漏掉 1px 列
        for x in range(minx, maxx + 1):
            px = x + 0.5
            w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
            w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
            w2 = 1.0 - w0 - w1
            if w0 >= -1e-9 and w1 >= -1e-9 and w2 >= -1e-9:
                visit(x, y, w0, w1, w2)


__all__ = [
    "GRAZE_COS",
    "INPAINT_PROMPT",
    "UV_SIZE",
    "VIEW_YAW",
    "ViewBundle",
    "build",
    "camera_dirs",
]
