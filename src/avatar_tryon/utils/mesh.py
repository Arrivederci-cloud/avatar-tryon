"""网格工具：OBJ 读写、三角化、法线。纯 stdlib，不依赖 numpy。

单独成模块的原因：pipeline 与 models 两侧都要用（三视图 UV 反投影、软件渲染器），
放在任一方的内部都会造成循环导入。
"""

from __future__ import annotations

import math
from pathlib import Path

Face = tuple[list[int], list[int] | None]
Triangle = tuple[
    tuple[float, float],  # p0 (u, v)
    tuple[float, float],  # p1
    tuple[float, float],  # p2
    tuple[float, float, float],  # 3D 顶点 a
    tuple[float, float, float],  # b
    tuple[float, float, float],  # c
    tuple[float, float, float],  # 朝外单位法线
]


def load_obj(path: Path):
    """解析 OBJ 的 v / vt / f。返回 (verts, uvs 或 None, faces)；不可解析返回 None。

    faces 元素为 (v 索引列表, vt 索引列表或 None)，均为 1-based，多边形按扇形三角化。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    faces: list[Face] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v" and len(parts) >= 4:
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tag == "vt" and len(parts) >= 3:
            uvs.append((float(parts[1]), float(parts[2])))
        elif tag == "f" and len(parts) >= 4:
            v_ids: list[int] = []
            t_ids: list[int | None] = []
            for tok in parts[1:]:
                seg = tok.split("/")
                v_ids.append(int(seg[0]))
                t_ids.append(int(seg[1]) if len(seg) > 1 and seg[1] else None)
            faces.append((v_ids, t_ids if all(t is not None for t in t_ids) else None))
    if len(verts) < 3 or not faces:
        return None
    return verts, (uvs if uvs else None), faces


def to_obj(verts, uvs, faces) -> str:
    """序列化为 OBJ 文本，供调试与测试夹具使用。"""
    lines = [f"v {x} {y} {z}" for x, y, z in verts]
    lines += [f"vt {u} {v}" for u, v in uvs]
    for v_ids, t_ids in faces:
        if t_ids is not None and len(t_ids) == len(v_ids):
            lines.append("f " + " ".join(f"{a}/{b}" for a, b in zip(v_ids, t_ids, strict=False)))
        else:
            lines.append("f " + " ".join(str(a) for a in v_ids))
    return "\n".join(lines) + "\n"


def write_obj(path: Path, verts, uvs, faces) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_obj(verts, uvs, faces), encoding="utf-8")
    return path


def norm_index(i: int, n: int) -> int:
    return i - 1 if i > 0 else n + i


def _islands(uv_ids_list):
    """UV 连通域：共享 uv 顶点的三角形属于同一域。

    分块图集里一个域就是一个身体部件（躯干/头/手臂/腿）。
    返回 {uv_id: 域根}。
    """
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # 路径压缩
            parent[a], a = root, parent[a]
        return root

    for ids in uv_ids_list:
        for i in ids:
            parent.setdefault(i, i)
        first = find(ids[0])
        for i in ids[1:]:
            r = find(i)
            if r != first:
                parent[r] = first
    return find


def triangles(verts, uvs, faces):
    """产出 Triangle。法线按「背离所属 UV 连通域的质心」定向。

    为什么不能用整网格质心：手臂、腿是偏心部件，按整网格质心定向会把
    朝向躯干那一侧的法线整个翻转（左臂全部法线被掰成 -x，右臂全 +x）。
    后果不只是光照错——三视图投影靠法线与机位夹角剔除掠射面，侧视机位下
    左臂会被整体判为背向而不参与烘焙（实测 arm_l 残留 97%，arm_r 仅 15%）。
    按 UV 域（即部件）取质心，域质心落在部件轴线上，定向才符合几何直觉。
    """
    n_v = len(verts)
    n_t = len(uvs)

    expanded = []
    uv_groups = []
    for v_ids, t_ids in faces:
        ids = [norm_index(i, n_v) for i in v_ids]
        if any(i < 0 or i >= n_v for i in ids):
            continue
        uv_ids = ids
        if t_ids is not None and len(t_ids) == len(v_ids):
            cand = [norm_index(t, n_t) for t in t_ids]
            if all(0 <= t < n_t for t in cand):
                uv_ids = cand
        for k in range(1, len(ids) - 1):
            ja, jb, jc = uv_ids[0], uv_ids[k], uv_ids[k + 1]
            expanded.append((ids[0], ids[k], ids[k + 1], ja, jb, jc))
            uv_groups.append((ja, jb, jc))

    find = _islands(uv_groups)
    acc: dict[int, list[float]] = {}
    for ia, ib, ic, ja, _jb, _jc in expanded:
        key = find(ja)
        g = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
        for i in (ia, ib, ic):
            g[0] += verts[i][0]
            g[1] += verts[i][1]
            g[2] += verts[i][2]
        g[3] += 3
    centers = {
        key: (g[0] / g[3], g[1] / g[3], g[2] / g[3]) for key, g in acc.items()
    }

    for ia, ib, ic, ja, jb, jc in expanded:
        a, b, c = verts[ia], verts[ib], verts[ic]
        abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = aby * acz - abz * acy
        ny = abz * acx - abx * acz
        nz = abx * acy - aby * acx
        gx = (a[0] + b[0] + c[0]) / 3.0 - centers[find(ja)][0]
        gy = (a[1] + b[1] + c[1]) / 3.0 - centers[find(ja)][1]
        gz = (a[2] + b[2] + c[2]) / 3.0 - centers[find(ja)][2]
        if nx * gx + ny * gy + nz * gz < 0.0:
            nx, ny, nz = -nx, -ny, -nz
        norm = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        yield (
            uvs[ja],
            uvs[jb],
            uvs[jc],
            a,
            b,
            c,
            (nx / norm, ny / norm, nz / norm),
        )


def proxy_cylinder(seg: int = 16, rings: int = 8):
    """程序化圆柱：mock mesh 不可解析时的兜底，以及观测逻辑的测试夹具。"""
    verts, uvs, faces = [], [], []
    for j in range(rings + 1):
        y = j / rings
        for i in range(seg):
            ang = 2.0 * math.pi * i / seg
            verts.append((math.sin(ang), y, math.cos(ang)))
            uvs.append((i / seg, 1.0 - y))
    for j in range(rings):
        for i in range(seg):
            a = j * seg + i
            b = j * seg + (i + 1) % seg
            c = (j + 1) * seg + (i + 1) % seg
            d = (j + 1) * seg + i
            faces.append(([a + 1, b + 1, c + 1], [a + 1, b + 1, c + 1]))
            faces.append(([a + 1, c + 1, d + 1], [a + 1, c + 1, d + 1]))
    return verts, uvs, faces


__all__ = [
    "Face",
    "Triangle",
    "load_obj",
    "norm_index",
    "proxy_cylinder",
    "to_obj",
    "triangles",
    "write_obj",
]
