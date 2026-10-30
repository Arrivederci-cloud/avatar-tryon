"""人体几何相关的适配器：LHM / LHM++ 与 MHR。

两个模型分工不同，不要混用：
- MHR：参数化人体，输出与视角无关的骨架与体尺，是三视图的几何基准。
- LHM++：前馈重建，输出带蒙皮权重的高斯人体，负责外观与驱动。

为什么几何必须来自参数化先验而不是生成图：生成的侧视图与背视图互相矛盾时，
重建会得到「三张脸」（Janus 问题），几何必须有一个唯一的锚。
"""

from __future__ import annotations

from pathlib import Path

from .base import HumanGeometry, ReconModel, ThreeViews, ViewModel
from .errors import ModelNotWired


class LHMRecon(ReconModel):
    name = "lhm++"
    license = "Apache-2.0"
    upstream = "https://github.com/aigc3d/LHM"

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir

    def reconstruct(self, image: Path, height_cm: float) -> HumanGeometry:
        if not (self.weights_dir / "lhm").exists():
            raise ModelNotWired(
                "未找到 LHM 权重与仓库。上游的推理依赖仓库内的脚本，不在 PyPI 上，"
                "需按官方步骤准备：\n"
                "  git clone https://github.com/aigc3d/LHM.git\n"
                "  # 下载 prior model 权重并解压到仓库根目录\n"
                "  wget https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/"
                "data/LHM/LHM_prior_model.tar && tar -xvf LHM_prior_model.tar\n"
                "  # 推理（第三个参数是 SMPL-X 动作序列目录）\n"
                "  bash inference.sh LHM-500M-HF <图片目录> <smplx_params 目录>\n"
                "  bash inference_mesh.sh LHM-500M-HF   # 导出网格\n"
                "\n"
                "注意：LHM 的 CLI 设计是「图 + 动作 -> 驱动后的结果」，"
                "纯静态重建也要传一个动作序列；\n"
                "本适配器后续应封装上述两条命令的子进程调用，而不是重新实现模型。"
            )
        raise ModelNotWired("LHM 子进程封装尚未接入，命令见上方报错信息与 docs/pipeline.md")


class LHMRender(ViewModel):
    name = "lhm++-render"
    license = "Apache-2.0"
    upstream = "https://github.com/aigc3d/LHM"

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir

    def render_views(self, image: Path, geometry: HumanGeometry) -> ThreeViews:
        raise ModelNotWired(
            "三视图渲染尚未接入。实现要点：以 geometry 为唯一几何源，"
            "在 yaw=0/90/180 三个固定机位渲染，正面贴原图反投影纹理，"
            "侧面与背面走 UV 空间补全（见 models/edit_adapter 的 EditModel）。"
        )


class MHRRecon(ReconModel):
    """只做几何与体尺，不做外观。"""

    name = "mhr"
    license = "Apache-2.0"
    upstream = "https://github.com/facebookresearch/MHR"

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir

    def reconstruct(self, image: Path, height_cm: float) -> HumanGeometry:
        if not (self.weights_dir / "mhr").exists():
            raise ModelNotWired(
                f"未找到 MHR 权重：{self.weights_dir / 'mhr'}\n"
                f"先运行 python scripts/fetch_weights.py --model mhr\n"
                f"上游实现：{self.upstream}"
            )
        raise ModelNotWired("MHR 前向推理尚未接入；注意围度需身高归一化后才可比")
