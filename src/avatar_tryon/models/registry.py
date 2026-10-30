"""模型注册表：按名字取实现，按许可证分类，按引擎档位选择实现。

三档引擎（engine）：
    mock  —— 全部占位，CI 用，秒级跑完
    proxy —— 参数化几何 + 软件渲染 + 基线补全，纯 CPU，用来验证链路与降级
    real  —— LHM / MHR + Qwen-Image-Edit + FASHN，需要权重与显卡

关键约定：只有 `COMMERCIAL_OK` 里的模型可以随仓库分发生效的默认配置；
`OPTIONAL_RESTRICTED` 里的模型必须由使用者自行安装并接受其条款。
"""

from __future__ import annotations

import os
from pathlib import Path

from .api_views import DashscopeViews
from .base import (
    EditModel,
    HairModel,
    ReconModel,
    RigModel,
    TryOnModel,
    ViewModel,
)
from .baseline_edit import BaselineInpaint
from .generation_adapters import (
    DashscopeWanxEdit,
    FashnVton,
    HairFastGan,
    QwenImageEdit,
)
from .mock import MockEdit, MockHair, MockRecon, MockRig, MockTryOn, MockViews
from .proxy import ProxyBodyRecon
from .recon_adapters import LHMRecon, LHMRender, MHRRecon
from .rig_adapter import UniRig
from .software_render import SoftwareViews

ENGINES = ("mock", "proxy", "real")

# 可商用、可再分发：默认配置只允许出现这些
COMMERCIAL_OK: dict[str, str] = {
    "lhm": "Apache-2.0",
    "mhr": "Apache-2.0",
    "qwen_image_edit": "Apache-2.0",
    "fashn_vton": "Apache-2.0",
    "hairfastgan": "MIT",
    "unirig": "MIT",
    "proxy_body": "Apache-2.0（本仓库原创）",
    "software_render": "Apache-2.0（本仓库原创）",
    "baseline_edit": "Apache-2.0（本仓库原创）",
    "qwen_image_edit_api": "托管服务（阿里云 Model Studio）按量计费；权重 Apache-2.0",
}

# 受限或需单独确认：不提供默认实现，只在文档中说明
OPTIONAL_RESTRICTED: dict[str, str] = {
    "sam3d_body": "SAM License (Meta) - 非 OSI，需自行接受条款，勿再分发",
    "magicman": "未见明确许可 - 默认保留所有权利，不可再分发",
    "catvton": "CC BY-NC-SA 4.0 - 禁止商用",
    "idm_vton": "CC BY-NC-SA 4.0 - 禁止商用",
    "trellis2": "研究许可 - 禁止商用再分发",
    "hunyuan3d": "社区许可 - 含月活上限与地域排除",
}


def _check_engine(engine: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"未知 engine：{engine}，可选 {ENGINES}")
    return engine


def build_recon(
    name: str, weights_dir: Path, engine: str = "mock", **_kw: object
) -> ReconModel:
    _check_engine(engine)
    if engine == "mock":
        return MockRecon()
    if engine == "proxy":
        return ProxyBodyRecon(weights_dir)
    return {"lhm": LHMRecon, "mhr": MHRRecon}[name](weights_dir)


def build_views(
    name: str, weights_dir: Path, engine: str = "mock", backend: str = "local"
) -> ViewModel:
    _check_engine(engine)
    if engine == "mock":
        return MockViews()
    if engine == "proxy":
        # proxy 几何 + 生成式视图：没有 LHM 时的最佳组合
        return DashscopeViews(weights_dir) if backend == "api" else SoftwareViews()
    if backend == "api":
        return DashscopeViews(weights_dir)
    return {"lhm": LHMRender}[name](weights_dir)


def build_edit(
    name: str,
    weights_dir: Path,
    engine: str = "mock",
    backend: str = "local",
    api_key: str | None = None,
) -> EditModel:
    _check_engine(engine)
    if engine == "mock":
        return MockEdit()
    if engine == "proxy":
        # proxy 档默认基线补全（零依赖、确定性）；--edit-backend api 且配置了
        # key 时升级为真 mask 托管补全——实测已观测区 MAE 3.95/255，约束守得住，
        # 但细节里偶有小伪影，所以仍以基线为默认。
        # 没有 key 时静默降级：proxy 档的承诺是「无需任何 key 纯 CPU 可跑」。
        if backend == "api" and (api_key or os.environ.get("DASHSCOPE_API_KEY")):
            return DashscopeWanxEdit(weights_dir, api_key=api_key)
        return BaselineInpaint()
    if backend == "api":
        # 默认用真 mask 后端：对话式端点（DashscopeEdit）会整图重绘，掩码约束守不住
        return DashscopeWanxEdit(weights_dir, api_key=api_key)
    return {"qwen_image_edit": QwenImageEdit}[name](weights_dir)


def build_vton(
    name: str,
    weights_dir: Path,
    engine: str = "mock",
    backend: str = "api",
    api_key: str | None = None,
) -> TryOnModel:
    _check_engine(engine)
    if engine == "mock":
        return MockTryOn()
    if name != "fashn_vton":
        raise KeyError(name)
    return FashnVton(weights_dir, backend=backend, api_key=api_key)


def build_hair(name: str, weights_dir: Path, engine: str = "mock") -> HairModel:
    _check_engine(engine)
    if engine == "mock":
        return MockHair()
    return {"hairfastgan": HairFastGan}[name](weights_dir)


def build_rig(name: str, weights_dir: Path, engine: str = "mock") -> RigModel:
    _check_engine(engine)
    if engine == "mock":
        return MockRig()
    # proxy 档没有 CPU 绑定实现：如实返回 not_wired，由编排器软失败
    return {"unirig": UniRig}[name](weights_dir)


__all__ = [
    "COMMERCIAL_OK",
    "ENGINES",
    "OPTIONAL_RESTRICTED",
    "build_edit",
    "build_hair",
    "build_recon",
    "build_rig",
    "build_vton",
    "build_views",
]
