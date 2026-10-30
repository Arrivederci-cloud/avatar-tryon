"""生成式三视图：调用图像编辑 API 产出同一人物的侧/背视图。

为什么是「生成视图」而不是「对 UV 调补全」：
实测通用编辑模型不认识 UV 图集，会把贴图当成一件衣服的商品照整张重生成
（已观测区平均变化 58/255，两个模型都一样）。而「给一张正面全身照，
生成同一个人物的侧/背全身照」是它们最擅长的任务——身份、服装、光照都能靠
提示词锁住，我们拿到的是一张照片级的视图，再由几何投影进 UV。

自由度被限制在「生成一张合理的视图」上，UV 的全局一致性由几何保证，
所以不会出现 Janus（三张图各自长一张脸）的问题。
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import HumanGeometry, ThreeViews, ViewModel
from .errors import ModelNotWired
from .generation_adapters import (
    DASHSCOPE_BASE_URL,
    _data_uri,
    _download,
    _extract_image_url,
    _post_json,
)

VIEW_PROMPTS = {
    "side": (
        "把这张全身照转成同一个人物的正侧面全身照（从人物右侧看）："
        "保持完全相同的发型、肤色、服装款式与颜色、鞋履，"
        "保持相同的身材比例与姿态（直立、双臂自然下垂），"
        "同样的纯色摄影棚背景与中性光照，从头到脚完整入画、构图居中"
    ),
    "back": (
        "把这张全身照转成同一个人物的正背面全身照："
        "保持完全相同的发型（从背后的轮廓）、肤色、服装背面款式与颜色、鞋履，"
        "保持相同的身材比例与姿态（直立、双臂自然下垂），"
        "同样的纯色摄影棚背景与中性光照，从头到脚完整入画、构图居中，不要出现面部"
    ),
}


class DashscopeViews(ViewModel):
    """托管 API 生成侧/背视图（DashScope Qwen-Image 编辑）。

    许可同前：权重 Apache-2.0，托管服务按量计费（每张视图一次调用）。
    """

    name = "qwen-image-3.0-pro"
    license = "托管服务条款（阿里云 Model Studio）/ 权重 Apache-2.0"
    upstream = "https://help.aliyun.com/en/model-studio/qwen-image-edit-guide"
    # 视图由 API 直接产出，不需要先有 UV 纹理
    requires_texture = False

    def __init__(
        self,
        weights_dir: Path,
        api_key: str | None = None,
        model: str = "qwen-image-3.0-pro",
        base_url: str = DASHSCOPE_BASE_URL,
        timeout: float = 300.0,
    ) -> None:
        self.weights_dir = weights_dir
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def render_views(self, image: Path, geometry: HumanGeometry) -> ThreeViews:
        if not self.api_key:
            raise ModelNotWired(
                "未配置 DASHSCOPE_API_KEY，无法生成侧/背视图。\n"
                "到 https://bailian.console.aliyun.com 申请并写入 .env；\n"
                "或改用 --views-backend local（需 LHM 权重）或 --engine proxy。"
            )
        out_dir = image.parent / "views"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {"front": image}
        for name, prompt in VIEW_PROMPTS.items():
            paths[name] = self._generate(image, prompt, out_dir / f"{name}.png")

        return ThreeViews(
            front=paths["front"],
            side=paths["side"],
            back=paths["back"],
            confidence={"front": 1.0, "side": 0.6, "back": 0.45},
            notes=[
                "side/back 由图像编辑模型基于正面照生成，不是观测结果",
                "身份与服装靠提示词约束；透视与比例仍可能与正面照不一致，UI 需标注",
            ],
        )

    def _generate(self, image: Path, prompt: str, dest: Path) -> Path:
        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"image": _data_uri(image)}, {"text": prompt}],
                    }
                ]
            },
            "parameters": {"n": 1, "prompt_extend": False, "watermark": False},
        }
        url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        body = _post_json(url, payload, self.api_key or "", self.timeout)
        _download(_extract_image_url(body), dest)
        return dest


__all__ = ["VIEW_PROMPTS", "DashscopeViews"]
