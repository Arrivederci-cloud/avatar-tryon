"""生成类适配器：试衣、试发、UV 纹理补全。

三个模型的共同点：许可证都允许商用与再分发（Apache-2.0 / MIT），
这是它们被选中的首要理由，效果是第二位的。
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import EditModel, HairModel, TryOnModel, TryOnResult
from .errors import ModelNotWired


def _mime_of(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def _data_uri(path: Path) -> str:
    """本地文件转 data URI。

    不先上传到对象存储，是因为人脸图片多传一站就多一份泄露面，
    而且托管接口本身就接受 data URI。
    """
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime_of(path)};base64,{payload}"


def _first_url(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list | tuple) and output:
        first = output[0]
        return first if isinstance(first, str) else str(first)
    raise RuntimeError(f"无法从 output 中取出结果地址：{output!r}")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())
    return dest


class FashnVton(TryOnModel):
    """2D 服装试穿。支持托管接口与本地权重两种后端。

    许可要分开表述，别写混：
    - 权重是 Apache-2.0，可自托管、可商用、可再分发；
    - 托管接口是商业服务，按其服务条款计费，不适用 Apache-2.0。

    backend="api"   调用官方托管接口，最快跑通，适合先验证效果；
    backend="local" 加载本地权重推理，适合私有化与批量成本控制。
    """

    name = "fashn-vton"
    license = "Apache-2.0 (weights) / 商业服务条款 (hosted API)"
    upstream = "https://github.com/fashn-AI/fashn-vton-1.5"

    # 官方 v1.6 的三个品类档位；不传则由服务端判断
    CATEGORIES = ("tops", "bottoms", "one-pieces")
    MODES = ("performance", "balanced", "quality")

    def __init__(
        self,
        weights_dir: Path,
        backend: str = "api",
        api_key: str | None = None,
        model_name: str = "tryon-v1.6",
        timeout_s: int = 180,
        poll_s: float = 3.0,
    ) -> None:
        self.weights_dir = weights_dir
        self.backend = backend
        self.api_key = api_key or os.environ.get("FASHN_API_KEY")
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._pipe: Any = None

    # ---------- 对外入口 ----------

    def apply(
        self,
        person: Path,
        garment: Path,
        garment_id: str = "demo",
        category: str = "tops",
        mode: str = "balanced",
        seed: int | None = None,
    ) -> TryOnResult:
        if category not in self.CATEGORIES:
            raise ValueError(f"category 必须是 {self.CATEGORIES} 之一，收到 {category!r}")
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES} 之一，收到 {mode!r}")
        if self.backend == "api":
            return self._apply_api(person, garment, garment_id, category, mode, seed)
        return self._apply_local(person, garment, garment_id, category)

    # ---------- 托管接口 ----------

    def _apply_api(
        self,
        person: Path,
        garment: Path,
        garment_id: str,
        category: str,
        mode: str,
        seed: int | None,
    ) -> TryOnResult:
        if not self.api_key:
            raise ModelNotWired(
                "缺少 FASHN_API_KEY。二选一：\n"
                "  1) 在 .env 里填 FASHN_API_KEY=<你的密钥>，用托管接口快速验证；\n"
                "  2) 改用本地权重：--vton-backend local 并先跑 "
                "python scripts/fetch_weights.py --model fashn_vton"
            )
        try:
            from fashn import Fashn
        except ImportError as exc:
            raise ModelNotWired("缺少依赖：pip install -e '.[api]'") from exc

        client = Fashn(api_key=self.api_key)
        inputs: dict[str, Any] = {
            "model_image": _data_uri(person),
            "garment_image": _data_uri(garment),
            "category": category,
            "mode": mode,
            "num_samples": 1,
            "output_format": "png",
            "return_base64": False,
        }
        if seed is not None:
            inputs["seed"] = seed

        job = client.predictions.run(model_name=self.model_name, inputs=inputs)
        if getattr(job, "error", None):
            raise RuntimeError(f"FASHN 提交失败：{job.error}")

        deadline = time.monotonic() + self.timeout_s
        while True:
            st = client.predictions.status(job.id)
            status = str(getattr(st, "status", "")).lower()

            if status == "failed" or getattr(st, "error", None):
                raise RuntimeError(f"FASHN 推理失败：{st.error}")

            if status in {"completed", "succeeded", "success"} and getattr(st, "output", None):
                url = _first_url(st.output)
                out = person.parent / "tryon" / f"{garment_id}_{self.model_name}.png"
                _download(url, out)
                return TryOnResult(
                    image_path=out,
                    garment_id=garment_id,
                    notes=[f"backend=api model={self.model_name} category={category} mode={mode}"],
                )

            if time.monotonic() > deadline:
                raise TimeoutError(f"FASHN 推理超时（{self.timeout_s}s），任务 id={job.id}")

            time.sleep(self.poll_s)

    # ---------- 本地权重 ----------

    def _load_pipeline(self):
        """懒加载并缓存流水线。

        权重约 2 GB，每次调用都重新加载会让批量任务慢到不可用，
        所以首次构造后挂在实例上复用。
        """
        if getattr(self, "_pipe", None) is not None:
            return self._pipe

        weights = self.weights_dir / "fashn_vton"
        if not (weights / "model.safetensors").exists():
            raise ModelNotWired(
                f"未找到本地权重：{weights / 'model.safetensors'}\n"
                f"上游仓库：{self.upstream}\n"
                "步骤：\n"
                "  git clone https://github.com/fashn-AI/fashn-vton-1.5.git\n"
                "  pip install -e ./fashn-vton-1.5\n"
                "  python ./fashn-vton-1.5/scripts/download_weights.py --weights-dir "
                f"{weights}\n"
                "注意：fashn_vton 不在 PyPI 上，必须从仓库以可编辑模式安装。"
            )

        try:
            from fashn_vton import TryOnPipeline
        except ImportError as exc:
            raise ModelNotWired(
                "缺少本地推理包 fashn_vton。它未发布到 PyPI，需从上游仓库安装：\n"
                "  git clone https://github.com/fashn-AI/fashn-vton-1.5.git\n"
                "  pip install -e ./fashn-vton-1.5\n"
                "GPU 环境默认装 onnxruntime-gpu；纯 CPU 请改装 onnxruntime。"
            ) from exc

        self._pipe = TryOnPipeline(weights_dir=str(weights))
        return self._pipe

    def _apply_local(
        self, person: Path, garment: Path, garment_id: str, category: str
    ) -> TryOnResult:
        pipe = self._load_pipeline()

        try:
            from PIL import Image
        except ImportError as exc:
            raise ModelNotWired("缺少 Pillow：pip install pillow") from exc

        with Image.open(person) as p, Image.open(garment) as g:
            person_img = p.convert("RGB")
            garment_img = g.convert("RGB")
            result = pipe(
                person_image=person_img,
                garment_image=garment_img,
                category=category,
            )

        out = person.parent / "tryon" / f"{garment_id}_fashn_vton.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out)

        return TryOnResult(
            image_path=out,
            garment_id=garment_id,
            notes=[
                f"backend=local weights={self.weights_dir / 'fashn_vton'}",
                "本地 v1.5 不支持 mode/seed 参数，这两个是托管接口 v1.6 才有的",
            ],
        )



class HairFastGan(HairModel):
    """StyleGAN FS 隐空间，亚秒级。代码与权重均为 MIT。"""

    name = "hairfastgan"
    license = "MIT"
    upstream = "https://github.com/AIRI-Institute/HairFastGAN"

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir

    def transfer(self, person: Path, hairstyle: Path, color: str | None = None) -> Path:
        if not (self.weights_dir / "hairfastgan").exists():
            raise ModelNotWired(
                f"未找到 HairFastGAN 权重：{self.weights_dir / 'hairfastgan'}\n"
                f"先运行 python scripts/fetch_weights.py --model hairfastgan"
            )
        raise ModelNotWired(
            "接入要点：输入 (face, shape, color) 三张图；只换形状时 shape=hairstyle、color=person。"
            "注意人脸对齐由模型内部的 Pose alignment 完成，不要自己做裁剪，否则会二次变形。"
        )


class QwenImageEdit(EditModel):
    """UV 空间的纹理补全引擎。

    它替代了不可再分发的多视图扩散模型：我们不在「新视角」上生成，
    而是在「UV 贴图」上 inpaint。UV 空间只有一个，因此天然不存在
    多个视角各自长出一张脸的 Janus 问题。
    """

    name = "qwen-image-edit-2509"
    license = "Apache-2.0"
    upstream = "https://huggingface.co/Qwen"

    def __init__(self, weights_dir: Path) -> None:
        self.weights_dir = weights_dir

    def inpaint(self, image: Path, mask: Path, prompt: str) -> Path:
        if not (self.weights_dir / "qwen_image_edit").exists():
            raise ModelNotWired(
                f"未找到 Qwen-Image-Edit 权重：{self.weights_dir / 'qwen_image_edit'}\n"
                f"先运行 python scripts/fetch_weights.py --model qwen_image_edit"
            )
        raise ModelNotWired(
            "接入要点：mask 为白色待补区域；prompt 建议描述材质与光照而非内容，"
            "例如「接缝处的针织纹理，与周围一致的中性光照」。"
        )


# ---------------------------------------------------------------------------
# 托管 API 后端：DashScope 上的 Qwen-Image-Edit
# ---------------------------------------------------------------------------

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
# 实测：qwen-image-edit / -plus / -max 在公共 DashScope 端点返回 Model not exist，
# 当前可用的是 qwen-image-3.0-pro（2026-10 实测）。换账号/区域时用 --model 覆盖。
DASHSCOPE_EDIT_MODEL = "qwen-image-3.0-pro"


def _build_edit_payload(model: str, image_uri: str, mask_uri: str, prompt: str) -> dict:
    """构造请求体（纯函数，便于单测）。

    该端点没有显式 mask 参数：UV 纹理作图1、掩码作图2，在指令中约束
    「只重绘白色区域」。prompt_extend 关闭——我们的指令已经写全，重写会丢约束。
    """
    text = (
        "图1是一张 3D 人体的 UV 展开贴图，图2 是它的掩码（白色=缺失待补区域，黑色=已观测区域）。"
        f"请只重绘白色区域，内容为：{prompt}；"
        "黑色区域逐像素保持不变，不要改变贴图的整体布局与分辨率。"
    )
    return {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_uri},
                        {"image": mask_uri},
                        {"text": text},
                    ],
                }
            ]
        },
        "parameters": {"prompt_extend": False, "n": 1, "watermark": False},
    }


def _extract_image_url(body: dict) -> str:
    try:
        choices = body["output"]["choices"]
        for item in choices[0]["message"]["content"]:
            if "image" in item:
                return item["image"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelNotWired(f"DashScope 返回结构不符预期：{body!r:.200}") from exc
    raise ModelNotWired(f"DashScope 返回中没有图像：{body!r:.200}")


def _post_json(
    url: str, payload: dict, api_key: str, timeout: float, async_task: bool = False
) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if async_task:
        headers["X-DashScope-Async"] = "enable"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {}
        raise ModelNotWired(
            f"DashScope 请求失败 HTTP {exc.code}: {body.get('code', '?')} - {body.get('message', '')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ModelNotWired(f"无法连接 DashScope：{exc.reason}（检查网络与代理）") from exc
    if body.get("code"):
        raise ModelNotWired(
            f"DashScope 业务错误 {body['code']}: {body.get('message', '')}"
        )
    return body


class DashscopeEdit(EditModel):
    """UV 纹理补全的托管 API 后端（DashScope / Qwen-Image-Edit）。

    和 FASHN 一样，许可要分开表述：
    - 模型权重是 Apache-2.0；
    - 托管服务按量计费，受阿里云 Model Studio 服务条款约束。

    图片以 data URI 内联发送，不先上传到对象存储——人脸图片少传一站。
    """

    name = "qwen-image-edit-plus"
    license = "托管服务条款（阿里云 Model Studio）/ 权重 Apache-2.0"
    upstream = "https://help.aliyun.com/en/model-studio/qwen-image-edit-guide"

    def __init__(
        self,
        weights_dir: Path,
        api_key: str | None = None,
        model: str = DASHSCOPE_EDIT_MODEL,
        base_url: str = DASHSCOPE_BASE_URL,
        timeout: float = 300.0,
    ) -> None:
        self.weights_dir = weights_dir
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def inpaint(self, image: Path, mask: Path, prompt: str) -> Path:
        if not self.api_key:
            raise ModelNotWired(
                "未配置 DASHSCOPE_API_KEY。\n"
                "到 https://bailian.console.aliyun.com 申请，写入 .env 或环境变量。\n"
                "没有 key 时改用 --edit-backend local（需 Qwen-Image-Edit 权重）"
                "或 --engine proxy（基线补全）。"
            )
        url = f"{self.base_url}/services/aigc/multimodal-generation/generation"
        payload = _build_edit_payload(
            self.model, _data_uri(image), _data_uri(mask), prompt
        )
        body = _post_json(url, payload, self.api_key, self.timeout)
        image_url = _extract_image_url(body)

        out = image.parent / "inpaint" / f"{image.stem}_filled.png"
        _download(image_url, out)
        return out


# ---------------------------------------------------------------------------
# 真 mask 的局部重绘：wanx2.1-imageedit（异步任务）
# ---------------------------------------------------------------------------

DASHSCOPE_WANX_MODEL = "wanx2.1-imageedit"


def _build_wanx_payload(model: str, image_uri: str, mask_uri: str, prompt: str) -> dict:
    """构造 wanx 图像编辑请求体（纯函数，便于单测）。

    这个模型支持真正的掩码局部重绘：mask 白色区域被重绘，其余像素保留。
    这是 UV 补全的正确后端——对话式端点会整图重绘，掩码约束守不住。

    function 必须是 description_edit_with_mask：服务端只在该 function（和
    expand）下接受 mask_image_url，传别的会直接报
    「mask_image_url should not be provided」（实测）。
    """
    return {
        "model": model,
        "input": {
            "function": "description_edit_with_mask",
            "prompt": prompt,
            "base_image_url": image_uri,
            "mask_image_url": mask_uri,
        },
        "parameters": {"n": 1},
    }


def _task_id_of(body: dict) -> str:
    try:
        return body["output"]["task_id"]
    except (KeyError, TypeError) as exc:
        raise ModelNotWired(f"提交任务后未拿到 task_id：{body!r:.200}") from exc


def _poll_task(
    base_url: str, task_id: str, api_key: str, timeout: float, interval: float = 3.0
) -> dict:
    """轮询异步任务直到完成。"""
    deadline = time.time() + timeout
    url = f"{base_url}/tasks/{task_id}"
    while time.time() < deadline:
        time.sleep(interval)
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ModelNotWired(f"轮询任务失败：{exc}") from exc
        output = body.get("output", {})
        status = output.get("task_status")
        if status == "SUCCEEDED":
            return body
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            raise ModelNotWired(
                f"补全任务 {status}：{output.get('message') or output.get('code') or body}"
            )
    raise ModelNotWired(f"补全任务超时（{timeout:.0f}s），task_id={task_id}")


def _task_result_url(body: dict) -> str:
    try:
        return body["output"]["results"][0]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelNotWired(f"任务完成但没有图片地址：{body!r:.200}") from exc


WANX_MIN_SIDE = 512  # 服务端要求图像最短边 512~4096（实测报错文案）


def _upscale_for_api(image: Path, mask: Path):
    """UV 纹理是 256² 时放大到服务端最小尺寸，返回 (img, mask, 原始尺寸)。

    NEAREST 放大：保持掩码的硬边界与图集的色块边缘不被插值糊掉。
    没有 Pillow 时原样返回（由服务端报尺寸错误兜底）。
    """
    try:
        from PIL import Image
    except ImportError:
        return image, mask, None

    src = Image.open(image).convert("RGB")
    if min(src.size) >= WANX_MIN_SIDE:
        return image, mask, None
    scale = WANX_MIN_SIDE / min(src.size)
    size = (round(src.width * scale), round(src.height * scale))
    up_img = src.resize(size, Image.NEAREST)
    up_mask_img = Image.open(mask).convert("L").resize(size, Image.NEAREST)

    up_image = image.with_name(f"{image.stem}_up{size[1]}.png")
    up_mask = mask.with_name(f"{mask.stem}_up{size[1]}.png")
    up_img.save(up_image, "PNG")
    up_mask_img.save(up_mask, "PNG")
    return up_image, up_mask, src.size


def _resize_back(path: Path, size: tuple[int, int]) -> None:
    """把服务端返回的放大版结果缩回原始 UV 尺寸。"""
    try:
        from PIL import Image
    except ImportError:
        return
    img = Image.open(path).convert("RGB")
    if img.size != tuple(size):
        img.resize(tuple(size), Image.LANCZOS).save(path, "PNG")


class DashscopeWanxEdit(EditModel):
    """UV 纹理补全的真 mask 后端（DashScope wanx2.1-imageedit）。

    比对话式端点更适合本场景的原因：它接受显式 mask，
    白色区域重绘、其余像素保留，正面已观测纹理不会被模型改掉。

    许可同前：权重 Apache-2.0，托管服务按量计费。
    """

    name = "wanx2.1-imageedit"
    license = "托管服务条款（阿里云 Model Studio）/ 权重 Apache-2.0"
    upstream = "https://help.aliyun.com/zh/model-studio/image-edit-api-reference"

    def __init__(
        self,
        weights_dir: Path,
        api_key: str | None = None,
        model: str = DASHSCOPE_WANX_MODEL,
        base_url: str = DASHSCOPE_BASE_URL,
        timeout: float = 300.0,
        poll_interval: float = 3.0,
    ) -> None:
        self.weights_dir = weights_dir
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

    def inpaint(self, image: Path, mask: Path, prompt: str) -> Path:
        if not self.api_key:
            raise ModelNotWired(
                "未配置 DASHSCOPE_API_KEY。\n"
                "到 https://bailian.console.aliyun.com 申请，写入 .env 或环境变量。\n"
                "没有 key 时改用 --edit-backend local（需 Qwen-Image-Edit 权重）"
                "或 --engine proxy（基线补全）。"
            )
        url = f"{self.base_url}/services/aigc/image2image/image-synthesis"
        up_image, up_mask, original_size = _upscale_for_api(image, mask)
        payload = _build_wanx_payload(
            self.model, _data_uri(up_image), _data_uri(up_mask), prompt
        )
        submitted = _post_json(url, payload, self.api_key, self.timeout, async_task=True)
        task_id = _task_id_of(submitted)
        done = _poll_task(
            self.base_url, task_id, self.api_key, self.timeout, self.poll_interval
        )

        out = image.parent / "inpaint" / f"{image.stem}_filled.png"
        _download(_task_result_url(done), out)
        if original_size is not None:
            _resize_back(out, original_size)
        return out
