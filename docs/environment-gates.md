# 环境门槛：为什么这台机器现在只能跑 proxy 档

代码是通的，卡住的是运行环境、账号权限与权重体积。这份文档把「哪一档能跑、
缺什么、为什么」写成可复核的实测记录，避免下一个人重复踩。

所有数字来自 2026-10-30 在本机（Windows / Python 3.14.5 / RTX 4060 Laptop 8 GB）
的实测。复核命令：`python scripts/doctor.py`。

## 一、运行时代差：LHM 的依赖上限是 Python 3.12

`real` 档的后端是 [LHM](https://github.com/aigc3d/LHM)。它的 `requirements.txt`
是一组强钉版本，其中几项在 Python 3.13/3.14 上**没有轮子**：

| 依赖 | LHM 钉的版本 | cp314 轮子 | 后果 |
|---|---|---|---|
| `torch` | `==2.3.0` | 只有 2.15.0 有 | 装得上 2.15 但不满足钉死版本 |
| `numpy` | `==1.23.0` | 无（1.x 全系不支持 3.14） | 源码编译必失败 |
| `chumpy` | 无钉版本 | 无 | 该包早已停止维护，`inspect.getargspec` 在新版本已移除 |
| `decord` | `==0.6.0` | 无 Windows 轮子 | 只能编译 |
| `xatlas` / `gsplat` | `0.0.9` / `1.4.0` | 无 | `gsplat` 需 CUDA toolkit + MSVC 现场编译 |

结论：**这不是网络问题，是运行时代差**。PyPI 上确实有 `torch-2.15.0-cp314`
（126.7 MB，本机实测 2.8 MB/s 下载无阻），但 LHM 要的是 2.3.0 那套 ABI，
换不过去。`real` 档必须用独立的 Python 3.12 环境：

```bash
py -3.12 -m venv .venv-lhm
.venv-lhm\Scripts\activate
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r <LHM 仓库>/requirements.txt     # 先装官方 requirements，再处理 gsplat
```

`gsplat` 与 `basicsr` 在 Windows 上需要 CUDA toolkit 与 MSVC 现场编译，这是整条
链路上最容易失败的一步。**预算按半天到一天算，不要在演示前夜做这件事。**

## 二、账号门槛：试穿模型存在，但没给这个账号开通

阿里云 Model Studio 的 AI 试衣家族是 `aitryon` / `aitryon-plus`（试衣）与
`aitryon-parsing-v1`（服饰分割）。用现有 key 探测，返回的是：

```
HTTP 404  {"code":"InvalidParameter","message":"Model not exist."}
```

单看这句话会误判成「模型名写错了」。做一次对照实验就能区分：

| 探测对象 | 状态码 | 文案 | 结论 |
|---|---|---|---|
| `aitryon`（真实模型） | **404** | Model not exist. | 名字有效，**账号未开通** |
| `aitryon-parsing-v1`（真实模型） | **404** | Model not exist. | 同上（同族，不同端点） |
| `totally-bogus-model-xyz`（虚构） | **400** | Model not exist. | 名字无效 |

**文案相同、状态码不同**——这就是判据。`classify_model_probe()` 按状态码 + 文案
联合判定，`tests/test_doctor.py` 把这个区别钉住了。

对照之下，同一个 key 探测 `wanx2.1-imageedit`（纹理编辑）返回的是参数校验报错，
说明模型名被接受、模型可用——这也是 `--edit-backend api` 能真跑通的原因。

两条出路：
1. 去 Model Studio 控制台开通「AI试衣」，然后 `--vton-backend api` 直接可用；
2. 换 [FASHN](https://github.com/fashn-AI/fashn-vton-1.5) 托管接口（`FASHN_API_KEY`），
   端点实测可达（HTTP 401，即缺鉴权而非不可达）。

## 三、权重门槛：18.8 GB，且 HF 仓库是 gated

| 来源 | 实测 | 说明 |
|---|---|---|
| 阿里云 OSS `LHM_prior_model.tar` | HEAD 200，**18.8 GB** | 官方脚本里的下载地址，量大但直连快 |
| HF `3DAIGC/LHM-500M-HF` | 模型页 200，**tree 接口 403** | gated 仓库：需 HF token 并接受条款 |
| `hf-mirror.com` 同仓库 | 同上 | 镜像不绕过 gated，仍需 token |

权重不进仓库（`.gitignore` 已排除 `weights/`），`scripts/fetch_weights.py` 会在下载后
写 `MANIFEST.json` 钉住哈希，避免上游静默更新导致结果漂移。

## 四、网络：会话之间会变，别把某次的结论当常态

同一台机器，两次会话的实测结果相反：

| 目标 | 上次会话 | 本次 |
|---|---|---|
| `github.com` | 502 | 200 |
| `huggingface.co` | 502 | 200 |
| `hf-mirror.com` | 200 | 200 |
| Google Storage（MediaPipe `.task`） | 稳定截断在 8 MiB | 未复测 |
| PyPI | 正常 | 正常（126 MB 轮子 2.8 MB/s） |

教训：**「下载不到」要先分清是「网络不通」还是「环境不兼容」**。本次的 LHM 是后者，
而上一次把前者当成了主因。

## 五、所以现在能跑什么

| 档位 | 现状 | 缺什么 |
|---|---|---|
| `mock` | 可跑 | 无 |
| `proxy`（本地基线） | 可跑，34 → 46 个测试覆盖 | 无 |
| `proxy --edit-backend api` | 可跑（wanx 真掩码端点） | 已具备 |
| `proxy --views-backend api` | 可跑 | 已具备 |
| 试穿（aitryon / FASHN） | **不可跑** | 账号开通 或 `FASHN_API_KEY` |
| `real`（LHM 重建） | **不可跑** | Python 3.12 环境 + CUDA torch + 18.8 GB 权重 + `gsplat` 编译 |

把这张表和 `doctor` 的输出对齐，就不会再出现「代码明明写完了却跑不起来」的困惑。
