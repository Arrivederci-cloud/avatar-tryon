# avatar-tryon

一张全身照 → 系统补出三视图 → 可旋转的 3D 数字人 → 试穿 / 试发。

> 这是一个**技术研究与原型验证**项目，不是量体工具。三视图中的侧面与背面为模型生成结果，
> 不代表本人真实外观；请勿用于医疗、诊断或任何需要精确人体尺寸的场景。

## 它解决什么问题

用户只上传一张全身照，系统自动生成正面、侧面、背面三视图供其确认，再据此构建可换装、
可换发型的 3D 形象。相比「让用户自己拍三视图」，入口门槛低了一个量级；
代价是系统必须为不可见的信息负责。

## 核心设计判断

**几何来自参数化先验，纹理才来自生成模型。**

把三视图整张交给扩散模型去「想」，正面和背面会各自脑补出一张脸（Janus 问题），
三张图的肩宽腰线也互相对不上，后续重建必然失败。所以本项目：

1. 用 MHR / SMPL-X 这类参数化模型确定几何，得到与视角无关的骨架与体尺；
2. 正面纹理由原图反投影，几乎无损；
3. 侧面与背面的缺失纹理**在 UV 空间补全**，而不是在新视角上生成。

UV 空间只有一个，天然不存在「多视角各自生成」的可能，Janus 问题被结构性地规避掉了。

```
单图 + 身高
    │
    ├─ 质检与尺度标定 ── 像素 → 厘米
    │
    ├─ 几何先验（MHR）── 唯一几何锚
    │
    ├─ 三视图 ── 几何投影 + 正面反投影 + UV 纹理补全
    │
    ├─ 用户确认 ── 可重绘、可重传
    │
    └─ 装配 ── 版型形变目标 / 发片资产 / 服装试穿
```

## 快速开始

核心路径**零第三方依赖**，clone 下来即可跑通（mock 模式不加载任何模型）：

```bash
git clone https://github.com/<your-name>/avatar-tryon.git
cd avatar-tryon

PYTHONPATH=src python -m avatar_tryon.cli --image photo.jpg --height 170 --mock
```

输出 `<out>/result.json`，包含每一阶段的状态、产物路径与告警。

### 先体检：一条命令看清哪一档能跑

```bash
PYTHONPATH=src python scripts/doctor.py          # 只读、不产生费用
PYTHONPATH=src python scripts/doctor.py --offline # 不联网，只看本地
```

它逐条报「现状 / 影响哪一档 / 怎么补」：Python 版本、显卡、依赖、权重、密钥，
以及试穿与纹理编辑模型**是否对当前账号开通**。项目里最费时间的坑都在环境而非代码，
所以先把这张表跑出来再动手。

| 档位 | 现状 |
|---|---|
| `mock` / `proxy`（含 `--edit-backend api`） | 开箱可跑 |
| 试穿（aitryon / FASHN） | 需账号开通「AI试衣」或 `FASHN_API_KEY` |
| `real`（LHM 重建） | 需独立的 Python 3.12 环境 + CUDA torch + 18.8 GB 权重 |

原因、实测数字与复核方法见 **[docs/environment-gates.md](docs/environment-gates.md)**。

### 三档引擎

`--engine` 决定每一环节用哪套实现，**同一套接口，可整体替换**：

| engine | 几何 | 三视图 | 纹理补全 | 需要什么 | 用途 |
|---|---|---|---|---|---|
| `mock` | 占位 | 占位 | 占位 | 无 | CI、验证数据流 |
| `proxy` | 参数化体型（按身高/三围生成 OBJ，GrabCut 剪影分带拟合） | 软件渲染器（画家算法，纯 CPU）或生成视图 API | Voronoi 扩散补全 / 可选 wanx API | 无（API 后端需 key） | **无 GPU 环境跑通全链路**、降级方案 |
| `real` | LHM / MHR（待接入） | LHM（待接入） | Qwen-Image-Edit（待接入） | 权重 + 显卡 | 正式效果 |

`proxy` 档在普通笔记本上约 1–2 秒跑完（API 后端约 2 分钟）：单图 → GrabCut 人物抠图 →
参数化体型按剪影**逐带拟合**（网格搜索胸围缩放与手臂张角）→ UV 反投影 → 补全 → 三视图渲染。
**它是链路验证与 CI 夹具，不是产品效果**：几何是先验拟合结果而非真实重建，
侧面背面的纹理来自投影与外推/生成。

实测（256² 图集，诚实口径 fill_ratio，分母为整个图集）：
UV 覆盖率从早期实现的真实 **44.7%** 降到 **14.5%–23.0%**——数字变差不是因为退化，
而是修掉了度量口径、法线朝向与像素中心三处系统性错误后的真实值
（详见 [docs/pipeline.md](docs/pipeline.md) 的逐步修正记录）。

> 已知限制（当前真实状态）：
> - 手臂区域有条纹状错位——代理网格的手臂姿态与照片中真实手臂不一致，投影必然错位；
> - 脚踝发白——代理网格没有脚部几何；
> - wanx API 补全区有小块伪影（已知区域 MAE 3.95/255，mask 被正确遵守，但生成质量非零瑕疵）。
> 这就是需要 MHR/LHM 做真实重建的原因——proxy 档不替代它们。

```bash
PYTHONPATH=src python -m avatar_tryon.cli --image photo.jpg --height 170 --engine proxy
```

真实推理需要先拉权重（权重不入库，见 `docs/model-selection.md`）：

```bash
pip install -e ".[weights]"
python scripts/fetch_weights.py --model fashn_vton
PYTHONPATH=src python -m avatar_tryon.cli --image photo.jpg --height 170 \
    --garment tee_001 --garment-image tee.jpg --vton-backend local
```

### 试穿与纹理补全：两条后端

| 环节 | 后端 | 命令 | 适用 |
|---|---|---|---|
| 侧面/背面视图 | 生成 API | `--views-backend api`（需 `DASHSCOPE_API_KEY`） | 比软件渲染多一层真实纹理来源 |
| 服装试穿 | 托管接口 | `--vton-backend api`（需 `FASHN_API_KEY`） | 最快验证效果，按次计费 |
| 服装试穿 | 本地权重 | `--vton-backend local` | 私有化、批量、无按次成本 |
| 纹理补全 | 托管接口 | `--edit-backend api`（需 `DASHSCOPE_API_KEY`） | 已实测可用（见下），按量计费 |
| 纹理补全 | 本地权重 | `--edit-backend local` | 私有化、批量 |

完整 proxy 全链路示例（生成视图 + API 补全，实测 2 分钟跑完）：

```bash
PYTHONPATH=src python -m avatar_tryon.cli --image photo.jpg --height 168 \
    --engine proxy --views-backend api --edit-backend api --out outputs/demo
```

接口与权重是两套条款：**权重是 Apache-2.0，托管接口是商业服务**，别在文档里写混。
图片以 data URI 内联发送，不先落地到对象存储——人脸图片多传一站就多一份泄露面。

托管纹理补全走阿里云 Model Studio 的 `wanx2.1-imageedit`，`function` 用
`description_edit_with_mask`（**真掩码端点**，掩码内白区重绘、黑区逐像素不变）。
该端点要求输入最短边 ≥512，适配器先把图集 NEAREST 放大到 512、返回后 LANCZOS 缩回
（见 `models/generation_adapters.py`）。
早期结论「通用图像编辑模型无法在 UV 空间工作」是**错的**——错在用错了端点
（conversational 编辑不尊重 UV 掩码）。换成真掩码端点后实测：
已知区域 MAE 3.95/255（阈值 8），补全区方差 3668，掩码被正确遵守。
保留本地基线作为 proxy 默认，因为 API 输出仍带少量小块伪影。

本地后端需要先装上游包（`fashn_vton` 不在 PyPI，只能从仓库以可编辑模式安装）：

```bash
git clone https://github.com/fashn-AI/fashn-vton-1.5.git
pip install -e ./fashn-vton-1.5
python ./fashn-vton-1.5/scripts/download_weights.py --weights-dir ./weights/fashn_vton
```

权重约 2 GB（`model.safetensors` + `dwpose/`），人体解析器权重首次运行时自动下载到 HF 缓存。
Ampere 及以上显卡按 bf16 运行，其余自动转 fp32。

跑测试：

```bash
pip install -e ".[dev]"
pytest -q
```

## 模型清单

只有许可证允许商用与再分发的模型才进默认配置。

| 环节 | 模型 | 许可 | 说明 |
|---|---|---|---|
| 人体几何 | [MHR](https://github.com/facebookresearch/MHR) | Apache-2.0 | 骨架与体尺的几何基准 |
| 人体重建 | [LHM / LHM++](https://github.com/aigc3d/LHM) | Apache-2.0 | 单图秒级，原生 SMPL-X 驱动 |
| 纹理补全 | [Qwen-Image-Edit](https://huggingface.co/Qwen) / wanx2.1-imageedit | Apache-2.0 / 托管服务条款 | UV 空间 inpaint（`description_edit_with_mask` 真掩码端点已实测）；支持本地权重与 DashScope 托管两条后端 |
| 服装试穿 | [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5) | Apache-2.0 | 像素空间、无掩码 |
| 发型迁移 | [HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) | MIT | 代码与权重均为 MIT |
| 骨骼绑定 | [UniRig](https://github.com/VAST-AI-Research/UniRig) | MIT | 1–5 秒出骨架与蒙皮 |
| 渲染 | three.js r186 + Spark | MIT | 原生高斯泼溅 + LOD 流式 |

**不随仓库分发的模型**（受限许可，需自行接受条款）：SAM 3D Body（SAM License，非 OSI）、
MagicMan（未见明确许可）、CatVTON / IDM-VTON（CC BY-NC-SA，禁止商用）。
完整许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [docs/model-selection.md](docs/model-selection.md)。

## 目录结构

```
src/avatar_tryon/
├── camera.py        # 固定机位约定（三视图与渲染器共用）
├── models/          # 模型适配器：接口在这里，实现可替换
│   ├── base.py      # 抽象接口，上层只依赖它
│   ├── registry.py  # 按许可证分组、按引擎档位分派的注册表
│   ├── mock.py      # 零依赖占位实现，CI 只跑它
│   ├── proxy.py     # 参数化体型先验 + GrabCut 剪影分带拟合（无 GPU 的几何来源）
│   ├── software_render.py  # 软件渲染器（CPU 出三视图）
│   ├── generation_adapters.py  # DashScope wanx 真掩码编辑适配器（≥512 放缩）
│   └── baseline_edit.py    # Voronoi 最近源 + 扩散补全（生成模型的对照基线）
├── pipeline/
│   ├── quality_check.py  # 质检与尺度标定
│   ├── three_view.py     # 单图 → 三视图（本项目核心）
│   ├── garment.py        # 形变目标插值
│   └── orchestrator.py
├── utils/
│   ├── image.py     # 零依赖读图尺寸 / 写 PNG
│   ├── mesh.py      # OBJ 读写、三角化、按 UV 连通域定向法线
│   ├── silhouette.py # 人物掩码 → 分带宽度剖面（剪影对齐用）
│   ├── matte.py     # GrabCut 人物抠图 + 全图误判重试 + 边缘侵蚀
│   └── body_mesh.py # 参数化人体网格（身高 + 三围 + 手臂张角）
└── cli.py
```

## 三条设计取舍

**1. 运行时不做布料仿真。**
版型仿真在离线的 CLO / Style3D 里跑，产出若干尺码档位的形变目标，运行时只做插值。
实时仿真在浏览器里不可能，而且商业软件会让仓库不可复现。

**2. 尺码分档，不承诺围度。**
单张平面图无法观测围度——这是尺度歧义，不是算法不够好。想要围度级精度，
必须让用户补充围度或侧面照。产品上分两档承诺，比事后解释便宜得多。

**3. LLM 不在关键路径上。**
搭配建议是独立的后置步骤。生成 3D、试穿、试发全是视觉模型的事，
语言模型挂了主流程照常完成，走模板兜底。

## 许可

代码采用 [Apache-2.0](LICENSE)。第三方模型与权重各自保留原许可，见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

本项目的技术路线建立在上述开源工作之上。若用于研究，请引用对应论文。
