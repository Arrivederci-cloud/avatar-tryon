# 流水线说明

## 阶段与契约

| 阶段 | 输入 | 输出 | 失败影响 |
|---|---|---|---|
| 质检与标定 | 图片、身高 | `QualityReport`（尺度 cm/px、告警） | 硬失败，终止 |
| 几何 | 图片、身高 | `HumanGeometry`（网格、体尺、关节） | 硬失败，终止 |
| 三视图 | 图片、几何 | `ThreeViews`（三张图 + 置信度） | 硬失败，终止 |
| 绑定 | 网格 | `Avatar`（带蒙皮） | 软失败，告警继续 |
| 试衣 | 三视图正面、单品 | `TryOnResult` | 软失败，告警继续 |
| 试发 | 三视图正面、参考发型 | 图片路径 | 软失败，告警继续 |

「硬失败」会立即返回并写出 `result.json`，不会静默吞掉；「软失败」只记告警，
因为它们是增强能力，缺失不应阻断主线。

## 为什么三视图是硬失败点

三视图是后面所有步骤的输入。如果它错了，绑定和试穿全错，而且错得很自然——
看不出问题，只是结果不像本人。所以这里必须停下来让用户确认。

工程上体现为：`three_view.build()` 之后应当有一个阻塞点（当前由调用方决定，
Web 端是一个确认页），确认通过才继续。

## 接入一个新模型

1. 在 `src/avatar_tryon/models/` 下新建适配器，继承 `base.py` 里对应的抽象类；
2. 在 `registry.py` 里注册，并**先判断它该进 `COMMERCIAL_OK` 还是 `OPTIONAL_RESTRICTED`**；
3. 在 `THIRD_PARTY_NOTICES.md` 里补一条声明；
4. 加一个 mock 实现，保证 CI 仍然零依赖可跑。

许可证判断先于一切：进不了 `COMMERCIAL_OK` 的模型，不要写进默认配置，
否则会让整个仓库变成不可分发。

## 关键接口

```python
class ReconModel(ABC):
    def reconstruct(self, image: Path, height_cm: float) -> HumanGeometry: ...

class ViewModel(ABC):
    def render_views(self, image: Path, geometry: HumanGeometry) -> ThreeViews: ...

class EditModel(ABC):
    def inpaint(self, image: Path, mask: Path, prompt: str) -> Path: ...

class TryOnModel(ABC):
    def apply(self, person: Path, garment: Path, garment_id: str) -> TryOnResult: ...

class HairModel(ABC):
    def transfer(self, person: Path, hairstyle: Path, color: str | None) -> Path: ...

class RigModel(ABC):
    def rig(self, mesh_path: Path) -> Avatar: ...
```

## 实施要点（容易踩的坑）

**UV 补全优于新视角生成。**
别去调多视图扩散模型生成侧背视角。UV 空间只有一个，不存在各视角各自长一张脸的问题；
需要生成的面积也小得多（正面反投影几乎无损），稳定性和成本都更好。

> **实测更正（2026-10-29，2026-10-30 修订）**：
> **对话式**编辑端点在 UV 空间**行不通**——`qwen-image-3.0-pro` 把 UV 图集
> 理解成一件 T 恤的商品照，整张重生成，已观测区域平均变化 58/255。
> 但 **wanx2.1-imageedit 的真 mask function 可以用**：`function` 必须是
> `description_edit_with_mask`（服务端只在该 function 和 `expand` 下接受
> mask_image_url，传 `description_change_with_mask` 会直接报错），配合
> 图像最短边 ≥512 的要求。实测窄缝掩码下已观测区平均变化 **3.95/255**、
> 待补区方差 3668（真实纹理非糊）——掩码约束守得住。
> 结论修订为：**确定性投影烘焙仍是颜色主来源**（成本 0、无伪影、可复现），
> 真 mask 托管补全（`--edit-backend api`）只兜底窄缝，且细节里偶有小伪影，
> 所以 proxy 档默认仍用基线补全。图像 ≤256² 时由适配器自动放大/回缩。

**正确的生成式路径：先生成视图，再几何投影烘焙。**
模型擅长的、且我们验证过有效的是「同一个人物的侧/背全身照」（给足身份约束：
发型、服装、姿态、背景、光照都写进提示词）。拿到三张视图后，颜色进入 UV 靠的是
确定性的几何投影：

```
三视图（front/side/back）
    │  每张图 GrabCut 抠图 → 逐带宽度剖面（utils/silhouette.py）
    │  网格逐带半宽  →  照片逐带半宽   （逐带仿射映射，不是单一线性映射）
    │  每个 UV texel 只采信「法线与该机位夹角小于掠射阈值」的视图
    │  多张都可见时取入射角最小（最正对）的那张，接缝不重影
    ▼
UV 图集 → 残留窄缝用基线/生成模型兜底
```

这样模型的自由度被限制在「生成一张合理的侧视图」上，UV 的全局一致性由几何保证，
Janus 问题同样不会出现。实现见 `pipeline/three_view.py: _backproject_views`。

**逐带对齐为什么必须（实测 44.7% → 23.5% 残留）。**
烘焙的横向映射若只用「网格总宽度 → 人物包围盒」一个线性映射，网格比真人宽的部位
（标准 A-pose 先验的手臂末端到中线 27cm，垂手站立真人只有 ~20cm）会落到人物轮廓
外的背景上，被抠图判定丢弃——这些 texel 全部变成待补区。逐带仿射映射把
「网格在高度 y 的横向范围」映射到「照片在高度 y 的横向范围」，贴图不再越界。

**法线定向必须按部件（UV 连通域），不能用整网格质心。**
手臂、腿是偏心部件：按整网格质心定向会把左臂全部法线掰成 -x、右臂全 +x，
侧视机位下一侧手臂整体被判为背向、完全不参与烘焙（实测单侧残留 97%，
另一侧 15%）。`utils/mesh.py: triangles()` 现在按 UV 连通域（=身体部件）
取质心定向，域质心落在部件轴线上，左右臂可见面完全对称（41.7% / 41.7%）。

**fill_ratio 的分母必须是整个 UV 图集。**
若用「本组视图可见三角形的覆盖」做分母，视图越少分母越小，fill_ratio 虚低
（单视图时后半个身体全空也不计入）。`_backproject_views` 先用全部三角形
光栅化一次得到图集面积，再做分母。

**采样避开轮廓边缘 1px。**
抠图边缘混有背景的抗锯齿色，直接采进 UV 会在模型边缘形成灰边。
背景判定用收缩 1px 后的掩码（`matte.erode`），剖面统计仍用原始掩码。

**GrabCut 的 rect 失败要重试。**
上游包围盒估计失败时会把「整张图」当 rect 传给 GrabCut；没有明确背景可供对比，
它会把整张图都判成前景（表现为前景 bbox 盖满 rect，实测 96.1% 高度 × 98.5% 宽度）。
`matte.person_matte` 现在检测到这种情况会用默认边框 rect 重试一次——
不修的话背景灰会被当成人烘进 UV，正面纹理半边发灰。

**UV 光栅化必须用像素中心。**
图集矩形 u0=0.02 对应像素 5.12，整数栅格化从 x=6 开始，留下 1px 纯白；
渲染器恰好在这个 u 上采样，正面出现一条贯穿的白色竖缝。
`_raster_uv_core` 改用像素中心（x+0.5）后这 1px 被真正覆盖；
掩码也对覆盖区做 1px 膨胀兜底。

**代理几何要先拟合到照片轮廓。**
`models/proxy.py: _fit_to_silhouette` 用 GrabCut 轮廓的逐带半宽对
（围度缩放, 手臂外展）做网格搜索（尺度按「人物像素高 / 网格身高跨度」定，
跳过顶部 15% 头发区与底部 6% 鞋区），实测手臂外展 ×0.70、残留再降 6 个百分点。
这一步不需要姿态关键点，也不需要下载任何权重。

`EditModel` 因此只负责「残留窄缝」的兜底，不承担主体纹理：

| 实现 | 场景 | 说明 |
|---|---|---|
| `BaselineInpaint` | `--engine proxy`（默认） | 最近源 Voronoi 初始化 + 局部扩散，纯 CPU、零依赖 |
| `DashscopeWanxEdit` | `--edit-backend api`（任一 engine） | 真 mask 托管补全，实测已观测区 MAE 3.95/255 |
| `QwenImageEdit` | `--engine real --edit-backend local` | 本地权重（待接入） |

**已知限制：背景剔除目前是颜色启发式 + GrabCut。**
GrabCut（`utils/matte.py`，OpenCV pip 包，无需下载权重）对渐变背景已经稳健；
包围盒估计仍是颜色启发式，只做粗定位。正式方案应换成分割/抠图模型
（SAM、BiRefNet、rembg），否则投影会带进背景色。

**已知限制：代理几何没有头发与脚。**
顶部 15% 的轮廓差异来自头发，网格无法表达；拟合与烘焙都把这一段排除或容忍。
正式方案需要接重建模型（LHM/MHR）或单独的头发网格。

**已知限制：本机网络挡住了重建模型。**
`huggingface.co` 与 `github.com` 在本机均 502（hf-mirror 与各 OSS 可达），
LHM 的代码只在 GitHub，无法克隆。MediaPipe 姿态模型的 `.task` 文件
（storage.googleapis.com）单连接被代理掐在 8 MiB，Range 续传无效
（9.4 MB 差 1 MB 永远下不完）。所以当前的几何修正路线是
「GrabCut 轮廓 + 参数化拟合」，完全不依赖需要下载的模型。

**基线补全 = 最近源初始化 + 局部扩散，不是均值、不是整行抹平。**
待补 texel 先取二维上最近的已知像素颜色（Voronoi，`cv2.distanceTransformWithLabels`），
再做 80 次 Jacobi 迭代把块状边界磨成平滑过渡。历史教训：整行轴向填充会在整行
缺色时把整行抹平（手臂/腿图集成片灰条的来源）；均值填充会把蓝衬衫糊成灰蓝。
已知像素一个字节都不动（`Image.composite` 的参数顺序曾写反，把柔化涂到了已知区）。

**尺度必须来自用户输入的身高。**
单图有尺度歧义，任何从照片直接推断绝对尺寸的做法都不可靠。
`quality_check` 会先算 `scale_cm_per_px`，后续所有长度都乘这个系数。

**UniRig 的输出需要一次规范化。**
它给的骨骼原点在重心、且没有语义命名。接 three.js 之前要把 hips 移到脚底、
按 Mixamo 命名重写骨骼，否则动画重定向会错位。

**形变目标要按档位预烘焙。**
不要试图实时仿真。每款衣服在 3-5 个体型档位上离线仿真一次，
运行时按实际体尺线性插值——`garment.py` 已经实现了插值与越界夹取。

## 评测指标

| 指标 | 方法 | 达标线 |
|---|---|---|
| 三视图一致性 | 肩宽、腰线、臀宽在三视图间的相对偏差 | ≤ 3% |
| 三视图认可率 | 用户判断「这是我」的比例 | ≥ 80% |
| 身份保真度 | 人脸嵌入余弦相似度 | ≥ 0.75 |
| 尺码分档准确率 | 推荐 S/M/L 与实测一致比例 | ≥ 85% |
| 试穿主观质量 | 双盲 5 分制 | ≥ 3.5 |

围度误差只在用户补充了围度输入时统计。单图条件下不统计，因为不可观测。
