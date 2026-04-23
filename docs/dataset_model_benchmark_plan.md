# 多数据集压缩 Benchmark 数据处理方案

## 1. 目标

本文档定义九个数据集在七类压缩方法下应该如何准备和处理数据。

- 数据集：Kodak、UVG、显微数据、遥感数据、晶体衍射、ERA5、Nyx、JHTDB、Hurricane。
- 压缩方法：cuSZ-Hi、nvJPEG、General 图像压缩模型、DCMVC、CAESAR-V、CAESAR-D、CRA5。

核心原则是：不要把所有数据都强行转成 RGB 图像。每个压缩方法应该使用它最自然的数据入口；如果数据经过归一化、切片、分组或渲染，必须在结果里明确说明。对于科学数据，指标应尽量在反归一化后的原始物理尺度上计算。

## 2. 数据集分类

| 类别 | 数据集 | 原始结构 | 主要风险 |
|---|---|---|---|
| 自然图像 | Kodak | RGB 图像，通常是 uint8 | 基本直接可测 |
| 视频 | UVG | RGB/YUV 帧序列 | 图像模型逐帧压缩时不会利用时间冗余 |
| 科学图像 | 显微、遥感、晶体衍射 | 2D/3D，高 bit depth，常见多通道 | 转成 8-bit 会丢失强度或物理意义 |
| 科学场数据 | ERA5、Nyx、JHTDB、Hurricane | 多变量 3D/4D float 场 | 容易破坏变量尺度、维度顺序和时间结构 |

## 3. 统一中间表示

建议显式生成这些中间格式，不要把转换逻辑隐藏在模型脚本内部。

| 表示 | 形状 | 适用方法 |
|---|---|---|
| 图像张量 | `[C, H, W]`，float32，范围 `[0, 1]` | General 图像模型 |
| 视频张量 | `[T, C, H, W]`，float32，范围 `[0, 1]` | General 图像模型逐帧压缩，CAESAR-D 重排后使用 |
| PNG 帧序列 | `sequence/im00001.png, im00002.png, ...`，RGB uint8 | DCMVC |
| CAESAR 张量 | `[V, S, T, H, W]` | CAESAR-V、CAESAR-D |
| 科学 3D block | `[C, D, H, W]` 或 `[D, H, W]` | cuSZ-Hi、3D 科学压缩 baseline |
| Raw binary | 连续 `.data` 文件，额外记录 dtype 和 shape | cuSZ-Hi |
| ERA5 全量场 | `[1, 268, 721, 1440]` | CRA5 |

每个 prepared sample 都应有 metadata，例如：

```json
{
  "dataset": "era5",
  "variable": "t_850",
  "dtype": "float32",
  "original_shape": [721, 1440],
  "prepared_shape": [1, 721, 1440],
  "normalization": "mean_std",
  "min": null,
  "max": null,
  "mean": 273.1,
  "std": 12.4,
  "unit": "K",
  "dimension_order": "lat_lon"
}
```

## 4. 方法和数据集兼容性

兼容等级：

- A：原生适合，推荐作为主实验。
- B：可以合理适配，适合作为补充实验。
- C：只能作为 slice、frame 或 visualization baseline。
- D：不推荐，除非重训或大幅改模型。

| 方法 | Kodak | UVG | 显微 | 遥感 | 晶体衍射 | ERA5 | Nyx | JHTDB | Hurricane |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuSZ-Hi | C | C | A | A | A | A | A | A | A |
| nvJPEG | A | B | C | C | C | D | D | D | D |
| General 图像模型 | A | B | B | B | B | C | C | C | C |
| DCMVC | D | A | C | C | C | C | C | B | B |
| CAESAR-V | C | B | B | B | B | A | B | A | A |
| CAESAR-D | D | A | B | B | B | A | B | A | A |
| CRA5 | D | D | D | D | D | A | D | D | D |

## 5. cuSZ-Hi 数据处理

cuSZ-Hi 应该作为科学数组压缩器使用。它的输入是 raw binary 数组，加上 dtype 和维度参数。

典型命令：

```bash
./cuszhi \
  --report time,cr \
  -z \
  -t f32 \
  -m r2r \
  --dim3 DimXxDimYxDimZ \
  -e REL_ERROR_BOUND \
  --predictor spline3 \
  -i input.data \
  -s cr
```

关键规则：

- 优先保留原始物理值，不做 `[0, 1]` 图像归一化。
- 将 NetCDF、HDF5、TIFF 或 NumPy 数组导出成连续 raw binary。
- 必须记录 dtype、shape、维度顺序、变量名、单位和 error bound。
- 2D 数据如果必须走 `--dim3`，可以使用退化的第三维。
- `DimX` 是最快维度，导出和读回时维度顺序必须一致。

### cuSZ-Hi x 数据集处理

| 数据集 | 处理方式 |
|---|---|
| Kodak | 不作为主实验。可把 RGB `uint8` 转成 `float32` raw，形状为 `H x W x 3`，仅用于 sanity check。可报 PSNR 和压缩比，但这不是 cuSZ-Hi 的自然使用场景。 |
| UVG | 不作为视频 codec。可做两个 baseline：逐帧压缩 `H x W x 3` raw，或把短序列的单通道/RGB 通道组织成 `T x H x W` 3D 场压缩。结果中要说明它没有像视频 codec 那样做时序预测。 |
| 显微 | 推荐。保留 `uint16` 或 `float32` 强度值。2D 图像用 `H x W x 1`；z-stack 用 `Z x H x W`；多通道数据可以逐通道压缩，或组织成 `C x H x W`。 |
| 遥感 | 推荐。每个 band 单独导出 raw，或把多个 band 组织成 `B x H x W`。超大 GeoTIFF 先切固定 patch，CRS/transform 等地理元数据单独保存。 |
| 晶体衍射 | 推荐。把原始强度作为 `float32` raw 压缩。若动态范围太大，可以额外做一组 `log1p` 实验，但要和原始强度实验分开报告。重建后评估 peak 位置和 peak 相对强度误差。 |
| ERA5 | 强烈推荐。每个变量/level/time field 导出 raw，或把多个 level 打包成 `L x H x W`。使用原始物理尺度，计算 per-variable RMSE、NRMSE、max error 和压缩比。 |
| Nyx | 强烈推荐。density、temperature、velocity 等 3D 场按 `D x H x W` block 压缩。block 大小要适配 GPU 显存。density 这类高动态范围变量可考虑 `log1p` 版本。 |
| JHTDB | 强烈推荐。`u`、`v`、`w` 可分量单独压缩，也可作为 3 通道场处理。3D 湍流 block 需保留物理尺度，并尽量报告 velocity RMSE、频谱误差和 vorticity error。 |
| Hurricane | 强烈推荐。每个变量、time、level 可作为 2D 或 3D raw block 压缩。保留变量和 level metadata。报告 pressure、wind、temperature 的 RMSE，以及可用的风暴相关指标。 |

## 6. nvJPEG 数据处理

nvJPEG 是 GPU JPEG 编解码器，适合 uint8 图像和视频帧，不适合直接压缩科学 float 场。

期望输入：

- `uint8`
- 灰度或 RGB
- nvJPEG wrapper 支持的 planar 或 interleaved layout

关键规则：

- nvJPEG 应作为图像或可视化 baseline。
- 不要把它作为 ERA5、Nyx、JHTDB、Hurricane 的主科学压缩器。
- 如果科学数据被映射到 uint8，那么重建科学值的指标只对这条明确的 visualization/quantization pipeline 有意义。

### nvJPEG x 数据集处理

| 数据集 | 处理方式 |
|---|---|
| Kodak | 原生适合。直接输入 RGB uint8 图像。报告 compressed size、bpp、PSNR、MS-SSIM、encode/decode time。 |
| UVG | 先把视频解码成帧，转成 RGB uint8，然后逐帧 JPEG 压缩。报告平均 frame bpp 和 PSNR。说明这是 intra-frame JPEG，不是真正的视频压缩。 |
| 显微 | 仅作为可视化 baseline。将 `uint16` 或 float 强度用固定 min/max 或 percentile clipping 映射到 uint8。多通道显微可选择单通道、RGB composite 或逐通道 JPEG。 |
| 遥感 | 仅作为可视化 baseline。使用 RGB band 或固定 3-band false-color composite。除非实验明确研究这条量化路径，否则不要用 nvJPEG 结果判断多光谱科学保真度。 |
| 晶体衍射 | 仅作为可视化 baseline。先做 `log1p` 和固定 clipping，再映射到 uint8。可报视觉 PSNR，但 peak 强度等科学指标不能和 raw-value compressor 直接比较。 |
| ERA5 | 不推荐。只能把变量场渲染成 8-bit 图像后压缩，不应放进 ERA5 主科学压缩表。 |
| Nyx | 不推荐。只能用于 2D slice visualization。 |
| JHTDB | 不推荐。只能用于 velocity 或 vorticity slice visualization。 |
| Hurricane | 不推荐。只能用于 weather field visualization。 |

## 7. General 图像压缩模型数据处理

这里的 General 图像压缩模型指 CompressAI 风格或类似接口的神经图像 codec。

期望输入：

```text
[B, C, H, W]
C = 1 或 3，取决于具体模型
dtype = float32
range = [0, 1]
```

这一节作为四个 General 图像压缩模型的共享数据协议。等四个具体模型名确定后，结果表中可以把 General 分组替换成具体模型名，但数据处理规则保持一致。

关键规则：

- 推理前归一化到 `[0, 1]`。
- 重建后反归一化，再计算科学数据指标。
- 大图和大场数据要切成模型兼容的 patch。
- 如果模型只支持 3 通道，多通道数据按 3 个通道一组；最后一组不足 3 个通道时复制最后一个真实通道补齐。
- 对 3D/4D 科学数据，这类模型只能作为 2D slice baseline。

### General 图像模型 x 数据集处理

| 数据集 | 处理方式 |
|---|---|
| Kodak | 原生适合。RGB uint8 转成 float32 `[3, H, W] / 255`。如模型有 stride 要求，先 padding。报告 bpp、PSNR、MS-SSIM、时间和吞吐。 |
| UVG | 解码成帧后逐帧压缩。每帧使用 RGB `[3, H, W] / 255`。报告每个序列和所有帧的平均指标。该流程不利用时间冗余。 |
| 显微 | 灰度图如果模型支持单通道则用 `[1, H, W]`，否则复制成 `[3, H, W]`。多通道按 3 通道分组。保留原始 bit depth metadata，并在 inverse scaling 后算误差。 |
| 遥感 | 大场景先切 patch。图像风格评测可选 RGB bands；多光谱评测则按 spectral bands 每 3 个一组。使用 per-band 固定归一化或训练集统计量。 |
| 晶体衍射 | 先用 `log1p` 或 percentile clipping 映射到 `[0, 1]`。如果模型支持单通道，使用 `[1, H, W]`；否则复制到 3 通道。报告图像指标，并在 inverse transform 后报告 peak/intensity error。 |
| ERA5 | 将每个变量、level、time 切成 2D field。按项目已有 ERA5 约定，把 268 通道每 3 个一组。每个变量单独归一化，重建后反归一化。这是 2D field baseline。 |
| Nyx | 将 3D volume 沿某一轴切成 2D slice，或从 3D block 中抽取 2D plane。density、temperature、velocity 分量可单独压缩或按 3 通道组压缩。该方式不利用完整 3D 相关性。 |
| JHTDB | 使用 2D slice。天然 3 通道输入可以是同一 slice 的 `[u, v, w]`。每个 velocity component 使用一致归一化，重建后计算 velocity error 和频谱相关指标。 |
| Hurricane | 按 time、level、variable 切 2D field。只有在每个变量单独归一化的前提下，才把不同变量组为 3 通道。pressure/wind 等指标必须在原始单位上计算。 |

## 8. DCMVC 数据处理

DCMVC 是神经视频压缩模型，适合有真实时间顺序的 RGB/YUV 视频帧序列。当前仓库中的 DCMVC README 和 `dataset_config_example_rgb.json` 使用 RGB PNG 帧目录作为测试入口。

推荐输入目录：

```text
test_datasets/
  UVG/
    Beauty_1920x1080_120fps_420_8bit_YUV/
      im00001.png
      im00002.png
      im00003.png
      ...
```

配置文件中每个 sequence 需要记录：

```json
{
  "width": 1920,
  "height": 1080,
  "frames": 96,
  "gop": 32
}
```

典型运行入口：

```bash
python test.py \
  --rate_num 4 \
  --test_config ./dataset_config_example_rgb.json \
  --cuda 1 \
  --worker 1 \
  --output_path output.json \
  --i_frame_model_path ./ckpt/cvpr2023_image_psnr.pth.tar \
  --p_frame_model_path ./ckpt/dcmvc_p_frame.pth.tar
```

关键规则：

- DCMVC 是视频 codec，应优先使用真实时间帧序列。
- 输入应准备为 RGB PNG 帧，或使用脚本支持的 YUV420 路径。
- 静态图像和无时间维数据不适合作为 DCMVC 主实验。
- 如果科学数据被渲染成 RGB 帧，结果只能作为 visualization/video baseline，不能替代原始科学值压缩。
- GOP、frames、width、height 必须和帧目录一致。

### DCMVC x 数据集处理

| 数据集 | 处理方式 |
|---|---|
| Kodak | 不推荐。Kodak 是静态图像集，没有真实时间顺序。若强行把 24 张图串成视频，会引入虚假的时序关系，不应作为主实验。 |
| UVG | 原生适合。将 UVG YUV/RGB 视频解码成 RGB PNG 帧，按 `sequence/im00001.png` 格式组织。配置 `width`、`height`、`frames=96`、`gop=32`。报告 bpp、PSNR、MS-SSIM、I/P frame 统计、encode/decode time。 |
| 显微 | 仅适合 time-lapse 显微或 z-stack visualization。将每个时间点或 z-slice 映射成 RGB/gray PNG 帧。若原始是 `uint16` 或 float，应说明 8-bit/PNG 映射方式；科学强度误差不能直接和 raw compressor 比较。 |
| 遥感 | 仅适合多时相遥感 visualization。选固定 RGB/false-color bands，把不同 acquisition time 渲染为 PNG 帧。单景遥感图不适合 DCMVC。 |
| 晶体衍射 | 仅适合 time-resolved diffraction 或 scan sequence 的可视化视频。通常需要 `log1p + clipping + uint8` 后输出 PNG 帧。peak intensity 科学指标不应和原始值压缩直接比较。 |
| ERA5 | 可以作为可视化 baseline。选择少量变量，例如 t2m、z500、u10/v10，按时间渲染成 RGB 或伪彩色 PNG 帧后运行 DCMVC。主科学实验仍应使用 CRA5、cuSZ-Hi 或 CAESAR。 |
| Nyx | 可以作为 2D slice visualization baseline。选择某个变量和切片位置，按 simulation time 输出 PNG 帧。若只有单个 snapshot，不适合。 |
| JHTDB | 可以合理适配。湍流本身有时间序列，可将速度模长、vorticity 或 `[u,v,w]` 可视化为 RGB PNG 帧。若目标是科学压缩，仍需在原始速度场上用 cuSZ-Hi/CAESAR 评估。 |
| Hurricane | 可以合理适配。飓风演化有真实时间维，可将 pressure、wind speed、temperature 等变量渲染为 PNG 帧序列。用于视频可视化压缩；主科学误差仍需回到原始变量尺度计算。 |

## 9. CAESAR-V 数据处理

CAESAR-V 使用 CAESAR 的科学数据表示：

```text
npz key: data
shape: [V, S, T, H, W]
V = variables 或 channels
S = independent samples、scenes、patches 或 sequences
T = time steps 或 ordered slices
H, W = spatial dimensions
```

关键规则：

- 使用 `np.savez(path, data=array)` 保存 `.npz`。
- 有真实物理变量时，优先把变量放在 `V`。
- 有真实时间时，优先把时间放在 `T`。
- 如果是 3D 非时间数据，必须明确是把 `Z` 当作 `T`，还是把 volume 拆成 2D samples。
- 不同物理变量应分别归一化，不要把无关变量做全局 min-max。

### CAESAR-V x 数据集处理

| 数据集 | 处理方式 |
|---|---|
| Kodak | 可以测但不是主实验。使用 `V=3`，`S=24`，`T=1`，`H,W=image size`。这相当于把 RGB channel 当变量，把静态图像当单帧样本。 |
| UVG | 可以合理适配。使用 `V=3`，`S=sequence or clip index`，`T=frame index`，`H,W=frame size`。长视频切成固定长度 clips。 |
| 显微 | 可以合理适配。fluorescence channels 作为 `V`；样本或 tile 作为 `S`；time-lapse frame 或 z-slice 作为 `T`。普通单张 2D 图像用 `T=1`。 |
| 遥感 | 可以合理适配。spectral bands 作为 `V`；scene patches 作为 `S`；如果有多时相数据，acquisition time 作为 `T`，否则 `T=1`。 |
| 晶体衍射 | 可以合理适配。detector channel 或派生 intensity channel 作为 `V`；样本作为 `S`；time-resolved diffraction 或 scan index 作为 `T`。单张 pattern 用 `T=1`。 |
| ERA5 | 推荐。pressure/surface variables 和 levels 组织成 `V`；日期或 spatial crop 作为 `S`；reanalysis/forecast time 作为 `T`；lat/lon 作为 `H,W`。按 ERA5 变量/level 单独归一化。 |
| Nyx | 可以转换后测试。3D snapshot 可把 `Z` slices 映射为 `T`，或拆成 2D slice samples。如果有多个 simulation time，则用真实时间作为 `T`，`Z` 维通过切片处理。 |
| JHTDB | 推荐。`u,v,w` 作为 `V=3`；独立 spatial crops 作为 `S`；time steps 或 z-slices 作为 `T`。如果评估时序重建，优先用真实时间作为 `T`。 |
| Hurricane | 推荐。气象变量和 levels 作为 `V`；storm cases 或 spatial crops 作为 `S`；simulation/reanalysis time 作为 `T`；lat/lon 作为 `H,W`。 |

## 10. CAESAR-D 数据处理

CAESAR-D 在 CAESAR-V 的基础上加入 keyframe compression 和 conditional diffusion/interpolation。它只适合有真实顺序帧的数据。

期望输入仍然是：

```text
[V, S, T, H, W]
```

关键规则：

- 只要存在真实时间维，就优先把真实时间放在 `T`。
- 测试前固定 keyframe interval，例如 4、8 或 16。
- 不要把 CAESAR-D 作为静态单图数据的主实验。
- 如果可行，分别报告 keyframe bitrate 和 intermediate frame reconstruction quality。

### CAESAR-D x 数据集处理

| 数据集 | 处理方式 |
|---|---|
| Kodak | 不推荐。静态图像只有 `T=1`，没有 intermediate frame 可供 diffusion interpolation。应使用 CAESAR-V 或图像 codec。 |
| UVG | 推荐。使用 `V=3`，`S=sequence/clip`，`T=frame index`。选择固定长度 clip 和 keyframe interval。分别评估 keyframes 和生成的 intermediate frames。 |
| 显微 | 仅在 time-lapse 或 z-stack 数据上推荐。channels 作为 `V`，sample/tile 作为 `S`，time 或 z 作为 `T`。静态显微图不作为主实验。 |
| 遥感 | 适合多时相遥感。bands 作为 `V`，scene/patch 作为 `S`，acquisition date 作为 `T`。单景遥感图不作为主实验。 |
| 晶体衍射 | 仅适合 time-resolved diffraction、scan sequence 或有序 tilt/rotation 数据。intensity channel 作为 `V`，sample 作为 `S`，scan/time 作为 `T`。 |
| ERA5 | 推荐。使用真实 weather time 作为 `T`。变量和 levels 组成 `V`。keyframe interval 应匹配数据时间分辨率，例如数据支持时可选每 6 或 12 小时一个 keyframe。 |
| Nyx | 仅在有多个 simulation time steps 时推荐。simulation time 作为 `T`。如果只有单个 snapshot，不应作为 CAESAR-D 主结果。 |
| JHTDB | 推荐。湍流 time steps 作为 `T`，`u,v,w` 作为 `V`。对 interpolated frames 报 velocity RMSE 和 spectral error。 |
| Hurricane | 推荐。storm evolution time 作为 `T`；variables 和 levels 作为 `V`。报告 pressure/wind RMSE，以及可用的 storm-track 或 storm-center error。 |

## 11. CRA5 数据处理

CRA5 是 ERA5 专用压缩模型。它应作为 ERA5 原生模型评测，不应强行套到无关数据集。

期望输入：

```text
[1, 268, 721, 1440]
```

项目已有约定：

- `268 channels = 7 个 pressure variables x 37 个 pressure levels + 9 个 surface variables`
- 输入文件为配对的 `{timestamp}_pressure.nc` 和 `{timestamp}_single.nc`
- 归一化参数来自 `models/CRA5/cra5/api/mean_std.json` 和 `mean_std_single.json`

关键规则：

- 除非重训或改输入接口，否则 CRA5 只测 ERA5。
- 不要为了统一表格把其他数据集 padding 成 268 通道。
- 报告物理指标前必须反归一化。

### CRA5 x 数据集处理

| 数据集 | 处理方式 |
|---|---|
| Kodak | 不推荐。无法有意义地映射到 268 个 ERA5 channels。 |
| UVG | 不推荐。RGB 视频帧不符合 ERA5 变量结构和 CRA5 训练分布。 |
| 显微 | 不推荐。需要重训或新模型。 |
| 遥感 | 不推荐。需要重训或新模型。 |
| 晶体衍射 | 不推荐。需要重训或新模型。 |
| ERA5 | 原生适合。读取 `pressure.nc` 和 `single.nc`，组装 268 通道，应用 CRA5 mean/std normalization，执行 encode/decode，反归一化后评估全部通道或 selected target channels。 |
| Nyx | 不推荐。虽然是科学数据，但变量布局和空间语义不匹配 CRA5。 |
| JHTDB | 不推荐。湍流变量不匹配 ERA5 channels 和 normalization。 |
| Hurricane | 原始形态下不推荐。如果数据变量、levels、grid 和 ERA5 完全一致，可以考虑适配；否则需要重训或设计新的 CRA5-style 模型。 |

## 12. 推荐实验顺序

建议分阶段运行，方便定位失败原因。

1. 图像 sanity check：
   - Kodak 用 nvJPEG 和 General 图像模型。
   - UVG 用 nvJPEG 和 General 图像模型逐帧压缩。
   - UVG 用 DCMVC 跑真实视频压缩，作为视频模型主 baseline。

2. 先打通已有 ERA5 路径：
   - CRA5 on ERA5。
   - General 图像模型按已有 3-channel grouping 跑 ERA5。
   - cuSZ-Hi 跑 ERA5 raw arrays。
   - CAESAR-V 和 CAESAR-D 跑转换为 `[V, S, T, H, W]` 的 ERA5。

3. 科学图像数据：
   - 显微、遥感、晶体衍射先跑 cuSZ-Hi、General 图像模型和 visualization-only nvJPEG。
   - 多通道或有序数据再跑 CAESAR-V。
   - 只有存在有意义 time/z/scan order 时才跑 CAESAR-D。
   - 如果能形成真实时间帧或有序可视化帧，再额外跑 DCMVC visualization baseline。

4. 3D/4D 科学场：
   - Nyx、JHTDB、Hurricane 先跑 cuSZ-Hi。
   - 有时间或有序 slice 时再加入 CAESAR-V/D。
   - JHTDB 和 Hurricane 可额外生成 PNG 帧序列跑 DCMVC，但要和原始值科学压缩结果分表。
   - General 图像模型只作为 2D slice baseline。

## 13. 指标

所有方法通用指标：

- compressed size
- compression ratio
- encode time
- decode time
- encode throughput
- decode throughput

图像/视频指标：

- bpp
- PSNR
- MS-SSIM
- UVG 的 per-frame average
- DCMVC 的 I-frame/P-frame bits 和 GOP 平均指标

科学数据指标应在 inverse normalization 后计算：

- MSE
- RMSE
- NRMSE
- max absolute error
- mean relative error
- per-variable RMSE

有条件时加入领域指标：

| 数据集 | 额外指标 |
|---|---|
| 晶体衍射 | peak position error、peak intensity relative error |
| 遥感 | spectral angle mapper、per-band error |
| ERA5 | selected variable RMSE、anomaly correlation |
| Nyx | power spectrum error、density relative error |
| JHTDB | velocity RMSE、vorticity error、energy spectrum error |
| Hurricane | minimum pressure error、maximum wind error、storm-center error |

## 14. 输出目录结构

推荐目录结构：

```text
benchmark_artifacts/
  prepared/
    <dataset>/<method>/<sample_id>/
  compressed/
    <dataset>/<method>/<sample_id>/
  reconstructed/
    <dataset>/<method>/<sample_id>/
  metrics/
    <dataset>/<method>/summary.json
    <dataset>/<method>/per_sample.csv
  logs/
    <dataset>/<method>/<run_id>.log
```

每个 `summary.json` 建议包含：

```json
{
  "dataset": "era5",
  "method": "cra5",
  "num_samples": 1,
  "compressed_size_bytes": 123,
  "original_size_bytes": 456,
  "compression_ratio": 3.7,
  "bpp": null,
  "mse": 0.0,
  "rmse": 0.0,
  "nrmse": 0.0,
  "psnr": null,
  "encode_time_avg": 0.0,
  "decode_time_avg": 0.0,
  "encode_throughput": 0.0,
  "decode_throughput": 0.0,
  "notes": "metrics computed after de-normalization"
}
```

## 15. 主要结论

- cuSZ-Hi 应作为显微、遥感、晶体衍射、ERA5、Nyx、JHTDB、Hurricane 的主要非学习科学压缩 baseline。
- nvJPEG 主要用于 Kodak 和 UVG frames，也可作为科学图像的可视化 baseline。
- General 图像模型原生适合 Kodak，对其他数据集主要作为 frame、slice 或 patch baseline。
- DCMVC 是 UVG 的主视频压缩模型，也可用于 JHTDB、Hurricane 等有真实时间维数据的可视化视频 baseline。
- CAESAR-V 应用于转换成 `[V, S, T, H, W]` 的科学数据。
- CAESAR-D 只应在存在真实时间维或有序帧维时使用。
- CRA5 应视为 ERA5 原生模型，不应在未重训的情况下强行扩展到无关数据集。
