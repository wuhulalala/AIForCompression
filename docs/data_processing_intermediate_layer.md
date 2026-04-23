# 数据处理中间层设计说明

## 1. 先说结论

不应该为每个“数据集 x 模型”都写一个专用脚本。

错误方向是：

```text
Kodak_to_CAESAR.py
Kodak_to_DCMVC.py
ERA5_to_CAESAR.py
ERA5_to_CRA5.py
ERA5_to_cuSZ.py
UVG_to_DCMVC.py
UVG_to_CAESAR.py
...
```

如果有 9 个数据集和 7 个模型，这种方式会变成：

```text
9 x 7 = 63 条数据处理链
```

后面新增一个模型或一个数据集，脚本数量还会继续膨胀。

正确方向是加一个**数据处理中间层**：

```text
原始数据
  -> Dataset Adapter
  -> Canonical Dataset + Metadata
  -> Model View Builder
  -> Model Runner
  -> Inverse Transform
  -> Metrics
```

一句话解释：

```text
Dataset Adapter 负责把不同数据集读成统一中间表示；
Model View Builder 负责把统一中间表示变成某个模型需要的输入；
Metadata 负责记录所有变换，让重建结果能反变换并公平评测。
```

## 2. 为什么需要中间层

这些数据集的原始形态完全不同：

| 数据集 | 原始形态 |
|---|---|
| Kodak | RGB 静态图像 |
| UVG | 视频帧序列 |
| 显微 | 可能是 uint16、多通道、z-stack |
| 遥感 | GeoTIFF、多 band、超大图 |
| 晶体衍射 | 高动态范围强度图 |
| ERA5 | NetCDF，多变量、多 level、时间、经纬度 |
| Nyx | 3D 科学仿真场 |
| JHTDB | 3D/4D 湍流速度场 |
| Hurricane | 多变量、多 level、时间演化场 |

这些模型的输入也完全不同：

| 模型/方法 | 需要的输入 |
|---|---|
| cuSZ-Hi | raw binary + dtype + shape |
| nvJPEG | uint8 RGB/gray image |
| General 图像模型 | `[B, C, H, W]`，float32，`[0,1]` |
| DCMVC | RGB PNG 帧序列 + dataset config |
| CAESAR-V | `[V, S, T, H, W]` 的 `.npz` |
| CAESAR-D | `[V, S, T, H, W]`，且需要真实时间维 |
| CRA5 | ERA5 专用 `[1, 268, 721, 1440]` |

如果每个数据集直接对接每个模型，就会很乱。中间层的作用就是把复杂度拆开：

```text
数据集怎么读，归 Dataset Adapter 管；
模型吃什么格式，归 Model View Builder 管。
```

## 3. 核心设计

中间层分成三部分：

```text
Dataset Adapter
Canonical Dataset
Model View Builder
```

### 3.1 Dataset Adapter

Dataset Adapter 只负责一件事：

```text
把某个原始数据集读进来，整理成统一中间格式，并写 metadata。
```

例如：

```text
Kodak Adapter:
  JPEG/PNG -> canonical image samples

UVG Adapter:
  YUV/video -> canonical frame sequence

ERA5 Adapter:
  pressure.nc + single.nc -> canonical scientific field

JHTDB Adapter:
  velocity fields -> canonical spatiotemporal field
```

它不关心后面用 CAESAR、cuSZ-Hi、DCMVC 还是 CRA5。

### 3.2 Canonical Dataset

Canonical Dataset 是项目内部统一保存的中间数据。

它不是某个模型的输入，而是一个稳定的“标准数据层”。

推荐目录：

```text
prepared_datasets/
  <dataset_id>/
    dataset_manifest.json
    samples.jsonl
    arrays/
    frames/
    metadata/
```

其中最重要的是：

```text
dataset_manifest.json  # 数据集级别说明
samples.jsonl          # 每个 sample/clip/patch/block 的说明
```

### 3.3 Model View Builder

Model View Builder 也只负责一件事：

```text
从 Canonical Dataset 生成某个模型需要的输入格式。
```

例如同一个 ERA5 canonical sample，可以生成不同 view：

```text
ERA5 canonical sample
  -> cuSZ-Hi view: raw binary + dim3
  -> CRA5 view: [1, 268, 721, 1440]
  -> CAESAR view: [V, S, T, H, W] npz
  -> General image view: 每 3 个变量一组 [3, H, W]
  -> DCMVC view: 少量变量渲染成 PNG 时间帧
```

这样 ERA5 只需要一个 Adapter，不需要分别写 `ERA5_to_cuSZ.py`、`ERA5_to_CRA5.py`、`ERA5_to_CAESAR.py`。

## 4. 一个具体例子：ERA5

### 4.1 没有中间层时

你可能会写：

```text
run_era5_cra5.py
run_era5_cuszhi.py
run_era5_caesar.py
run_era5_general_image.py
run_era5_dcmvc.py
```

每个脚本都要重复处理：

```text
读取 pressure.nc
读取 single.nc
组装变量
处理 level
归一化
记录 shape
计算指标
反归一化
```

这会导致两个问题：

1. 代码重复，容易不一致。
2. 不同模型的结果可能不是在同一套预处理上得到的，比较不公平。

### 4.2 有中间层时

只写一个 ERA5 Adapter：

```text
ERA5 Adapter
  input:
    2024-06-01T00:00:00_pressure.nc
    2024-06-01T00:00:00_single.nc

  output:
    prepared_datasets/era5_2024_subset/
      dataset_manifest.json
      samples.jsonl
      arrays/era5_20240601_0000.npy
```

这个 `.npy` 可以是统一中间表示，例如：

```text
shape: [V, T, H, W]
V = 268
T = 1
H = 721
W = 1440
```

然后不同模型只从这个 canonical sample 生成自己的 view。

### 4.3 ERA5 的 dataset_manifest.json

`dataset_manifest.json` 描述整个数据集。

示例：

```json
{
  "dataset_id": "era5_2024_subset",
  "dataset_name": "ERA5",
  "dataset_type": "scientific_field",
  "source_format": "netcdf",
  "canonical_format": "npy",
  "canonical_layout": "variable_time_height_width",
  "canonical_shape_semantics": ["variable", "time", "height", "width"],
  "variables": [
    {
      "name": "t_850",
      "unit": "K",
      "dtype": "float32",
      "normalization": {
        "type": "mean_std",
        "mean": 273.1,
        "std": 12.4,
        "scope": "training_set"
      }
    }
  ],
  "spatial_axes": {
    "height": "latitude",
    "width": "longitude"
  },
  "time_axis": {
    "name": "valid_time",
    "step": "1h"
  }
}
```

这个文件回答：

```text
这个数据集是什么？
原始格式是什么？
中间格式是什么？
维度分别代表什么？
变量有哪些？
每个变量单位是什么？
每个变量怎么归一化？
```

### 4.4 ERA5 的 samples.jsonl

`samples.jsonl` 每一行描述一个样本。

示例：

```json
{"sample_id":"era5_20240601_0000","array_path":"arrays/era5_20240601_0000.npy","original_files":["2024-06-01T00:00:00_pressure.nc","2024-06-01T00:00:00_single.nc"],"canonical_shape":[268,1,721,1440],"dtype":"float32","transform_chain":["read_netcdf","assemble_268_channels","mean_std_normalize"],"inverse_transform_chain":["mean_std_denormalize"],"time":"2024-06-01T00:00:00"}
```

这一行回答：

```text
这个样本来自哪些原始文件？
prepared array 在哪里？
shape 是什么？
做过哪些 transform？
怎么反变换？
对应哪个时间？
```

## 5. 同一个 ERA5 sample 如何给不同模型用

假设 canonical ERA5 是：

```text
arrays/era5_20240601_0000.npy
shape = [268, 1, 721, 1440]
```

### 5.1 给 CRA5

CRA5 需要：

```text
[1, 268, 721, 1440]
```

View Builder 做：

```text
读取 canonical [268, 1, 721, 1440]
  -> squeeze/transpose
  -> [1, 268, 721, 1440]
  -> 保存 CRA5 input 或直接传给 runner
```

它还会写 view metadata：

```json
{
  "view_type": "cra5_tensor",
  "source_sample_id": "era5_20240601_0000",
  "input_shape": [1, 268, 721, 1440],
  "normalization": "cra5_mean_std",
  "model": "CRA5"
}
```

### 5.2 给 cuSZ-Hi

cuSZ-Hi 需要 raw binary：

```text
input.data
dtype = f32
dim3 = DimX x DimY x DimZ
```

View Builder 做：

```text
读取 canonical array
  -> 选择一个变量或一组 level
  -> 导出 contiguous float32 raw
  -> 写 shape metadata
```

例如：

```json
{
  "view_type": "raw_binary",
  "source_sample_id": "era5_20240601_0000",
  "raw_path": "views/cuszhi/era5_20240601_0000_t850.data",
  "dtype": "float32",
  "dim3": [1440, 721, 1],
  "variable": "t_850",
  "error_bound": 0.001
}
```

注意这里 `dim3` 的顺序必须和 raw 导出顺序一致。

### 5.3 给 General 图像模型

General 图像模型需要：

```text
[B, C, H, W]
C 通常是 3
range = [0, 1]
```

View Builder 做：

```text
读取 canonical [268, 1, 721, 1440]
  -> 按变量每 3 个通道一组
  -> 每个变量用自己的 min/max 或 mean/std 转到 [0,1]
  -> 生成多个 [3, H, W] patch/tensor
```

view metadata 需要记录：

```json
{
  "view_type": "image_tensor_groups",
  "source_sample_id": "era5_20240601_0000",
  "groups": [
    {
      "group_id": 0,
      "channels": ["z_50", "z_100", "z_150"],
      "input_shape": [3, 721, 1440],
      "padding": [0, 0, 0, 0]
    }
  ],
  "range": [0, 1],
  "inverse_transform": "per_variable_denormalize"
}
```

### 5.4 给 CAESAR-V / CAESAR-D

CAESAR 需要：

```text
[V, S, T, H, W]
```

View Builder 做：

```text
多个 canonical ERA5 time samples
  -> 按时间排序
  -> stack 成 [V, S, T, H, W]
  -> 保存 npz，key=data
```

例如：

```json
{
  "view_type": "caesar_npz",
  "npz_path": "views/caesar/era5_20240601_sequence.npz",
  "shape": [268, 1, 24, 721, 1440],
  "V": "era5_variables_and_levels",
  "S": "sequence",
  "T": "hourly_time",
  "normalization": "per_variable"
}
```

### 5.5 给 DCMVC

DCMVC 需要 PNG 帧序列：

```text
sequence/
  im00001.png
  im00002.png
  ...
```

对于 ERA5，这不是主科学压缩，只能做 visualization baseline。

View Builder 做：

```text
读取某个变量的多个时间步
  -> 渲染成 RGB 或伪彩色图
  -> 保存 im00001.png, im00002.png ...
  -> 写 DCMVC dataset_config.json
```

metadata 必须明确：

```json
{
  "view_type": "png_frame_sequence",
  "source_dataset": "ERA5",
  "source_variable": "t2m",
  "lossy_preprocessing": true,
  "rendering": "colormap_to_uint8_png",
  "scientific_metrics_comparable": false,
  "intended_use": "visualization_video_baseline"
}
```

这句话很关键：

```text
DCMVC 压的是 ERA5 渲染视频，不是 ERA5 原始科学场。
```

## 6. 另一个具体例子：UVG

UVG 是真实视频数据，所以它的 canonical representation 可以是：

```text
sequence frames
T x C x H x W
```

### 6.1 UVG Adapter

UVG Adapter 做：

```text
读取原始 YUV 或视频文件
  -> 解码成 RGB frames
  -> 保存 canonical frames
  -> 写 manifest 和 samples.jsonl
```

目录：

```text
prepared_datasets/uvg/
  dataset_manifest.json
  samples.jsonl
  frames/
    Beauty/
      im00001.png
      im00002.png
      ...
```

sample metadata：

```json
{"sample_id":"uvg_beauty_clip000","sequence":"Beauty","frame_dir":"frames/Beauty","start_frame":1,"num_frames":96,"width":1920,"height":1080,"gop":32,"source_color_format":"yuv420","prepared_color_format":"rgb_png","transform_chain":["decode_yuv420","convert_to_rgb","save_png_frames"]}
```

### 6.2 UVG 给 DCMVC

DCMVC view 几乎就是直接使用 canonical frames：

```text
frames/Beauty/im00001.png
frames/Beauty/im00002.png
...
```

只需要生成 DCMVC 的 config：

```json
{
  "root_path": "prepared_datasets/uvg/frames",
  "test_classes": {
    "UVG": {
      "test": 1,
      "base_path": ".",
      "src_type": "png",
      "sequences": {
        "Beauty": {
          "width": 1920,
          "height": 1080,
          "frames": 96,
          "gop": 32
        }
      }
    }
  }
}
```

### 6.3 UVG 给 General 图像模型

General 图像模型不需要视频结构，只逐帧处理：

```text
读取每个 PNG frame
  -> [3, H, W] / 255
  -> compress/decompress
  -> 按帧算 PSNR/bpp
  -> 对整个 sequence 求平均
```

### 6.4 UVG 给 CAESAR-D

CAESAR-D 需要：

```text
[V, S, T, H, W]
```

UVG 可以转成：

```text
V = 3       # RGB
S = clip id
T = frame index
H, W = frame size
```

也就是：

```text
[3, 1, 96, 1080, 1920]
```

## 7. Metadata 应该怎么处理

metadata 是中间层最重要的部分。

它不是附属说明，而是整个 benchmark 的“账本”。

没有 metadata，你无法回答：

```text
这个模型到底压了什么？
输入是不是原始物理值？
有没有先转成 8-bit？
归一化参数是什么？
重建后怎么反归一化？
指标是在 [0,1] 上算的，还是在原始单位上算的？
不同模型是不是用了同一份 prepared data？
```

## 8. Metadata 分三层

建议分三层：

```text
Dataset Manifest
Sample Metadata
Model View Metadata
```

### 8.1 Dataset Manifest

描述整个数据集。

负责记录：

```text
dataset_id
dataset_name
dataset_type
source_format
canonical_format
canonical_layout
variables
units
global normalization policy
spatial axes
time axis
```

### 8.2 Sample Metadata

描述每个 sample、clip、patch 或 block。

负责记录：

```text
sample_id
source_files
prepared_path
original_shape
canonical_shape
dtype
time
patch_info
transform_chain
inverse_transform_chain
valid_mask
```

### 8.3 Model View Metadata

描述某个模型实际吃到的输入。

负责记录：

```text
view_id
view_type
source_sample_id
model_family
input_paths
input_shape
normalization used for this view
padding
channel grouping
lossy_preprocessing
scientific_metrics_comparable
```

## 9. transform_chain 是重点

`transform_chain` 记录从原始数据到 canonical data 做过什么。

例如显微数据：

```json
{
  "transform_chain": [
    {
      "name": "read_tiff",
      "dtype": "uint16"
    },
    {
      "name": "percentile_clip",
      "p_low": 0.1,
      "p_high": 99.9
    },
    {
      "name": "normalize_minmax",
      "min": 12,
      "max": 60521
    },
    {
      "name": "tile",
      "tile_size": [512, 512],
      "overlap": 0
    }
  ],
  "inverse_transform_chain": [
    {
      "name": "stitch_tiles",
      "tile_size": [512, 512],
      "overlap": 0
    },
    {
      "name": "inverse_minmax",
      "min": 12,
      "max": 60521
    }
  ]
}
```

这里要注意：

```text
percentile_clip 是不可逆的；
normalize_minmax 是可逆的；
tile/stitch 在没有 overlap 丢失时是可逆的。
```

所以 metadata 还应该标记：

```json
{
  "lossy_preprocessing": true,
  "lossy_steps": ["percentile_clip"],
  "scientific_metrics_comparable": "only_after_accounting_for_clipping"
}
```

## 10. 可逆和不可逆必须分清楚

有些处理是可逆的：

```text
transpose
reshape
tile without loss
normalize_minmax
normalize_mean_std
float32 raw export
```

有些处理是不可逆的：

```text
uint16 -> uint8
float32 -> rendered RGB PNG
percentile clipping
log1p 后如果没有保存反变换参数或原始值范围
只选 RGB bands 丢弃其他 spectral bands
```

不可逆处理不是不能做，但必须明确标记。

例如 nvJPEG 和 DCMVC 经常需要 uint8 图像：

```json
{
  "lossy_preprocessing": true,
  "reason": "model requires uint8 image input",
  "scientific_metrics_comparable": false,
  "allowed_metrics": ["visual_psnr", "visual_msssim", "bitrate", "encode_time", "decode_time"]
}
```

这可以避免一个严重错误：

```text
把 ERA5 渲染成 PNG 后用 DCMVC 压缩，
然后声称 DCMVC 压缩了 ERA5 原始科学数据。
```

这种说法是不对的。它只压缩了 ERA5 的可视化视频。

## 11. 推荐文件结构

建议把中间层输出和模型 view 分开：

```text
benchmark_data/
  canonical/
    era5_2024_subset/
      dataset_manifest.json
      samples.jsonl
      arrays/
        era5_20240601_0000.npy

    uvg/
      dataset_manifest.json
      samples.jsonl
      frames/
        Beauty/
          im00001.png
          im00002.png

  views/
    cra5/
      era5_2024_subset/
        view_manifest.json

    cuszhi/
      era5_2024_subset/
        t850/
          input.data
          view_manifest.json

    caesar_v/
      era5_2024_subset/
        era5_sequence.npz
        view_manifest.json

    dcmvc/
      uvg/
        dataset_config.json
        Beauty/
          im00001.png
          im00002.png
```

注意：

```text
canonical 是数据集标准层；
views 是模型输入层。
```

不要把二者混在一起。

## 12. 推荐代码结构

后续如果实现，可以按这个结构：

```text
AIForCompression/
  data_layer/
    schemas.py
    registry.py

    adapters/
      kodak.py
      uvg.py
      era5.py
      microscopy.py
      remote_sensing.py
      crystal_diffraction.py
      nyx.py
      jhtdb.py
      hurricane.py

    views/
      image_tensor.py
      png_frames.py
      raw_binary.py
      caesar_npz.py
      cra5_tensor.py
      dcmvc_config.py

    transforms/
      normalize.py
      tiling.py
      rendering.py
      netcdf.py
      metadata.py
```

接口概念：

```python
class DatasetAdapter:
    def ingest(self, raw_root, canonical_root) -> DatasetManifest:
        ...

class ModelViewBuilder:
    def build(self, manifest, samples, view_root) -> ViewManifest:
        ...

class MetricEvaluator:
    def evaluate(self, manifest, view_manifest, reconstruction_root) -> dict:
        ...
```

## 13. 新增数据集和新增模型时怎么扩展

### 13.1 新增一个数据集

只需要新增：

```text
data_layer/adapters/new_dataset.py
```

它负责：

```text
raw -> canonical + metadata
```

已有模型 view builder 不需要改，除非这个数据集有全新的数据形态。

### 13.2 新增一个模型

只需要新增：

```text
data_layer/views/new_model_view.py
```

它负责：

```text
canonical + metadata -> model input
```

已有 dataset adapter 不需要改。

这就是中间层的价值。

## 14. 最小可行版本

第一版不需要一次支持所有复杂情况。

建议先做最小版本：

```text
Dataset Adapters:
  ERA5
  UVG
  Kodak

Model Views:
  CRA5 tensor
  cuSZ-Hi raw
  General image tensor
  DCMVC PNG frames
  CAESAR npz

Metadata:
  dataset_manifest.json
  samples.jsonl
  view_manifest.json
```

先用这三个数据集打通：

```text
ERA5 -> CRA5 / cuSZ-Hi / General / CAESAR / DCMVC visualization
UVG  -> DCMVC / General / nvJPEG / CAESAR-D
Kodak -> General / nvJPEG
```

打通后再扩展显微、遥感、晶体衍射、Nyx、JHTDB、Hurricane。

## 15. 最重要的判断标准

判断设计是否正确，看三个问题：

```text
1. 新增一个模型时，需不需要改 9 个数据集脚本？
   正确答案：不需要，只加一个 Model View Builder。

2. 新增一个数据集时，需不需要改 7 个模型脚本？
   正确答案：不需要，只加一个 Dataset Adapter。

3. 任何一个结果能不能追溯到原始数据和完整 transform？
   正确答案：必须可以，通过 metadata 追溯。
```

如果这三个问题都满足，这个数据处理中间层就是合理的。

## 16. 一句话总结

这个中间层的核心不是“多存一个 metadata 文件”，而是把数据处理拆成两个稳定边界：

```text
Dataset Adapter:
  理解原始数据，把它变成 canonical data。

Model View Builder:
  理解模型输入，把 canonical data 变成模型 view。
```

metadata 贯穿中间，负责记录：

```text
数据从哪里来；
做过什么变换；
哪些变换可逆；
哪些变换不可逆；
模型实际压了什么；
指标应该在哪个尺度上算。
```

这比写 63 个专用脚本更清楚，也更容易维护。
