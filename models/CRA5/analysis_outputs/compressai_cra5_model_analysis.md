# CompressAI 与 CRA5 VAEformer 268 通道结构分析

## 结论

可以把经典 `CompressAI` 模型扩展到 `268` 通道。

对经典 `CompressAI` 的卷积模型来说，这件事本身不需要 CRA5 那种特别的处理，核心就是把输入输出层从固定 `3` 通道改成可配置的 `in_channel/out_channel`。CRA5 在它自己的 `compressai` fork 里已经这样做了。

真正“特殊”的是 `CRA5 VAEformer(model_version=268)`。它不是简单把 `CompressAI` 的首尾卷积从 `3` 改成 `268`，而是换成了一套：

- patch embedding / patch decoding 的 transformer 主干
- `quant_conv` / `post_quant_conv` 的 latent 降维与回升维
- 单独的 hyperprior transformer

所以：

- 如果目标只是“让 CompressAI 能处理 ERA5 的 268 通道输入”，经典 `CompressAI` 直接扩输入输出通道即可。
- 如果目标是“做成 CRA5 那种模型”，那不是单纯改通道数，而是要引入 CRA5 的整套 VAEformer 结构。

## 关键源码证据

### 1. 原版 CompressAI 写死了 3 通道

文件：`/data/run01/scxj523/zsh/project/CompressAI/compressai/models/google.py`

- `FactorizedPrior.__init__`
  - `g_a` 第一层：`conv(3, N)`
  - `g_s` 最后一层：`deconv(N, 3)`

对应位置：

- [google.py](/data/run01/scxj523/zsh/project/CompressAI/compressai/models/google.py#L101)

### 2. CRA5 的 compressai fork 已改成可配置输入通道

文件：`/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/compressai/models/google.py`

- `FactorizedPrior.__init__(..., in_channel=3, ...)`
  - `g_a` 第一层：`conv(in_channel, N)`
  - `g_s` 最后一层：`deconv(N, in_channel)`

- `ScaleHyperprior.__init__(..., in_channel=3, ...)`
  - 同样把首尾层改成了 `in_channel`
  - 中间的超先验结构 `h_a / h_s`、熵瓶颈 `EntropyBottleneck`、高斯条件模型 `GaussianConditional` 没有因为 `268` 通道而改成别的算法

对应位置：

- [google.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/compressai/models/google.py#L100)
- [google.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/compressai/models/google.py#L265)

### 3. CRA5 VAEformer(268) 的特殊处理

文件：`/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/vaeformer/vaeformer.py`

`model_version == 268` 时：

- `embed_dim = 256`
- `z_channels = 256`
- `y_channels = 1024`
- `lower_dim = True`
- 主干输入输出：
  - `in_chans = 268`
  - `out_chans = 268`
- 主干 patch 配置：
  - `patch_size = (11, 10)`
  - `patch_stride = (10, 10)`
- hyperprior patch 配置：
  - `patch_size = (4, 4)`
  - `in_chans = 256`
  - `out_chans = 256`
  - `embed_dim = 360`

另外还显式加了：

- `quant_conv = Conv2d(2*y_channels, 2*embed_dim, 1)`，即 `2048 -> 512`
- `post_quant_conv = Conv2d(embed_dim, y_channels, 1)`，即 `256 -> 1024`

对应位置：

- [vaeformer.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/vaeformer/vaeformer.py#L93)

### 4. VAEformer 的 patch embedding / patch decoding 本体

文件：`/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/vaeformer/vit_nlc.py`

- patch embedding：
  - `self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride)`
- 解码端最终输出：
  - `self.final = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=out_chans, kernel_size=patch_size, stride=patch_stride, bias=False)`

对应位置：

- [vit_nlc.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/vaeformer/vit_nlc.py#L293)
- [vit_nlc.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/cra5/models/vaeformer/vit_nlc.py#L629)

## 真实运行入口

用户给出的真实运行入口：

- [test.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/test.py)
- [run_print_model_structures.sh](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/run_print_model_structures.sh)
- [print_model_structures.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/print_model_structures.py)

本次实际使用的命令：

```bash
/data/run01/scxj523/zsh/envs/zsh/bin/python -u /data/run01/scxj523/zsh/project/AIForCompression/CRA5/print_model_structures.py
```

## 真实输出摘录

下面是实际运行 `print_model_structures.py` 得到的关键输出摘录。

### 原版 CompressAI FactorizedPrior

```text
CompressAI FactorizedPrior(N=128, M=192)
StructFactorizedPrior(
  (g_a): Sequential(
    (0): Conv2d(3, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    ...
    (6): Conv2d(128, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
  )
  (g_s): Sequential(
    (0): ConvTranspose2d(192, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
    ...
    (6): ConvTranspose2d(128, 3, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
  )
)
Key convolution layers:
  g_a.0: Conv2d in=3, out=128, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_a.last: Conv2d in=128, out=192, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_s.0: ConvTranspose2d in=192, out=128, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_s.last: ConvTranspose2d in=128, out=3, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
```

### 扩成 268 通道后的 CompressAI FactorizedPrior

```text
CompressAI FactorizedPrior268(N=128, M=192)
FactorizedPrior268(
  (g_a): Sequential(
    (0): Conv2d(268, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    ...
    (6): Conv2d(128, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
  )
  (g_s): Sequential(
    (0): ConvTranspose2d(192, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
    ...
    (6): ConvTranspose2d(128, 268, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
  )
)
Key convolution layers for 268-channel adaptation:
  g_a.0: Conv2d in=268, out=128, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_a.last: Conv2d in=128, out=192, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_s.0: ConvTranspose2d in=192, out=128, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_s.last: ConvTranspose2d in=128, out=268, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
```

### 扩成 268 通道后的 CompressAI ScaleHyperprior

```text
CompressAI ScaleHyperprior268(N=128, M=192)
ScaleHyperprior268(
  (g_a): Sequential(
    (0): Conv2d(268, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    ...
    (6): Conv2d(128, 192, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
  )
  (g_s): Sequential(
    ...
    (6): ConvTranspose2d(128, 268, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
  )
  (h_a): Sequential(
    (0): Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ...
  )
  (h_s): Sequential(
    ...
    (4): Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
Key convolution layers for 268-channel adaptation:
  g_a.0: Conv2d in=268, out=128, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_a.last: Conv2d in=128, out=192, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_s.0: ConvTranspose2d in=192, out=128, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
  g_s.last: ConvTranspose2d in=128, out=268, kernel=(5, 5), stride=(2, 2), padding=(2, 2)
```

### CRA5 VAEformer(model_version=268)

```text
CRA5 VAEformer(model_version=268)
StructuralVAEformer268(
  (entropy_bottleneck): DummyEntropyBottleneck()
  (g_a): Sequential(
    (0): Conv2d(268, 1024, kernel_size=(11, 10), stride=(10, 10))
    (1): Identity()
  )
  (g_s): Sequential(
    (0): Identity()
    (1): ConvTranspose2d(1024, 268, kernel_size=(11, 10), stride=(10, 10), bias=False)
  )
  (quant_conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1))
  (post_quant_conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1))
  (h_a): Sequential(
    (0): Conv2d(256, 360, kernel_size=(4, 4), stride=(4, 4))
    (1): Identity()
  )
  (h_s): Sequential(
    (0): Identity()
    (1): Linear(in_features=360, out_features=8192, bias=False)
  )
  (gaussian_conditional): DummyGaussianConditional()
)
```

注意：这里的 `CRA5 VAEformer` 是 `print_model_structures.py` 的 fallback 结构摘要，不是完整 transformer 层级树；但这些输出已经足够说明它与经典 `CompressAI` 的差异。

## 真实运行时的异常

`print_model_structures.py` 最后没有完全成功结束，原因是脚本在打印 CRA5 关键层时假设 `model.g_a` 一定有 `patch_embed`，但 fallback 结构里 `g_a` 是 `Sequential`，没有这个属性。

真实报错如下：

```text
Traceback (most recent call last):
  File "/data/run01/scxj523/zsh/project/AIForCompression/CRA5/print_model_structures.py", line 467, in main
    print_cra5_model()
  File "/data/run01/scxj523/zsh/project/AIForCompression/CRA5/print_model_structures.py", line 432, in print_cra5_model
    ("g_a.patch_embed.proj", model.g_a.patch_embed.proj),
                             ^^^^^^^^^^^^^^^^^^^^^
  File ".../torch/nn/modules/module.py", line 1964, in __getattr__
    raise AttributeError(
AttributeError: 'Sequential' object has no attribute 'patch_embed'
```

对应代码位置：

- [print_model_structures.py](/data/run01/scxj523/zsh/project/AIForCompression/CRA5/print_model_structures.py#L426)

## 最终判断

### 经典 CompressAI 能不能扩到 268 通道？

能。

最直接的方法就是：

- 把首层输入从 `3` 改成 `268`
- 把末层输出从 `3` 改成 `268`
- 保持中间 latent 通道数 `N/M`、超先验结构、熵模型不变
- 重新训练

### CRA5 对 268 通道做了什么特殊处理？

如果说的是 CRA5 的 `compressai` 卷积分支，特殊处理很少，主要就是把 `in_channel` 参数化。

如果说的是 `CRA5 VAEformer(model_version=268)`，那特殊处理很多，主要包括：

- 268 通道直接做 patch embedding
- 主干不是 CNN，而是 transformer
- 中间显式做 latent 降维和回升维
- hyperprior 也不是普通 CNN，而是 patch-based encoder/decoder

所以，`268` 通道本身不是难点；`CRA5 VAEformer` 的特殊之处在于它的整体结构，而不是“因为 268 通道必须做特殊熵编码”。
