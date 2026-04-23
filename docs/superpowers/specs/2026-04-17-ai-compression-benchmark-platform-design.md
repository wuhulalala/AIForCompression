# AI 压缩 Benchmark 平台设计文档

## 1. 背景

`AIForCompression` 当前已经包含多种学习型压缩模型和 ERA5 benchmark 脚本，但整体工作流仍然偏研究脚本形态：

- benchmark 入口分散在各模型目录，缺少统一产品化入口
- 模型集成方式割裂，维护成本高
- 外部用户无法通过受控界面自助发起评测
- 结果散落在本地目录，缺少统一任务系统和结果归档

新的目标是在现有能力基础上，构建一个**面向外部用户**的 AI 压缩 benchmark 平台，并以 `CompressAI` 作为统一压缩后端基座。

## 2. 目标

构建一个 V1 公网 benchmark 平台，满足以下目标：

- 用户可以选择平台内置数据集，或上传自己的数据集
- 用户可以选择平台维护的模型和 checkpoint 发起 benchmark
- 同时支持图像数据和 ERA5/NetCDF 科学数据
- 平台自动执行压缩/解压评测，并返回统一指标、日志和结果文件
- 整体系统部署在单机单卡环境中，优先打通完整产品流程

## 3. 非目标

V1 明确不支持以下能力：

- 用户上传脚本并执行
- 用户上传模型代码
- 用户上传 checkpoint
- 多机多卡调度
- HPC/Slurm 集群接入
- `MS-SSIM` 指标

## 4. 产品范围

### 4.1 用户能力

用户可以：

- 选择平台内置数据集
- 上传自己的数据集
- 选择受支持的模型和 checkpoint
- 在平台限定范围内配置 benchmark 参数
- 提交异步 benchmark 任务
- 查看任务状态、日志、指标和可视化结果
- 下载 benchmark 结果工件

### 4.2 数据类型

V1 支持两类数据：

- `image_folder`
  - 平台内置经典图像数据集
  - 用户上传的 zip 图像集
- `era5_netcdf`
  - 平台内置 ERA5 benchmark 数据集
  - 用户上传的 NetCDF 数据

### 4.3 模型类型

V1 支持两类受控模型：

- CompressAI 官方模型
  - `bmshj2018-factorized`
  - `bmshj2018-hyperprior`
  - `mbt2018`
  - `mbt2018-mean`
  - `cheng2020-anchor`
  - `cheng2020-attn`
- 平台维护的科研模型
  - `DCAE`
  - `WeConvene`
  - `LIC_TCM`
  - `CRA5`

普通用户不能上传模型代码或 checkpoint。

## 5. 总体架构

V1 采用**前后端分离 + 独立 worker**的方案，但整体仍然部署在同一台机器上。

### 5.1 核心组件

- `frontend`
  - 任务创建页
  - 任务列表页
  - 任务结果页
- `web-api`
  - 用户鉴权
  - 上传入口
  - 参数校验
  - 任务创建与查询
  - 结果读取
- `scheduler`
  - 队列管理
  - 单卡串行调度
- `worker`
  - 数据准备
  - 模型加载
  - 压缩/解压执行
  - 指标计算
  - 结果工件生成
- `artifact-store`
  - 上传文件
  - prepared dataset
  - 日志
  - 结果文件
  - 图表
- `metadata-store`
  - 用户
  - 模型注册表
  - 数据集注册表
  - 任务记录
  - 指标摘要

### 5.2 部署方式

V1 部署为单机单卡：

- 1 个 Web/API 服务
- 1 个调度器
- 1 个 worker 进程
- 1 个数据库
- 1 个本地工件存储目录

## 6. 页面设计

### 6.1 任务创建页

提交流程如下：

1. 选择数据来源：内置数据集或上传数据
2. 选择数据类型：图像或 ERA5
3. 选择兼容的模型和 checkpoint
4. 配置 benchmark 参数
5. 提交前展示最终不可变配置摘要

参数建议包括：

- `quality`
- `batch_size`
- `num_samples`
- `selected_channels`，用于 ERA5 协议
- 少量模型特定白名单参数

### 6.2 任务列表页

展示内容：

- 任务 ID
- 数据集
- 模型
- 创建时间
- 当前状态
- 总耗时
- 成功/失败标记

支持按以下状态筛选：

- `queued`
- `running`
- `succeeded`
- `failed`

### 6.3 结果详情页

建议分为四个区域：

1. 任务摘要
2. 核心指标
3. 可视化结果
4. 工件下载

## 7. 数据模型设计

benchmark 任务一旦提交，即视为不可变。重跑不直接修改原任务，而是基于原配置创建新任务。

### 7.1 Task

- `task_id`
- `user_id`
- `task_type`
  - `builtin_dataset_benchmark`
  - `upload_dataset_benchmark`
- `dataset_kind`
  - `image_folder`
  - `era5_netcdf`
- `dataset_source`
  - 内置数据集 ID
  - 或上传数据包 ID
- `model_id`
- `checkpoint_id`
- `params`
- `status`
- `failure_stage`
- `status_message`
- `artifacts`
- `metrics_summary`
- `created_at`
- `started_at`
- `finished_at`

### 7.2 Dataset Registry

- `dataset_id`
- `name`
- `kind`
- `description`
- `storage_path`
- `sample_count`
- `default_eval_protocol`
- `visibility`

### 7.3 Model Registry

- `model_id`
- `name`
- `kind`
  - `compressai_builtin`
  - `custom_research`
- `supported_dataset_kinds`
- `checkpoint_list`
- `default_params`
- `visibility`

## 8. 评测协议设计

V1 建议引入 `evaluation protocol` 概念，将“数据集”与“如何评测”分离。

每个任务实际绑定：

- 一个数据集
- 一个模型
- 一个评测协议

### 8.1 协议职责

协议负责定义：

- 样本选择策略
- 最大样本数
- padding/cropping 规则
- 归一化规则
- 需要计算的指标
- 是否只统计 ERA5 目标通道
- 是否生成可视化
- 结果工件导出规则

### 8.2 ERA5 协议示例

- 全 268 通道协议
- 6 目标通道协议
- 指定变量子集协议

## 9. 核心抽象设计

为了同时支持图像和 ERA5，并避免 worker 逻辑充满分支判断，V1 采用两层 adapter 抽象：`DatasetAdapter` 和 `ModelAdapter`。

### 9.1 DatasetAdapter

建议接口：

- `validate(source) -> ValidationResult`
- `prepare(source) -> PreparedDataset`
- `iter_samples(prepared) -> Iterator[Sample]`
- `metadata(prepared) -> DatasetMetadata`

### 9.2 ModelAdapter

建议接口：

- `load(checkpoint, params) -> ModelHandle`
- `compress(model, sample) -> CompressedArtifact`
- `decompress(model, artifact) -> ReconstructedSample`
- `metric_inputs(original, reconstructed) -> MetricPayload`

### 9.3 V1 具体实现

数据集适配器：

- `ImageFolderDatasetAdapter`
- `ERA5NetcdfDatasetAdapter`

模型适配器：

- `CompressAIImageModelAdapter`
- `CustomResearchModelAdapter`

### 9.4 Worker 执行流程

worker 保持通用，不直接关心具体数据和模型细节：

1. 拉取任务
2. 通过 adapter 准备数据
3. 通过 adapter 加载模型
4. warmup
5. 执行 compress benchmark
6. 执行 decompress benchmark
7. 计算指标
8. 生成图表和结果文件
9. 更新任务状态

## 10. 任务生命周期

建议任务状态机如下：

- `draft`
- `validating`
- `queued`
- `preparing`
- `running`
- `succeeded`
- `failed`
- `canceled`

### 10.1 失败阶段

- `validation`
- `prepare_dataset`
- `load_model`
- `compress`
- `decompress`
- `metrics`
- `persist_artifacts`

### 10.2 状态消息

前端显示可读进度信息，例如：

- 正在校验上传数据
- 正在解压数据集
- 正在加载模型
- 正在评测第 `42/200` 个样本
- 正在写入结果文件

## 11. 调度与 Worker 设计

V1 采用单机单卡部署，因此调度系统不需要一开始就引入 Redis、Celery、Kafka 等重型基础设施。首版建议采用**数据库任务表 + 单后台 runner 轮询**的轻量方案。

### 11.1 角色划分

`web-api` 负责：

- 接收任务创建请求
- 校验参数与上传内容
- 创建任务记录
- 将通过校验的任务置为 `queued`

`task_runner` 负责：

- 轮询数据库中的待执行任务
- 以原子方式抢占一个任务
- 执行完整 benchmark 流程
- 更新任务状态、日志与结果工件

在 V1 中，`scheduler` 与 `worker` 可以在实现层合并为一个后台进程 `task_runner`。概念上仍可区分：

- `scheduler` 决定“下一个跑哪个任务”
- `worker` 决定“如何把该任务跑完”

### 11.2 V1 推荐实现

后台启动一个常驻进程，循环执行：

1. 查询最早进入 `queued` 状态的任务
2. 尝试以原子更新方式将任务状态改为 `preparing`
3. 若更新成功，说明该任务被当前 runner 抢占
4. 进入数据准备、模型加载和 benchmark 执行阶段
5. 执行结束后将任务更新为 `succeeded` 或 `failed`

这套机制在单机单卡环境下已经足够稳定，并且后续扩展到多 worker 时仍可复用同样的状态机和抢占逻辑。

### 11.3 原子抢占原则

不能使用“查到 queued 就直接跑”的方式，否则未来扩展到多个后台进程时可能出现重复执行。

应采用条件更新的方式抢任务，例如：

- 先读取一个候选 `queued` 任务
- 再执行带条件的状态更新：
  - 仅当该任务当前仍是 `queued` 时，才更新为 `preparing`
- 只有更新成功的进程才能继续执行该任务

这样即使未来引入第二个 runner，也不会重复占用同一个任务。

### 11.4 调度策略

V1 使用最简单的 FIFO 策略：

- 谁先提交谁先执行
- 同一时刻只有一个任务处于 `running`
- GPU 资源不做并发切分

后续如有需要，可再增加：

- 用户级限流
- 管理员优先级
- 重试队列
- 大任务/小任务优先级

但这些都不属于 V1 范围。

### 11.5 状态流转与执行边界

任务主状态流转如下：

- `draft`
- `validating`
- `queued`
- `preparing`
- `running`
- `succeeded`
- `failed`
- `canceled`

其中：

- `validating`
  - API 正在校验参数、上传内容和模型兼容性
- `queued`
  - 任务已入队，等待 GPU
- `preparing`
  - runner 已抢到任务，正在解压数据、准备工作目录、加载上下文
- `running`
  - 模型已加载，正在执行 benchmark

### 11.6 Runner 执行职责

一旦任务被抢占，runner 负责完成以下固定流程：

1. 创建任务工作目录
2. 加载任务配置
3. 调用 `DatasetAdapter.prepare`
4. 调用 `ModelAdapter.load`
5. 将状态更新为 `running`
6. 遍历样本执行 encode/decode benchmark
7. 记录时延、吞吐和中间统计
8. 汇总指标
9. 生成 `summary.json`、`metrics.csv`、`run.log` 等工件
10. 更新最终状态并回写数据库

这样调度层与执行层职责清晰：

- 调度层只负责决定“谁先跑”
- 执行层只负责决定“怎样把任务跑完”

### 11.7 前端状态展示

前端不需要直接感知调度细节，只需读取任务状态与少量辅助字段：

- `status`
- `status_message`
- `created_at`
- `started_at`
- `finished_at`

如需展示排队位置，可在查询时动态计算：

- 当前处于 `queued` 状态且创建时间早于本任务的任务数

从而向用户展示：

- 等待中
- 当前排队第 N 位
- 正在准备数据
- 正在运行第 `42/200` 个样本

### 11.8 为什么 V1 不引入重型队列系统

V1 的瓶颈是单张 GPU，而不是消息队列吞吐。

因此首版直接使用数据库任务表具备以下优势：

- 实现简单
- 排障成本低
- 状态天然可查询
- 适合与前端任务状态页面直接对接
- 后续扩展时仍可保留任务表和状态机

等平台进入多 worker、多 GPU 或多机部署阶段，再考虑引入独立队列系统更合适。

## 12. 指标体系

V1 不计算 `MS-SSIM`。

### 12.1 核心指标

- `mse`
- `rmse`
- `psnr`
- `bpp`
- `compression_ratio`
- `encode_time_avg`
- `decode_time_avg`
- `encode_throughput`
- `decode_throughput`
- `num_params`
- `model_size_mb`

### 12.2 指标定义

- `encode_time_avg`
  - 平均压缩耗时，单位 `s`
- `decode_time_avg`
  - 平均解压耗时，单位 `s`
- `encode_throughput`
  - 压缩吞吐，单位 `MB/s`
  - 公式：`原始输入字节数 / encode_time_avg`
- `decode_throughput`
  - 解压吞吐，单位 `MB/s`
  - 公式：`重建输出字节数 / decode_time_avg`

### 12.3 ERA5 扩展展示

ERA5 结果页可额外展示：

- 按通道误差统计
- 按变量误差表
- `z500`、`t850`、`v10`、`u10`、`t2m`、`msl` 的重点可视化

## 13. 内置数据集设计

### 13.1 图像 benchmark 数据集

建议内置：

- `Kodak`
- 小规模 `CLIC` 验证子集
- 小规模 `Tecnick` 验证子集

### 13.2 ERA5 benchmark 数据集

建议内置：

- 一套标准 ERA5 对外 benchmark 数据集
- 后续可扩展月度、区域、变量子集

### 13.3 管理原则

- 数据本体与评测协议分离
- 平台内置数据集统一注册
- 内置数据集只读
- 用户上传数据与平台数据分开存储

## 14. 上传校验与安全边界

所有上传内容都视为不可信输入。

### 14.1 接收格式

- 图像任务
  - `zip`
- ERA5 任务
  - `zip`
  - 或固定命名的 `*.nc`

### 14.2 校验规则

- 限制总上传大小，例如 `2GB`
- 限制文件数量，例如每任务最多 `5000` 张图像
- 校验扩展名和 MIME
- 解压后再次校验真实文件类型
- 禁止符号链接
- 禁止路径穿越
- 禁止隐藏可执行文件
- 强制 ERA5 `pressure/single` 配对规则
- 对上传图像做抽样解码检查
- 强制数据类型与模型兼容性校验
- 强制参数白名单和范围校验

### 14.3 运行期安全

- 上传目录与运行目录分离
- 原始上传文件只读保存
- 每个任务独立工作目录
- worker 只读取 prepared dataset
- 执行期间禁网
- 限制超时、内存和磁盘占用
- 限制日志长度，避免磁盘被刷满

## 15. 结果展示与工件输出

### 15.1 任务摘要

- 任务 ID
- 创建时间、完成时间
- 总耗时
- 数据集
- 模型
- checkpoint
- 提交参数
- 状态
- 失败阶段

### 15.2 核心指标区

- `mse`
- `rmse`
- `psnr`
- `bpp`
- `compression_ratio`
- `encode_time_avg`
- `decode_time_avg`
- `encode_throughput`
- `decode_throughput`
- `num_params`
- `model_size_mb`

### 15.3 可视化区

图像任务：

- 原图/重建图对比
- 样本指标分布图
- 不同质量档位对比图

ERA5 任务：

- 原场/重建场对比
- 通道误差分布
- 变量级热力图
- 重点变量对比图

### 15.4 可下载工件

- `summary.json`
- `metrics.csv`
- `run.log`
- 图表文件
- 任务配置快照

## 16. 非功能要求

### 16.1 可复现性

- 任务配置不可变
- 每次运行绑定明确模型版本、checkpoint 和 protocol
- 保存配置快照和结果工件

### 16.2 可比性

- 不允许用户上传模型代码
- 不允许用户上传 checkpoint
- 平台统一维护模型注册表和 benchmark 协议

### 16.3 可扩展性

设计上应支持后续扩展到：

- 多 worker
- 多 GPU
- Slurm/HPC
- 审核制插件式模型接入

## 17. V1 实施阶段

### Phase 1

- 建立任务、模型、数据集注册表
- 实现任务生命周期
- 接入本地工件存储
- 打通单 worker 执行闭环

### Phase 2

- 实现图像和 ERA5 数据集 adapter
- 实现 CompressAI 官方模型 adapter
- 实现科研模型 adapter

### Phase 3

- 实现 benchmark runner
- 加入 warmup、计时、指标计算和结果导出

### Phase 4

- 完成最小前端
- 任务创建页
- 任务列表页
- 结果详情页

### Phase 5

- 注册平台内置 benchmark 数据集
- 补齐结果可视化

### Phase 6

- 强化上传校验
- 加入运行时资源限制
- 补齐安全和运维防护

## 18. 结论

V1 平台应定位为**受控模型集合下的数据 benchmark 平台**，而不是公网脚本执行平台。它的核心价值在于：

- 以 `CompressAI` 作为统一后端基座
- 同时支持图像数据与 ERA5 科学数据
- 提供可复现、可比较、可追溯的 benchmark 流程
- 在单机单卡条件下快速形成可对外使用的产品闭环
