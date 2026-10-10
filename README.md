# YOLOv5 6.0 模型剪枝与知识蒸馏

基于 YOLOv5 v6.0 的目标检测实验项目，包含常规训练、BN 通道稀疏训练、结构化剪枝、剪枝后微调与知识蒸馏代码，并提供多种注意力模块及模型配置，便于开展模型压缩和检测效果对比实验。

## 主要功能

- 目标检测模型训练、验证、推理与模型导出。
- 基于 BatchNorm 参数的稀疏训练及通道剪枝。
- 剪枝模型微调，以及教师模型指导的知识蒸馏。
- 特征蒸馏实验代码，以及 CA、CBAM、ECA、SE 等注意力模块配置。

## 项目结构

| 文件或目录 | 用途 |
| --- | --- |
| `train.py` / `val.py` / `detect.py` | 常规训练、验证、推理 |
| `train_sparsity.py` / `prune.py` | 稀疏训练与模型剪枝 |
| `finetune_pruned.py` / `detectpruned.py` | 剪枝后微调与推理 |
| `train_distillation.py` | 知识蒸馏训练 |
| `train_pruned_distillation.py` | 剪枝模型蒸馏训练 |
| `train_featdistill.py` / `models/featdistill.py` | 特征蒸馏实验 |
| `models/` / `Attmodules/` | 模型结构、配置与注意力模块 |
| `data/` | 数据集配置与超参数配置 |
| `export.py` | 模型导出 |

## 环境准备

先安装与本机 CUDA 环境匹配的 PyTorch 和 torchvision，再安装项目依赖：

```bash
pip install -r requirements.txt
```

本项目基于较早的 YOLOv5 版本，依赖中的版本范围并不保证与所有最新版本兼容。请按实际环境确认依赖兼容性；`requirements.txt` 中未启用 PyTorch 的自动安装。

数据集和预训练权重需要自行准备。修改数据集 YAML 中的路径、类别数量 `nc` 和类别名称 `names`，并确认模型配置与权重的结构一致。

## 使用示例

下面以 `data/voc.yaml` 与 YOLOv5s 配置为例，路径、训练轮数和设备参数需按实际实验调整。

### 1. 常规训练

```bash
python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/voc.yaml --epochs 100 --imgsz 640 --name baseline
```

### 2. 稀疏训练

```bash
python train_sparsity.py --st --sr 0.0002 --weights runs/train/baseline/weights/best.pt --cfg models/yolov5s.yaml --data data/voc.yaml --epochs 100 --imgsz 640 --name sparse
```

稀疏系数需要结合验证集指标和 BN 参数分布调整。

### 3. 通道剪枝

```bash
python prune.py --percent 0.5 --weights runs/train/sparse/weights/best.pt --cfg models/yolov5s.yaml --data data/voc.yaml
```

脚本在项目根目录保存 `pruned_model.pt` 和 `orign_model.pt`。重复执行时会覆盖同名文件，请提前备份需要保留的权重。

### 4. 剪枝后微调

```bash
python finetune_pruned.py --weights pruned_model.pt --cfg models/yolov5s.yaml --data data/voc.yaml --epochs 100 --imgsz 640 --name finetune
```

### 5. 剪枝模型知识蒸馏

```bash
python train_pruned_distillation.py --distill --weights pruned_model.pt --t_weights runs/train/baseline/weights/best.pt --data data/voc.yaml --epochs 100 --imgsz 640 --name distilled
```

`--weights` 指定学生模型，`--t_weights` 指定教师模型；可通过 `--dist_loss` 和 `--temperature` 调整蒸馏设置。不同模型结构与剪枝比例需要单独验证兼容性。

### 6. 剪枝模型推理

```bash
python detectpruned.py --weights runs/train/finetune/weights/best.pt --source data/images
```

各脚本可通过 `python 脚本名.py --help` 查看参数。部分脚本默认使用本地实验路径，建议运行时显式指定权重、配置和数据集。

## 说明与参考

- 本仓库上传源代码、配置与示例说明，权重文件、训练输出、缓存和本地实验数据不纳入版本控制。
- 上述命令展示实验流程，未在本次仓库初始化中执行训练，实际精度和速度以复现实验为准。
- [历史剪枝说明与实验记录](README_pruning.md)
- [原 YOLOv5 说明](README_yolo.md)
- [YOLOv5 上游项目](https://github.com/ultralytics/yolov5)

## 许可证

保留项目现有的 [GPL-3.0 许可证](LICENSE)，使用与分发时请遵守相应条款。
