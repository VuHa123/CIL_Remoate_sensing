# BiCMPT

面向高光谱影像（HSI）与 LiDAR 的多模态类别增量学习研究代码。主模型使用双路 Prompt-ViT 提取模态特征，通过任务感知分组层融合，并在增量阶段扩展分类头、模态映射网络与 exemplar memory。

## 项目结构

```text
my_cil/
├── src/my_cil/               # 可复用 Python 包
│   ├── data/                 # Houston/Berlin 数据集与 patch 处理
│   ├── models/               # Prompt-ViT、任务感知层、ResNet、模态映射
│   ├── losses/               # CLUB 互信息估计与 Sinkhorn 距离
│   ├── training/             # 当前类别增量训练引擎
│   └── evaluation/           # 遗忘率等评估指标
├── scripts/                  # 训练、预训练、评估、数据准备和可视化入口
├── tests/                    # 可自动运行的单元测试
├── tools/                    # 独立辅助工具
```

## 安装

建议使用 Python 3.10+ 与独立虚拟环境：

```bash
python -m pip install -e ".[dev]"
```

## 数据

训练脚本默认从项目根目录下的 `data_houston/` 读取：

```text
data_houston/
├── train_patches.npy
├── train_labels.npy
├── test_patches.npy
└── test_labels.npy
```

原始 `.mat` 数据可通过数据准备入口转换：

```bash
python -m scripts.prepare_houston
```

## 常用入口

```bash
# 类别增量训练
python -m scripts.train_incremental

# ViT 主干预训练
python -m scripts.pretrain_backbone

# checkpoint 评估
python -m scripts.evaluate_checkpoint

# 单元测试
python -m pytest
```
