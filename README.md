# PSCI 实验 Notebook

日常使用 7 本 Notebook 和一个 `config.json`。代码都在 Notebook 内，不需要项目 Python 脚本。训练参数写在各本前面的参数单元里；模型和结果直接保存为 PT、joblib、NPZ 和 CSV。

| Notebook | 作用 | 建议运行位置 |
| --- | --- | --- |
| 00 数据准备 | Excel 匹配、影像检查、四层缓存与患者分组 | 完整数据电脑 |
| 01 影像组学 | 二维特征提取、影像组学及临床融合对照 | 完整数据电脑，CPU 即可 |
| 02 ResNet 分类 | 当前 ResNet50、临床对照、临床融合 | 算力机 |
| 03 消融实验 | 三种单序列、去病灶、去注意力 | 算力机 |
| 04 可解释性 | 折外病例的 Grad-CAM 与四层权重 | 算力机 |
| 05 自动勾画 | 二维 U-Net、四层 Dice、自动 mask | 完整数据电脑，NVIDIA GPU |
| 06 测试与对比 | 直接读取各组结果，统一时间测试 | 算力机 |

## 怎么开始

1. 在 `config.json` 填写 Excel、患者目录、缓存和清单路径。相对路径以 Notebook 所在文件夹为起点；各电脑的数据路径可以不同。
2. 已有 `cohort.csv` 和 `resnet50_manual_cache.npz` 时直接复用，不用重跑 00，也不需要删除原来的 outputs。云端只做分类可以不上传 Excel 和完整影像。
3. 打开相应 Notebook，从上往下执行。需要训练时开启对应的运行开关；新文件默认关闭耗时任务。
4. 02 和 03 使用相同参数、固定患者折。完成后在 06 设置结果目录并开启 `RUN_TEST`。

## 文件放在哪里

```text
results/
  resnet/         model.pt、clinical_models.joblib、五折权重、开发集预测
  ablation/       t1、t2、dwi、no_mask、no_attention 各自一个文件夹
  radiomics/      features.csv、models.joblib、开发集指标
  segmentation/  五折权重、validation_masks.npz、validation_dice.csv
  reports/        06 生成的预测、指标和图
```

01 在本地完成后，复制 `radiomics/features.csv` 和 `radiomics/models.joblib` 到算力机同名目录。05 在本地完成测试预测后，复制 `segmentation/validation_masks.npz` 和 `segmentation/validation_dice.csv`。06 直接读取这些文件，不检查电脑、版本编号或文件哈希。复制整个对应文件夹也可以。

想保留多次实验，直接修改 Notebook 的 `OUTPUT` 文件夹名称，再在 06 填对应位置。重训同名实验设 `RESTART=True`。02/03 默认复用已经完成的折，中断的当前折重训；05 每轮保存，可从上一轮继续。参数改变后不要复用上次折结果。02 重训后需要重跑临床融合。

## 数据与模型说明

- MoCA ≤22 定义结局，MoCA 不进入模型输入。临床输入固定为五项，缺失值在训练折内填补。
- 沿用时间分组；全部患者只有选定四层有标注。分割和影像组学只使用这四层。
- 02 内说明了当前 ResNet 的结构和训练调整。03 的消融针对这一模型。
- 05 使用直接写在 Notebook 中的二维 U-Net，不再运行 nnU-Net。仍需人工选层，不代表整套检查全自动。
- 影像组学二维形状按像素单位计算；分割 Dice 在工作网格上计算。不把它们当成全病灶体积或毫米距离。
- 临床融合的开发结果为一轮折外融合，完整效果由独立时间测试评价。开发调参完成后再看时间测试结果。

## 更新代码与依赖

这次调整了文件名和保存格式，更新时把这 7 本 Notebook 放在同一个文件夹，用一个新的简单 `config.json` 指向已有缓存即可。旧版代码已移到项目旁的 `1117_旧文件_20261120`，原始数据未移动。

当前网络的权重格式与旧版不同，不能直接把旧 PT 文件改名使用。旧实验的测试预测可以作为 CSV 加入 06；需要哪些列，06 里有说明。新网络和新的 U-Net 需要各自训练。

共同依赖：numpy、pandas、scipy、scikit-learn、matplotlib、joblib、Jupyter。00/01/05 读取原影像需要 nibabel；00 读取 Excel 需要 openpyxl。01 提取特征使用 MIRP 2.5.0（支持 Python 3.12）及其依赖 scikit-image；在 Notebook 中执行 `%pip install "mirp==2.5.0"` 后重启内核。强度和纹理由 MIRP 计算，二维形状由 scikit-image 计算；换库后需重新提取特征并训练01，06的文件路径保持不变。02～06 的神经网络部分需要配套的 PyTorch/torchvision。使用当前代码的 AMP 接口需要 PyTorch 2.3 或更新版本；scikit-learn 需支持 `keep_empty_features`（1.2 或更新），pandas 需支持混合日期解析（2.0 或更新）。异地读取 joblib 模型时，两端 scikit-learn 版本尽量一致。

本次在本机仅检查源码和 Notebook 结构，没有安装依赖或执行数据处理、训练、测试集评估。
