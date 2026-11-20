"""Structural ablations using the baseline's development training recipe."""
from __future__ import annotations

import copy
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torchvision.models import resnet50, ResNet50_Weights

from .baseline import BATCH_SIZE, DEVICE, MaskEncoder, augment_batch, make_criterion, make_optimizer, predict_proba, seed_everything
from .project import SEED, development, digest, dump_json, fresh, paths, record, source_snapshot, definition_contract
from .reporting import classification_metrics, paired_comparison, threshold_from_development


class ImageOnlyAttentionModel(nn.Module):
    """Three image sequences and slice attention, without a mask encoder."""
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        # Parameter-free placeholder preserves the original optimizer grouping.
        self.mask_features = nn.Identity()
        self.attention = nn.Sequential(
            nn.Linear(2048, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, 1),
        )
        self.head = nn.Sequential(nn.LayerNorm(2048), nn.Dropout(0.4), nn.Linear(2048, 1))

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = images.shape
        image_feat = self.image_features(images.view(B * S, C, H, W))
        feat = torch.flatten(image_feat, 1).view(B, S, -1)
        attn = torch.softmax(self.attention(feat).squeeze(-1), dim=1)
        pooled = (feat * attn.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class MeanPoolingDualBranchModel(nn.Module):
    """Three image sequences and masks, with uniform averaging over four slices."""
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        self.mask_features = MaskEncoder()
        self.attention = nn.Identity()
        self.head = nn.Sequential(nn.LayerNorm(2304), nn.Dropout(0.4), nn.Linear(2304, 1))

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = images.shape
        image_feat = self.image_features(images.view(B * S, C, H, W))
        image_feat = torch.flatten(image_feat, 1).view(B, S, -1)
        mask_feat = self.mask_features(masks.view(B * S, 3, H, W)).view(B, S, -1)
        feat = torch.cat([image_feat, mask_feat], dim=-1)
        pooled = feat.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def train_structure_fold(
    variant: str,
    Xtr: torch.Tensor,
    Mtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    Mva: torch.Tensor,
    yva: torch.Tensor,
    fold: int,
) -> tuple[nn.Module, int]:
    model_type = {"no_mask": ImageOnlyAttentionModel, "no_attention": MeanPoolingDualBranchModel}[variant]
    model = model_type().to(DEVICE)
    optimizer = make_optimizer(model)
    criterion = make_criterion(ytr.numpy())
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_auc, best_state, best_epoch, patience = 0.0, None, 1, 0

    for epoch in range(1, 31):
        model.train()
        perm = torch.randperm(len(Xtr))
        epoch_loss = 0.0
        for i in range(0, len(Xtr), BATCH_SIZE):
            bi = perm[i : i + BATCH_SIZE]
            xb, mb = Xtr[bi], Mtr[bi]
            xb, mb = augment_batch(xb, mb)
            xb = xb.to(DEVICE)
            mb = mb.to(DEVICE)
            yb = ytr[bi].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss = criterion(model(xb, mb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * len(bi)

        val_pred = predict_proba(model, Xva, Mva)
        val_auc = float(roc_auc_score(yva.numpy(), val_pred))
        if val_auc > best_auc:
            best_auc, best_state, best_epoch, patience = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            patience += 1
        if epoch % 3 == 0 or patience >= 8:
            print(
                f"fold {fold} epoch {epoch:02d} loss={epoch_loss/len(Xtr):.4f} "
                f"val_auc={val_auc:.4f} best={best_auc:.4f}",
                flush=True,
            )
        if patience >= 8:
            break

    model.load_state_dict(best_state)
    return model, best_epoch


def structure_output_directory(cfg, experiment="structure_v1"):
    if not isinstance(experiment, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", experiment):
        raise ValueError("结构消融实验名只使用字母、数字、下划线或短横线。")
    return paths(cfg)["run"] / "structure_ablation" / experiment


def structure_input_fingerprints(cfg):
    p = paths(cfg)
    return {"cache_sha256": digest(p["cache"]), "cohort_sha256": digest(p["cohort"]),
            "temporal_split_sha256": digest(p["temporal"] / "temporal_split.csv")}


def validate_development_predictions(table, dev):
    if not {"mr", "label", "probability"} <= set(table.columns) or not table.mr.is_unique:
        raise ValueError("开发预测缺少必要字段或患者编号重复。")
    if set(table.mr) != set(dev.mr):
        raise ValueError("开发预测与当前开发患者不完全一致。")
    aligned = table.set_index("mr").loc[dev.mr]
    if not np.array_equal(aligned.label.to_numpy(), dev.label.to_numpy()):
        raise ValueError("开发预测的结局与患者清单不一致。")
    probability = aligned.probability.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("开发预测概率必须是[0,1]内的有限值。")
    if "development_fold" in aligned and not np.array_equal(
            aligned.development_fold.to_numpy(), dev.development_fold.to_numpy()):
        raise ValueError("开发预测的折号与患者清单不同。")


def run_structure_ablation(cfg, variant, experiment="structure_v1"):
    if variant not in {"no_mask", "no_attention"}:
        raise ValueError("结构消融只支持 no_mask 或 no_attention。")
    p, dev = paths(cfg), development(cfg)
    output = structure_output_directory(cfg, experiment) / variant
    if output.exists():
        raise FileExistsError(f"该组结果已存在：{output}。可关闭对应开关读取结果；如需另跑，修改 STRUCTURE_EXPERIMENT。")
    fingerprints = structure_input_fingerprints(cfg)
    with np.load(p["cache"], allow_pickle=False) as cache:
        indices = dev.cache_index.to_numpy(dtype=int)
        if not np.array_equal(cache["mrs"].astype(str)[indices], dev.mr.to_numpy()):
            raise ValueError("开发患者与缓存索引不一致。")
        if not np.array_equal(cache["labels"][indices], dev.label.to_numpy()):
            raise ValueError("缓存结局与开发患者清单不一致。")
        X = torch.from_numpy(cache["images"][indices].astype(np.float32))
        M = torch.from_numpy(cache["masks"][indices].astype(np.uint8))
    if tuple(X.shape[1:]) != (4, 3, 224, 224) or M.shape != X.shape:
        raise ValueError("结构消融沿用四层、三序列、三类掩模、224×224缓存。")
    y = dev.label.to_numpy(dtype=int)
    if set(y) != {0, 1} or np.bincount(y, minlength=2).min() < 5:
        raise ValueError("开发患者需要两类结局，且每类至少5例。")
    folds = dev.development_fold.to_numpy(dtype=int)
    expected_folds = np.zeros(len(dev), dtype=int)
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y), 1):
        expected_folds[valid] = fold
    if not np.array_equal(folds, expected_folds):
        raise ValueError("当前折号与02的原始五折规则不一致，请核对患者顺序和清单。")
    baseline_oof = p["temporal"] / "development_oof.csv"
    if baseline_oof.exists():
        baseline = pd.read_csv(baseline_oof, dtype={"mr": str}).rename(columns={"dev_oof_prob": "probability"})
        validate_development_predictions(baseline, dev)
    output.mkdir(parents=True)
    snapshot = source_snapshot(cfg)
    notebook_name = "03_任务3_原单序列消融整合.ipynb"
    training = {"variant": variant, "experiment": experiment, "input_files": fingerprints,
        "notebook03_definitions_sha256": snapshot[notebook_name], "code_contract": structure_code_contract(cfg), "seed": SEED, "batch_size": BATCH_SIZE,
        "optimizer": "AdamW", "backbone_lr": 1e-4, "remaining_modules_lr": 5e-4, "weight_decay": 1e-4,
        "max_epochs": 30, "patience": 8, "gradient_clip_norm": 1.0,
        "loss": "original BCEWithLogitsLoss with training-fold pos_weight", "augmentation": "original augment_batch",
        "image_channels": 3, "mask_channels_used": 0 if variant == "no_mask" else 3,
        "slice_pooling": "attention" if variant == "no_mask" else "mean",
        "scope": "development_only", "selection": "same validation fold selects epoch and reports AUC"}
    dump_json(output / "training_config.json", training)
    seed_everything(SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_rows = []
    for fold in range(1, 6):
        valid, train = np.flatnonzero(folds == fold), np.flatnonzero(folds != fold)
        model, best_epoch = train_structure_fold(
            variant, X[train], M[train], torch.from_numpy(y[train].astype(np.float32)),
            X[valid], M[valid], torch.from_numpy(y[valid].astype(np.float32)), fold,
        )
        oof[valid] = predict_proba(model, X[valid], M[valid])
        auc = float(roc_auc_score(y[valid], oof[valid]))
        fold_rows.append({"fold": fold, "n_train": len(train), "n_valid": len(valid), "auc": auc,
                          "best_epoch": int(best_epoch), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        torch.save(model.state_dict(), output / f"fold_{fold}.pt")
        print(f"{variant}: fold {fold}, AUC={auc:.4f}, best_epoch={best_epoch}", flush=True)
        del model
    if fingerprints != structure_input_fingerprints(cfg) or training["code_contract"] != structure_code_contract(cfg):
        raise ValueError("训练期间输入或Notebook定义发生变化；保留折模型供排查，不发布汇总结果。")
    threshold = threshold_from_development(y, oof)
    if not np.isfinite(threshold):
        raise ValueError("该组未得到有限的Youden阈值，请检查开发预测。")
    table = dev[["mr", "label", "development_fold"]].copy()
    table["probability"] = oof
    validate_development_predictions(table, dev)
    table.to_csv(output / "development_oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    aucs = [row["auc"] for row in fold_rows]
    metrics = {"variant": variant, "experiment": experiment, "mean_fold_auc": float(np.mean(aucs)),
        "std_fold_auc": float(np.std(aucs)), "fold_aucs": aucs, "best_epochs": [row["best_epoch"] for row in fold_rows],
        **classification_metrics(y, oof, threshold, n_bootstrap=0)}
    metrics["pooled_auc"] = metrics["auc"]
    dump_json(output / "metrics.json", metrics)
    record(cfg, "structure_" + experiment + "_" + variant, {
        "protocol": "structural_ablation_original_training_settings", "training": training, "metrics": metrics})
    return metrics


def read_structure_results(cfg, experiment="structure_v1"):
    root, dev = structure_output_directory(cfg, experiment), development(cfg)
    results = {}
    fingerprints = None
    for variant in ("no_mask", "no_attention"):
        folder = root / variant
        if not (folder / "metrics.json").is_file():
            continue
        if fingerprints is None:
            fingerprints = structure_input_fingerprints(cfg)
        training = json.loads((folder / "training_config.json").read_text(encoding="utf-8"))
        if training["input_files"] != fingerprints or training["variant"] != variant or training["experiment"] != experiment:
            raise ValueError(f"{variant}的结果与当前数据或实验名不同。")
        expected_contract = training.get("code_contract")
        if expected_contract is None:
            if training["notebook03_definitions_sha256"] != "3384c3746316d5dcf1e91757d4d75a5bb4f8386efaf603d4951ac9b4eaccccdd":
                raise ValueError("旧消融结果来自未知版本，请先核对实现。")
            expected_contract = LEGACY_STRUCTURE_CONTRACT
        if structure_code_contract(cfg) != expected_contract:
            raise ValueError(f"{variant}的网络或训练实现与结果版本不同。")
        table = pd.read_csv(folder / "development_oof.csv", dtype={"mr": str})
        validate_development_predictions(table, dev)
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if not np.isclose(roc_auc_score(table.label, table.probability), metrics["pooled_auc"], atol=1e-12, rtol=0):
            raise ValueError(f"{variant}的汇总AUC与预测文件不一致。")
        results[variant] = {"metrics": metrics, "predictions": table}
    return results


def ablation_summary(cfg, experiment="structure_v1"):
    p = paths(cfg)
    rows = []
    baseline_path = p["temporal"] / "development_cv_metrics.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        rows.append({"model": "三序列＋病灶＋注意力", "mean_fold_auc": baseline["mean_auc"],
            "std_fold_auc": baseline["std_auc"], "pooled_auc": baseline["pooled_auc"],
            **{k: baseline[k] for k in ("accuracy", "sensitivity", "specificity")}})
    single_summary = p["single"] / "summary.csv"
    if single_summary.is_file():
        names = {"t1": "T1-FLAIR＋病灶＋注意力", "t2": "T2＋病灶＋注意力", "dwi": "DWI＋病灶＋注意力"}
        for _, row in pd.read_csv(single_summary).iterrows():
            rows.append({"model": names[row.sequence], "mean_fold_auc": row.mean_fold_auc, "pooled_auc": row.auc,
                         **{k: row[k] for k in ("accuracy", "sensitivity", "specificity")}})
    for variant, result in read_structure_results(cfg, experiment).items():
        m = result["metrics"]
        rows.append({"model": "去病灶分支" if variant == "no_mask" else "去注意力（平均汇总）",
                     **{k: m[k] for k in ("mean_fold_auc", "std_fold_auc", "pooled_auc", "accuracy", "sensitivity", "specificity")}})
    return pd.DataFrame(rows)


def compare_ablation_oof(cfg, experiment="structure_v1"):
    p, dev = paths(cfg), development(cfg)
    baseline = pd.read_csv(p["temporal"] / "development_oof.csv", dtype={"mr": str}).rename(columns={"dev_oof_prob": "probability"})
    validate_development_predictions(baseline, dev)
    candidates = {}
    for sequence in ("t1", "t2", "dwi"):
        path = p["single"] / sequence / "development_oof.csv"
        if path.is_file():
            candidates[sequence] = pd.read_csv(path, dtype={"mr": str})
    candidates.update({name: result["predictions"] for name, result in read_structure_results(cfg, experiment).items()})
    rows = []
    for name, table in candidates.items():
        validate_development_predictions(table, dev)
        rows.append({"a": "完整三序列模型", "b": name, **paired_comparison(baseline, table)})
    return pd.DataFrame(rows)


LEGACY_STRUCTURE_CONTRACT = {'seed_everything': '2f16b64e559c5d1efa0def1b98e9123cfdebaaccc483dfd396ab65d88b745761', 'MaskEncoder': 'b0a35f2673a19f24a5f492a757a65894b572a9819996d2aabb9572cc89b2de04', 'augment_batch': 'fff7579e369d643e92dbf59bed563b7b2b94344c48ce36598044feb088a19ae2', 'predict_proba': '94646b415f42d60d4a16416a05f26afed9656dd0fba15b09323c50f226898663', 'ImageOnlyAttentionModel': '782e3393d40f2625b0ef20bdf3f0634b54fdba5c32c2dcdf197ef23dba4ddfe7', 'MeanPoolingDualBranchModel': 'b32bc8847e1e040c085b0241be01197d45a2d849ba8bd45e15a5a577683cd1b0', 'make_optimizer': 'f36a1e6e87e23c160bca103ba98ea79aac68811c9a60324b91892a8413ae3730', 'make_criterion': '86e189018025dbbc4d0376bf5ec452ccb415b76507a5c4733d985901fd513528', 'train_structure_fold': '9208f3daa7e0cdac9cc1424cc5613164f3a07966c7e2dec362f71fb93379d160'}


def structure_code_contract(cfg):
    return definition_contract(cfg, "03", list(LEGACY_STRUCTURE_CONTRACT))
