"""Small-cohort ResNet candidate; isolated outputs and inner epoch selection."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from .project import SEED, paths, read_cohort, cohort_identity, definition_contract, digest, dump_json, fresh, record
from .baseline import DEVICE, seed_everything, augment_batch, make_criterion
from .reporting import classification_metrics, threshold_from_development


class GatedSliceAttention(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.value = nn.Linear(features, hidden)
        self.gate = nn.Linear(features, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, features):
        return self.score(torch.tanh(self.value(features)) * torch.sigmoid(self.gate(features)))


class CompactMaskEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(.3))

    def forward(self, masks):
        return self.features(masks.float())


class OptimizedResNet(nn.Module):
    def __init__(self, settings, pretrained=True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        self.image_projection = nn.Sequential(nn.Linear(2048, settings["image_dim"]),
            nn.LayerNorm(settings["image_dim"]), nn.GELU(), nn.Dropout(.3))
        self.mask_features = CompactMaskEncoder(settings["mask_dim"])
        features = settings["image_dim"] + settings["mask_dim"]
        self.attention = GatedSliceAttention(features, settings["attention_dim"])
        self.head = nn.Sequential(nn.LayerNorm(features), nn.Dropout(.4), nn.Linear(features, 1))

    def train(self, mode=True):
        super().train(mode)
        # Keep pretrained running means/variances stable with small patient batches.
        for module in self.image_features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_training_stage(self, fine_tune):
        for parameter in self.image_features.parameters():
            parameter.requires_grad_(False)
        if fine_tune:
            for parameter in self.image_features[7].parameters():
                parameter.requires_grad_(True)
        for module in self.image_features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def forward(self, images, masks):
        b, slices, channels, h, w = images.shape
        image = self.image_features(images.reshape(b * slices, channels, h, w)).flatten(1)
        image = self.image_projection(image).reshape(b, slices, -1)
        mask = self.mask_features(masks.reshape(b * slices, 3, h, w)).reshape(b, slices, -1)
        features = torch.cat([image, mask], dim=-1)
        attention = torch.softmax(self.attention(features).squeeze(-1), dim=1)
        return self.head((features * attention.unsqueeze(-1)).sum(dim=1)).squeeze(-1)


def optimized_contract(cfg):
    return definition_contract(cfg, "02", ["GatedSliceAttention", "CompactMaskEncoder", "OptimizedResNet",
        "optimized_predict", "fit_optimized_epochs", "run_optimized_training", "optimized_data",
        "seed_everything", "augment_batch", "make_criterion"])


def optimized_data(cfg, scope):
    table = read_cohort(cfg).query("split == @scope").reset_index(drop=True)
    indices = table.cache_index.to_numpy(dtype=int)
    with np.load(paths(cfg)["cache"], allow_pickle=False) as cache:
        if cache["mrs"][indices].astype(str).tolist() != table.mr.tolist() or not np.array_equal(cache["labels"][indices], table.label):
            raise ValueError("缓存患者或结局与队列不一致。")
        if not np.array_equal(cache["max_layers"][indices], table.max_layer):
            raise ValueError("缓存取层与队列不一致。")
        images = torch.from_numpy(cache["images"][indices].astype(np.float32))
        masks = torch.from_numpy(cache["masks"][indices].astype(np.uint8))
    if images.shape[1:] != (4, 3, 224, 224) or masks.shape != images.shape:
        raise ValueError("优化版继续使用四层、三序列、224×224的原缓存。")
    return table, images, masks, table.label.to_numpy(dtype=int)


def optimized_predict(model, images, masks, batch_size=16):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits = model(images[start:start + batch_size].to(DEVICE), masks[start:start + batch_size].to(DEVICE))
            # Store float32 probabilities; avoid rounding at the final sigmoid in half precision.
            result.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(result)


def fit_optimized_epochs(settings, images, masks, y, train, epochs, seed, inner_valid=None):
    from sklearn.metrics import roc_auc_score
    seed_everything(seed)
    model = OptimizedResNet(settings).to(DEVICE)
    backbone = list(model.image_features.parameters())
    heads = [p for name, p in model.named_parameters() if not name.startswith("image_features.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": settings["backbone_lr"]},
        {"params": heads, "lr": settings["head_lr"]}], weight_decay=settings["weight_decay"])
    criterion = make_criterion(y[train])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_auc, best_epoch, stale, history = -float("inf"), 1, 0, []
    for epoch in range(1, epochs + 1):
        model.set_training_stage(epoch > settings["warmup_epochs"])
        model.train()
        permutation = np.asarray(train)[torch.randperm(len(train)).numpy()]
        loss_sum = 0.0
        for start in range(0, len(train), settings["batch_size"]):
            indices = permutation[start:start + settings["batch_size"]]
            xb, mb = augment_batch(images[indices], masks[indices])
            target = torch.as_tensor(y[indices], dtype=torch.float32, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss = criterion(model(xb.to(DEVICE), mb.to(DEVICE)), target)
            if not torch.isfinite(loss):
                raise ValueError("训练损失出现非有限值，停止并检查输入。")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(indices)
        row = {"epoch": epoch, "loss": loss_sum / len(train)}
        if inner_valid is not None:
            probability = optimized_predict(model, images[inner_valid], masks[inner_valid], settings["batch_size"])
            auc = float(roc_auc_score(y[inner_valid], probability))
            row["inner_auc"] = auc
            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch, 0
            else:
                stale += 1
        history.append(row)
        print(row, flush=True)
        if inner_valid is not None and stale >= settings["patience"] and epoch > settings["warmup_epochs"]:
            break
    return model, best_epoch if inner_valid is not None else epochs, history


def run_optimized_training(cfg):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    p, settings = paths(cfg), dict(cfg["optimized"])
    if settings["batch_size"] < 1 or settings["max_epochs"] <= settings["warmup_epochs"] or not 0 < settings["inner_fraction"] < .5:
        raise ValueError("检查批大小、最大训练轮数、预热轮数和内部留出比例。")
    fresh(p["optimized"])
    table, images, masks, y = optimized_data(cfg, "development")
    folds = table.development_fold.to_numpy(dtype=int)
    if set(folds) != {1, 2, 3, 4, 5}:
        raise ValueError("优化版沿用00固定的五折。")
    fingerprint = {"cohort_identity": cohort_identity(cfg), "cache_sha256": digest(p["cache"]), "code_contract": optimized_contract(cfg)}
    p["optimized"].mkdir(parents=True)
    dump_json(p["optimized"] / "training_config.json", {"settings": settings, "seed": SEED, **fingerprint,
        "protocol": "outer_patient_folds_inner_holdout_epoch_selection_refit", "clinical_fusion": "not_fitted"})
    oof, rows, selected_epochs = np.zeros(len(y), dtype=np.float32), [], []
    for fold in range(1, 6):
        outer_train, outer_valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner_train, inner_valid = train_test_split(outer_train, test_size=settings["inner_fraction"],
            stratify=y[outer_train], random_state=SEED + fold)
        selection = {"outer_train": table.mr.iloc[outer_train].tolist(), "outer_valid": table.mr.iloc[outer_valid].tolist(),
                     "inner_train": table.mr.iloc[inner_train].tolist(), "inner_valid": table.mr.iloc[inner_valid].tolist()}
        dump_json(p["optimized"] / f"fold_{fold}_patients.json", selection)
        probe, chosen, history = fit_optimized_epochs(settings, images, masks, y, inner_train,
            settings["max_epochs"], SEED + fold, inner_valid)
        del probe
        pd.DataFrame(history).to_csv(p["optimized"] / f"fold_{fold}_inner_history.csv", index=False)
        model, _, history = fit_optimized_epochs(settings, images, masks, y, outer_train, chosen, SEED + 100 + fold)
        oof[outer_valid] = optimized_predict(model, images[outer_valid], masks[outer_valid], settings["batch_size"])
        torch.save(model.state_dict(), p["optimized"] / f"fold_{fold}.pt")
        pd.DataFrame(history).to_csv(p["optimized"] / f"fold_{fold}_refit_history.csv", index=False)
        selected_epochs.append(chosen)
        rows.append({"fold": fold, "auc": float(roc_auc_score(y[outer_valid], oof[outer_valid])), "selected_epochs": chosen})
        del model
    threshold = threshold_from_development(y, oof)
    if not np.isfinite(threshold):
        raise ValueError("开发OOF没有形成有限阈值，请检查预测。")
    final_epochs = max(1, int(np.median(selected_epochs)))
    model, _, history = fit_optimized_epochs(settings, images, masks, y, np.arange(len(y)), final_epochs, SEED + 1000)
    torch.save(model.state_dict(), p["optimized"] / "final_model.pt")
    pd.DataFrame(history).to_csv(p["optimized"] / "final_history.csv", index=False)
    output = table[["mr", "label", "development_fold"]].copy()
    output["probability"] = oof
    output.to_csv(p["optimized"] / "development_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(p["optimized"] / "fold_metrics.csv", index=False)
    aucs = [row["auc"] for row in rows]
    metrics = {**classification_metrics(y, oof, threshold, n_bootstrap=0), "pooled_auc": float(roc_auc_score(y, oof)),
        "mean_fold_auc": float(np.mean(aucs)), "std_fold_auc": float(np.std(aucs)), "fold_aucs": aucs,
        "selected_epochs": selected_epochs, "final_epochs": final_epochs}
    dump_json(p["optimized"] / "development_metrics.json", metrics)
    if fingerprint != {"cohort_identity": cohort_identity(cfg), "cache_sha256": digest(p["cache"]), "code_contract": optimized_contract(cfg)}:
        raise ValueError("训练期间输入或实现发生变化，不锁定模型。")
    artifacts = {file.name: digest(file) for file in p["optimized"].iterdir() if file.is_file()}
    dump_json(p["optimized"] / "frozen_artifacts.json", {**fingerprint, "artifacts": artifacts})
    record(cfg, "optimized_training", {"settings": settings, "metrics": metrics})
    return metrics


def verify_optimized(cfg):
    p = paths(cfg)
    lock = json.loads((p["optimized"] / "frozen_artifacts.json").read_text(encoding="utf-8"))
    if lock["cohort_identity"] != cohort_identity(cfg) or lock["cache_sha256"] != digest(p["cache"]) or lock["code_contract"] != optimized_contract(cfg):
        raise ValueError("优化模型与当前患者、缓存或实现不匹配。")
    for name, checksum in lock["artifacts"].items():
        if digest(p["optimized"] / name) != checksum:
            raise ValueError(f"优化模型文件发生变化：{name}")
    settings = json.loads((p["optimized"] / "training_config.json").read_text(encoding="utf-8"))["settings"]
    if settings != cfg["optimized"]:
        raise ValueError("优化参数与训练时不一致；加载已有模型时使用原参数，调参请更换 experiment。")
    return settings


def load_optimized_model(cfg, checkpoint=None):
    settings = verify_optimized(cfg)
    model = OptimizedResNet(settings, pretrained=False).to(DEVICE)
    checkpoint = checkpoint or paths(cfg)["optimized"] / "final_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()
    return model


def predict_optimized_validation(cfg):
    table, images, masks, y = optimized_data(cfg, "validation")
    probability = optimized_predict(load_optimized_model(cfg), images, masks, cfg["optimized"]["batch_size"])
    meta = json.loads((paths(cfg)["optimized"] / "development_metrics.json").read_text(encoding="utf-8"))
    result = table[["mr", "label"]].copy()
    result["model"], result["probability"], result["threshold"] = "ResNet50_Optimized", probability, meta["threshold"]
    result["prediction"] = (probability >= meta["threshold"]).astype(int)
    result["protocol"] = "optimized_resnet_inner_epoch_selection"
    return result, pd.DataFrame([{"model": "ResNet50_Optimized", **classification_metrics(y, probability, meta["threshold"])}])
