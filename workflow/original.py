"""Integration of physically copied ResNet definitions; no external code directory."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .project import CLINICAL, SEED, development, digest, dump_json, fresh, paths, read_cohort, record, source_snapshot, cohort_identity, definition_contract, portable_name
from .reporting import classification_metrics, threshold_from_development
from .baseline import (
    DEVICE, DualBranchModel, SINGLE_SEQUENCES, configure_baseline, evaluate, fit_fold_with_inner_cv,
    predict_proba, seed_everything, train_development_main, train_full_cv_main, train_single_sequence_fold,
)


def configure(cfg):
    p = paths(cfg)
    configure_baseline(p["cache"], p["full_cv"], p["temporal"], p["adapter"], cfg.get("patients_root"))
    return p


def run_resnet_training(cfg, scope="development"):
    p = configure(cfg)
    if scope == "development":
        fresh(p["temporal"] / "development_oof.csv", p["temporal"] / "final_development_model.pt")
        train_development_main()
    elif scope == "all_legacy":
        fresh(p["full_cv"])
        train_full_cv_main()
    else:
        raise ValueError("scope应为development或all_legacy。")
    record(cfg, "resnet_" + scope, {"architecture": "original DualBranchModel", "calibration": "none",
           "validation_note": "保留原验证折选最佳epoch的方法；未声称全流程嵌套无偏。"})


def fit_original_clinical_and_fusion(cfg):
    """Move the development-only fitting from the original evaluator into notebook02."""
    import joblib
    from sklearn.model_selection import StratifiedKFold
    p, dev = paths(cfg), development(cfg)
    fresh(p["temporal"] / "clinical_fusion.joblib", p["temporal"] / "clinical_fusion_meta.json")
    oof = pd.read_csv(p["temporal"] / "development_oof.csv", dtype={"mr": str})
    if oof.mr.tolist() != dev.mr.tolist() or not np.array_equal(oof.label, dev.label):
        raise ValueError("影像OOF与开发队列患者顺序或标签不一致。")
    y = dev.label.to_numpy(dtype=int)
    clinical = dev[CLINICAL].to_numpy(dtype=np.float32)
    fused = np.column_stack([oof.dev_oof_prob.to_numpy(dtype=float), clinical])
    models, metadata, outputs = {}, {}, dev[["mr", "label", "development_fold"]].copy()
    for name, X in (("Clinical5", clinical), ("ResNet50_Clinical5", fused)):
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        result, prob = evaluate(name, X, y, cv)
        models[name] = fit_fold_with_inner_cv(X, y)
        threshold = threshold_from_development(y, prob)
        if not np.isfinite(threshold):
            raise ValueError(f"{name}没有有限Youden阈值；请检查开发结果。")
        metadata[name] = {"threshold": threshold, "mean_fold_auc": float(result["mean_auc"]),
                          "pooled_auc": float(result["pooled_auc"]), "features": CLINICAL,
                          "protocol": "original_one_pass_oof_fusion_not_fully_nested"}
        outputs[name + "_prob"] = prob
    joblib.dump(models, p["temporal"] / "clinical_fusion.joblib")
    dump_json(p["temporal"] / "clinical_fusion_meta.json", metadata)
    outputs.to_csv(p["temporal"] / "clinical_fusion_development_oof.csv", index=False)
    record(cfg, "clinical_fusion", {"note": "保留原逻辑回归及一次OOF融合，不新增校准或融合方法。"})
    return metadata


def lock_original_models(cfg):
    p = paths(cfg)
    files = [p["temporal"] / name for name in ("final_development_model.pt", "development_cv_metrics.json",
             "clinical_fusion.joblib", "clinical_fusion_meta.json")]
    files += [p["cohort"], p["temporal"] / "temporal_split.csv"]
    lock = p["temporal"] / "frozen_artifacts.json"
    fresh(lock)
    mapping = {f.relative_to(p["run"]).as_posix(): digest(f) for f in files}
    dump_json(lock, {"artifacts": mapping, "notebook_definitions_sha256": source_snapshot(cfg), "calibration": "none",
                     "cache_sha256": digest(p["cache"]), "baseline_contract": baseline_contract(cfg)})
    return lock


def verify_lock(cfg):
    p = paths(cfg)
    metadata = json.loads((p["temporal"] / "frozen_artifacts.json").read_text(encoding="utf-8"))
    if digest(p["cache"]) != metadata["cache_sha256"]:
        raise ValueError("影像缓存与开发锁定版本不一致。")
    for name, expected in metadata["artifacts"].items():
        if digest(p["run"] / portable_name(name)) != expected:
            raise ValueError(f"固定产物发生变化：{name}")
    contract = baseline_contract(cfg)
    expected = metadata.get("baseline_contract", ORIGINAL_INFERENCE_CONTRACT)
    if contract != expected:
        raise ValueError("原基线的网络或推理定义发生变化，请使用对应模型版本。")


def load_original_model(cfg, checkpoint=None):
    import torch
    configure(cfg)
    model = DualBranchModel().to(DEVICE)
    checkpoint = checkpoint or paths(cfg)["temporal"] / "final_development_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True), strict=True)
    model.eval()
    return model


def predict_original_validation(cfg, automatic_masks=None):
    import joblib
    import torch
    verify_lock(cfg)
    p = configure(cfg)
    val = read_cohort(cfg).query("split == 'validation'").reset_index(drop=True)
    idx = val.cache_index.to_numpy(dtype=int)
    with np.load(p["cache"], allow_pickle=False) as cache:
        X = torch.from_numpy(cache["images"][idx].astype(np.float32))
        M = torch.from_numpy(cache["masks"][idx].astype(np.uint8))
    suffix = ""
    if automatic_masks is not None:
        with np.load(automatic_masks, allow_pickle=False) as predicted:
            if predicted["mrs"].astype(str).tolist() != val.mr.tolist() or predicted["masks"].shape != tuple(M.shape):
                raise ValueError("自动掩模患者顺序或形状不匹配。")
            M = torch.from_numpy(predicted["masks"].astype(np.uint8))
        suffix = "_AutoMask_Substitution"
    probability = predict_proba(load_original_model(cfg), X, M)
    dl_metrics = json.loads((p["temporal"] / "development_cv_metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((p["temporal"] / "clinical_fusion_meta.json").read_text(encoding="utf-8"))
    models = joblib.load(p["temporal"] / "clinical_fusion.joblib")
    clinical = val[CLINICAL].to_numpy(dtype=np.float32)
    probabilities = {"ResNet50" + suffix: (probability, dl_metrics["youden_threshold"]),
        "ResNet50_Clinical5" + suffix: (models["ResNet50_Clinical5"].predict_proba(np.column_stack([probability, clinical]))[:, 1], meta["ResNet50_Clinical5"]["threshold"])}
    if not suffix:
        probabilities["Clinical5"] = (models["Clinical5"].predict_proba(clinical)[:, 1], meta["Clinical5"]["threshold"])
    predictions, summaries = [], []
    for name, (prob, threshold) in probabilities.items():
        table = val[["mr", "label"]].copy()
        table["model"], table["probability"], table["threshold"] = name, prob, threshold
        table["prediction"] = (prob >= threshold).astype(int)
        table["protocol"] = "manual_model_auto_mask_substitution" if suffix else "original_baseline"
        predictions.append(table)
        summaries.append({"model": name, **classification_metrics(val.label, prob, threshold)})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def run_single_sequence_ablation(cfg):
    """Same original single-sequence model and train_fold, restricted to development."""
    import torch
    from sklearn.metrics import roc_auc_score
    p = configure(cfg)
    fresh(p["single"])
    dev = development(cfg)
    indices = dev.cache_index.to_numpy(dtype=int)
    with np.load(p["cache"], allow_pickle=False) as cache:
        X = torch.from_numpy(cache["images"][indices].astype(np.float32))
        M = torch.from_numpy(cache["masks"][indices].astype(np.uint8))
    y = dev.label.to_numpy(dtype=int)
    summaries = []
    for sequence, channel in SINGLE_SEQUENCES.items():
        seed_everything(SEED)
        output = p["single"] / sequence
        output.mkdir(parents=True)
        x = X[:, :, channel:channel + 1].contiguous()
        oof, aucs = np.zeros(len(y), dtype=np.float32), []
        for fold in range(1, 6):
            va = np.flatnonzero(dev.development_fold.to_numpy() == fold)
            tr = np.flatnonzero(dev.development_fold.to_numpy() != fold)
            model = train_single_sequence_fold(x[tr], M[tr], torch.tensor(y[tr], dtype=torch.float32),
                                      x[va], M[va], torch.tensor(y[va], dtype=torch.float32), fold)
            oof[va] = predict_proba(model, x[va], M[va])
            aucs.append(float(roc_auc_score(y[va], oof[va])))
            torch.save(model.state_dict(), output / f"fold_{fold}.pt")
        threshold = threshold_from_development(y, oof)
        table = dev[["mr", "label", "development_fold"]].copy()
        table["probability"] = oof
        table.to_csv(output / "development_oof.csv", index=False)
        summaries.append({"sequence": sequence, "mean_fold_auc": float(np.mean(aucs)),
                          **classification_metrics(y, oof, threshold, n_bootstrap=0)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(p["single"] / "summary.csv", index=False)
    record(cfg, "single_sequence", {"scope": "development", "manual_mask_channels": 3})
    return summary


ORIGINAL_INFERENCE_CONTRACT = {'MaskEncoder': 'b0a35f2673a19f24a5f492a757a65894b572a9819996d2aabb9572cc89b2de04', 'DualBranchModel': '75a158f454de456d8b6e452309c35eb101124771da83567a475d8af2e386bc95', 'predict_proba': '94646b415f42d60d4a16416a05f26afed9656dd0fba15b09323c50f226898663'}


def baseline_contract(cfg):
    return definition_contract(cfg, "02", list(ORIGINAL_INFERENCE_CONTRACT))
