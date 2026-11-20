"""Classification summaries; thresholds are supplied, never selected on validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample


def threshold_from_development(y, probability):
    fpr, tpr, thresholds = roc_curve(y, probability)
    return float(thresholds[np.argmax(tpr - fpr)])


def classification_metrics(y, probability, threshold, n_bootstrap=1000):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if len(y) != len(probability) or set(y) != {0, 1} or not np.isfinite(probability).all():
        raise ValueError("标签需包含两类，概率必须有限且与患者一一对应。")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("概率不在[0,1]。")
    tn, fp, fn, tp = confusion_matrix(y, probability >= threshold, labels=[0, 1]).ravel()
    result = {"n": len(y), "auc": float(roc_auc_score(y, probability)), "threshold": float(threshold),
              "accuracy": float((tp + tn) / len(y)), "sensitivity": float(tp / (tp + fn)),
              "specificity": float(tn / (tn + fp)), "brier": float(brier_score_loss(y, probability)),
              "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
    if n_bootstrap:
        values = []
        for seed in range(n_bootstrap):
            idx = resample(np.arange(len(y)), replace=True, stratify=y, random_state=seed)
            values.append(roc_auc_score(y[idx], probability[idx]))
        result.update(auc_ci_low=float(np.percentile(values, 2.5)), auc_ci_high=float(np.percentile(values, 97.5)))
    return result


def paired_comparison(a: pd.DataFrame, b: pd.DataFrame, n_bootstrap=1000):
    for frame in (a, b):
        if not frame["mr"].is_unique or set(frame["label"]) != {0, 1}:
            raise ValueError("比较需要唯一患者编号与两类结局。")
    if set(a["mr"]) != set(b["mr"]):
        raise ValueError("两组患者不完全一致；不静默取交集。")
    merged = a.merge(b, on="mr", suffixes=("_a", "_b"), validate="one_to_one")
    if not (merged.label_a == merged.label_b).all():
        raise ValueError("同一患者的标签不一致。")
    y, pa, pb = merged.label_a.to_numpy(), merged.probability_a.to_numpy(), merged.probability_b.to_numpy()
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():
        raise ValueError("预测包含非有限值。")
    differences = []
    for seed in range(n_bootstrap):
        idx = resample(np.arange(len(y)), replace=True, stratify=y, random_state=seed)
        differences.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = np.asarray(differences)
    return {"auc_difference": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "empirical_p": min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))}


def plot_predictions(predictions):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, table in predictions.groupby("model"):
        fpr, tpr, _ = roc_curve(table.label, table.probability)
        axes[0].plot(fpr, tpr, label=f"{name}: {roc_auc_score(table.label, table.probability):.3f}")
        truth, prob = calibration_curve(table.label, table.probability, n_bins=5, strategy="quantile")
        axes[1].plot(prob, truth, "o-", label=name)
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.legend(fontsize=8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Predicted probability", ylabel="Observed proportion", title="Calibration (no refitting)")
    fig.tight_layout()
    return fig


def run_validation_report(cfg, include_radiomics=True, include_automatic=False):
    from .original import predict_original_validation
    from .project import dump_json, fresh, paths, record
    if cfg.get("data_access") == "classification_cache_only" and (include_radiomics or include_automatic):
        raise ValueError("当前为分类缓存包，请关闭 INCLUDE_RADIOMICS 和 INCLUDE_AUTOMATIC_MASK_SUBSTITUTION。")
    p = paths(cfg)
    fresh(p["reports"] / "validation_predictions.csv", p["reports"] / "validation_metrics.csv")
    predictions, metrics = predict_original_validation(cfg)
    tables, summaries = [predictions], [metrics]
    if include_radiomics:
        from .radiomics import predict_radiomics_validation
        predicted, summary = predict_radiomics_validation(cfg)
        tables.append(predicted)
        summaries.append(summary)
    if include_automatic:
        from .segmentation import build_validation_mask_cache, evaluate_masks, locations, verify_validation_mask_cache
        automatic_cache = locations(cfg)["root"] / "validation_masks.npz"
        if not automatic_cache.exists():
            build_validation_mask_cache(cfg)
        verify_validation_mask_cache(cfg)
        predicted, summary = predict_original_validation(cfg, automatic_masks=automatic_cache)
        tables.append(predicted)
        summaries.append(summary)
        evaluate_masks(cfg, scope="validation")
    all_predictions = pd.concat(tables, ignore_index=True)
    all_metrics = pd.concat(summaries, ignore_index=True)
    p["reports"].mkdir(parents=True, exist_ok=True)
    all_predictions.to_csv(p["reports"] / "validation_predictions.csv", index=False)
    all_metrics.to_csv(p["reports"] / "validation_metrics.csv", index=False)
    groups = {name: table for name, table in all_predictions.groupby("model")}
    comparisons = []
    for name in groups:
        if name != "Clinical5":
            comparisons.append({"a": name, "b": "Clinical5", **paired_comparison(groups[name], groups["Clinical5"])})
    if "Radiomics" in groups:
        comparisons.append({"a": "ResNet50", "b": "Radiomics", **paired_comparison(groups["ResNet50"], groups["Radiomics"])})
    pd.DataFrame(comparisons).to_csv(p["reports"] / "paired_auc_comparisons.csv", index=False)
    figure = plot_predictions(all_predictions)
    figure.savefig(p["reports"] / "roc_calibration.png", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(figure)
    dump_json(p["reports"] / "report_notes.json", {
        "resnet": "原结构、训练、单模型及一次OOF晚期融合；没有新增校准或全流程嵌套改造。",
        "radiomics": "新增全三维ROI基线；覆盖范围与四层ResNet不同。" if include_radiomics else "未纳入本次报告。",
        "automatic": "人工掩模训练的原分类器替换输入敏感性实验；仍需人工取层。" if include_automatic else "未纳入本次报告。",
        "auc_p": "经验配对bootstrap；不是DeLong检验。"})
    record(cfg, "validation_report")
    return all_metrics, pd.DataFrame(comparisons)
