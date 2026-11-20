"""New explanation utilities; original model forward and weights remain untouched."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .original import configure, load_original_model
from .project import development, dump_json, fresh, paths
from .baseline import DEVICE


def choose_development_cases(cfg):
    p, dev = paths(cfg), development(cfg)
    oof = pd.read_csv(p["temporal"] / "development_oof.csv", dtype={"mr": str})
    table = dev.merge(oof[["mr", "dev_oof_prob"]], on="mr", validate="one_to_one")
    threshold = json.loads((p["temporal"] / "development_cv_metrics.json").read_text(encoding="utf-8"))["youden_threshold"]
    predicted = table.dev_oof_prob >= threshold
    table["category"] = np.select([(table.label == 1) & predicted, (table.label == 0) & ~predicted,
                                  (table.label == 0) & predicted], ["TP", "TN", "FP"], default="FN")
    # Fixed MR ordering, not manual selection of visually attractive maps.
    return table.sort_values("mr").groupby("category", sort=True).head(1)


def grad_cam(model, images, masks, target_class=1):
    import torch
    import torch.nn.functional as F
    if target_class not in (0, 1) or len(images) != 1:
        raise ValueError("当前解释入口每次一个患者，target_class为0或1。")
    captured = {}

    def feature_hook(_module, _inputs, output):
        captured["features"] = output
        output.retain_grad()

    def attention_hook(_module, _inputs, output):
        captured["attention"] = output

    # Original ResNet sequential: conv1,bn,relu,maxpool,layer1..layer4,avgpool.
    feature_handle = model.image_features[7].register_forward_hook(feature_hook)
    attention_handle = model.attention.register_forward_hook(attention_hook)
    model.eval()
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(images.requires_grad_(True), masks)
            (logits if target_class == 1 else -logits).sum().backward()
            feature = captured["features"]
            weights = feature.grad.mean(dim=(2, 3), keepdim=True)
            maps = torch.relu((weights * feature).sum(dim=1, keepdim=True))
            maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            maximum = maps.amax(dim=(1, 2), keepdim=True)
            maps = maps / maximum.clamp_min(1e-8)
            attention = torch.softmax(captured["attention"].squeeze(-1), dim=1)[0]
            return {"cam": maps.detach().cpu().numpy(), "attention": attention.detach().cpu().numpy(),
                    "probability": float(torch.sigmoid(logits)[0].detach().cpu())}
    finally:
        feature_handle.remove()
        attention_handle.remove()
        model.zero_grad(set_to_none=True)


def explain_development_cases(cfg, selected_mrs=None):
    import matplotlib.pyplot as plt
    import torch
    p, dev = paths(cfg), development(cfg)
    if selected_mrs:
        if not set(selected_mrs) <= set(dev.mr):
            raise ValueError("本入口只解释开发队列的折外患者。")
        selected = dev.loc[dev.mr.isin(selected_mrs)]
    else:
        selected = choose_development_cases(cfg)
    fresh(p["explain"] / "cases.csv")
    p["explain"].mkdir(parents=True, exist_ok=True)
    configure(cfg)
    rows = []
    with np.load(p["cache"], allow_pickle=False) as cache:
        for _, patient in selected.iterrows():
            fold = int(patient.development_fold)
            checkpoint = p["temporal"] / f"dev_fold_{fold}.pt"
            model = load_original_model(cfg, checkpoint)
            idx = int(patient.cache_index)
            images = cache["images"][idx].astype(np.float32)
            masks = cache["masks"][idx].astype(np.uint8)
            explanation = grad_cam(model, torch.from_numpy(images[None]).to(DEVICE),
                                   torch.from_numpy(masks[None]).to(DEVICE))
            np.savez_compressed(p["explain"] / f"{patient.mr}_cam.npz", **explanation)
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            for s in range(4):
                for c, title in enumerate(("T1-FLAIR", "T2", "DWI")):
                    axes[c, s].imshow(images[s, c], cmap="gray", vmin=0, vmax=1)
                    axes[c, s].set_title(f"{title} | slice {s + 1}")
                axes[3, s].imshow(images[s, 0], cmap="gray", vmin=0, vmax=1)
                axes[3, s].imshow(explanation["cam"][s], cmap="jet", alpha=.4, vmin=0, vmax=1)
                for label, color in enumerate(("cyan", "lime", "magenta")):
                    if masks[s, label].any() and not masks[s, label].all():
                        axes[3, s].contour(masks[s, label], levels=[.5], colors=[color], linewidths=.7)
                axes[3, s].set_title(f"CAM | attention={explanation['attention'][s]:.3f}")
            for ax in axes.flat:
                ax.axis("off")
            fig.suptitle(f"{patient.mr} | OOF fold={fold} | y={patient.label} | p={explanation['probability']:.3f}")
            fig.tight_layout()
            fig.savefig(p["explain"] / f"{patient.mr}.png", dpi=150)
            plt.close(fig)
            rows.append({"mr": patient.mr, "label": int(patient.label), "fold": fold,
                         "probability": explanation["probability"], **{f"slice_{i+1}_weight": float(w) for i, w in enumerate(explanation["attention"])}})
            del model
    table = pd.DataFrame(rows)
    table.to_csv(p["explain"] / "cases.csv", index=False)
    dump_json(p["explain"] / "method.json", {"target": "patient_positive_logit", "layer": "image_features[7]",
        "case_selection": "development fold model; first MR per TP/TN/FP/FN unless explicit list",
        "limitation": "原折验证标签用于早停；热图和切片权重不是分割结果或因果证明。"})
    return table
