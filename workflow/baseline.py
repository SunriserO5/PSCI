"""Physically copied baseline definitions; no imports from the old project code directory.

Only duplicate entry-point names and path globals are renamed. Algorithms stay unchanged.
Notebook cells contain these definitions literally; this file is a maintenance copy.
"""
from __future__ import annotations

import copy
import json
import random
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.ndimage import zoom
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode

SEED = 20260922
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 224
SLICES = 4
IMAGES = ("t1_flair.nii.gz", "t2.nii.gz", "dwi_b1000.nii.gz")
SINGLE_SEQUENCES = {"t1": 0, "t2": 1, "dwi": 2}
CACHE = OUT_DIR = TEMPORAL_OUT = TEMPORAL_SPLIT = None
PREP_BASE = PREP_OUT = ADAPTER_XLSX = None


def configure_baseline(cache, full_cv, temporal, adapter, patients_root=None):
    """Configure only input/output locations; do not alter the training recipe."""
    global CACHE, OUT_DIR, TEMPORAL_OUT, TEMPORAL_SPLIT
    global PREP_BASE, PREP_OUT, ADAPTER_XLSX
    CACHE, OUT_DIR, TEMPORAL_OUT = Path(cache), Path(full_cv), Path(temporal)
    TEMPORAL_SPLIT, ADAPTER_XLSX = TEMPORAL_OUT / "temporal_split.csv", Path(adapter)
    PREP_BASE, PREP_OUT = Path(patients_root) if patients_root else None, CACHE


# Copied from prep_resnet50_manual.py :: normalize_slices
def normalize_slices(vol: np.ndarray, zs: list[int]) -> np.ndarray:
    vals = vol[:, :, zs]
    lo, hi = np.percentile(vals, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)


# Copied from prep_resnet50_manual.py :: load_volume
def load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


# Copied from prep_resnet50_manual.py :: parse_max_layer
def parse_max_layer(value) -> int | None:
    text = str(value).strip()
    if text in {"无", "NONE", "NA", ""}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


# Copied from prep_resnet50_manual.py :: main
def prepare_cache_main() -> None:
    df = pd.read_excel(ADAPTER_XLSX, sheet_name="Sheet1", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    mr_series = df.iloc[:, 0].astype(str).str.strip().str.upper()
    max_series = df["脑室体素最大层"].map(parse_max_layer)
    moca = pd.to_numeric(df["MOCA"], errors="coerce")

    patient_dirs = {}
    for patient_dir in PREP_BASE.iterdir():
        if not patient_dir.is_dir():
            continue
        match = re.search(r"(MR\d+)$", patient_dir.name.upper())
        if match:
            patient_dirs[match.group(1)] = patient_dir

    images, masks, labels, mrs, max_layers = [], [], [], [], []
    skipped = []

    for idx in range(len(df)):
        if pd.isna(max_series[idx]) or pd.isna(moca[idx]):
            continue
        mr_raw = mr_series[idx]
        digits_match = re.search(r"(\d+)$", mr_raw)
        if not digits_match:
            skipped.append((mr_raw, "no MR number"))
            continue
        mr = f"MR{digits_match.group(1)}"
        patient_dir = patient_dirs.get(mr)
        if patient_dir is None:
            skipped.append((mr, "patient directory not found"))
            continue
        reg = patient_dir / "配准图像"
        mask_path = reg / "Untitled.nii.gz"
        if not mask_path.exists():
            skipped.append((mr, "missing Untitled.nii.gz"))
            continue

        try:
            t1 = load_volume(reg / IMAGES[0])
            t2 = load_volume(reg / IMAGES[1])
            dwi = load_volume(reg / IMAGES[2])
            label_vol = np.asarray(nib.load(str(mask_path)).dataobj, dtype=np.uint8)
        except Exception as exc:
            skipped.append((mr, type(exc).__name__))
            continue

        z0 = int(max_series[idx]) - 1
        zs = [z0, z0 + 1, z0 + 2, z0 + 3]
        if zs[-1] >= t1.shape[2]:
            skipped.append((mr, f"slice overflow: {zs}, shape={t1.shape}"))
            continue

        norm_vols = [
            normalize_slices(vol, zs) for vol in (t1, t2, dwi)
        ]
        h, w = t1.shape[0], t1.shape[1]
        patient_images = np.empty((SLICES, 3, SIZE, SIZE), dtype=np.float16)
        patient_masks = np.zeros((SLICES, 3, SIZE, SIZE), dtype=np.uint8)

        for s, z in enumerate(zs):
            for c, vol in enumerate(norm_vols):
                patient_images[s, c] = zoom(
                    vol[:, :, s],
                    (SIZE / h, SIZE / w),
                    order=1,
                    prefilter=False,
                ).astype(np.float16)

            for label_value in range(1, 4):
                patient_masks[s, label_value - 1] = zoom(
                    (label_vol[:, :, z] == label_value).astype(np.uint8),
                    (SIZE / h, SIZE / w),
                    order=0,
                    prefilter=False,
                )

        images.append(patient_images)
        masks.append(patient_masks)
        labels.append(int(moca[idx] <= 22))
        mrs.append(mr)
        max_layers.append(int(max_series[idx]))

        if len(images) % 40 == 0:
            print(f"processed {len(images)} patients", flush=True)

    images_arr = np.stack(images)
    masks_arr = np.stack(masks)
    labels_arr = np.asarray(labels, dtype=np.int64)
    max_layers_arr = np.asarray(max_layers, dtype=np.int16)

    np.savez_compressed(
        PREP_OUT,
        images=images_arr,
        masks=masks_arr,
        labels=labels_arr,
        mrs=np.asarray(mrs),
        max_layers=max_layers_arr,
    )

    print("\npatients", len(images))
    print("images", images_arr.shape, images_arr.dtype)
    print("masks", masks_arr.shape, masks_arr.dtype)
    print("labels", int(labels_arr.sum()), "positive /", int((1 - labels_arr).sum()), "negative")
    print("skipped", skipped)
    print("wrote", PREP_OUT)


# Copied from prepare_temporal_validation.py :: parse_time
def parse_time(value) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(str(value).strip(), format="%Y.%m.%d", errors="coerce")


# Copied from prepare_temporal_validation.py :: format_median_iqr
def format_median_iqr(series: pd.Series) -> str:
    return f"{series.median():.2f} ({series.quantile(0.25):.2f}-{series.quantile(0.75):.2f})"


# Copied from prepare_temporal_validation.py :: smd_continuous
def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2.0)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# Copied from prepare_temporal_validation.py :: smd_binary
def smd_binary(a: pd.Series, b: pd.Series) -> float:
    p1 = float(a.mean())
    p2 = float(b.mean())
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)
    if denom == 0:
        return 0.0
    return float((p1 - p2) / denom)


# Copied from prepare_temporal_validation.py :: main
def prepare_temporal_main() -> None:
    TEMPORAL_OUT.mkdir(parents=True, exist_ok=True)
    excel = pd.read_excel(ADAPTER_XLSX, sheet_name="Sheet1", engine="openpyxl")
    excel.columns = [str(c).strip() for c in excel.columns]
    excel["mr"] = excel.iloc[:, 1].astype(str).str.strip().str.upper()
    excel["check_time"] = excel["检查时间"].map(parse_time)

    cache = np.load(CACHE, allow_pickle=True)
    cache_mrs = np.asarray([str(x).upper() for x in cache["mrs"]])
    cohort = excel[excel["mr"].isin(set(cache_mrs))].copy()
    assert len(cohort) == 206 and cohort["mr"].is_unique
    cohort = cohort.sort_values(["check_time", "mr"], kind="mergesort").reset_index(drop=True)
    assert cohort["check_time"].notna().all()

    # All patients on the 70% boundary date stay in development.
    boundary_date = cohort.loc[143, "check_time"]
    cohort["split"] = np.where(cohort["check_time"] <= boundary_date, "development", "validation")
    split_df = cohort[["mr", "check_time", "split"]].rename(columns={"check_time": "check_time"})
    split_df.to_csv(TEMPORAL_OUT / "temporal_split.csv", index=False)

    dev = cohort[cohort["split"] == "development"].copy()
    val = cohort[cohort["split"] == "validation"].copy()
    print("split", len(dev), len(val), "boundary", boundary_date.date())
    print("outcome", dev["MOCA"].le(22).sum(), val["MOCA"].le(22).sum())

    dev_idx = {mr: i for i, mr in enumerate(cache_mrs)}
    lesion = np.asarray(cache["masks"], dtype=np.int32)
    for df in (dev, val):
        idx = np.asarray([dev_idx[mr] for mr in df["mr"]], dtype=int)
        for label in (1, 2, 3):
            df[f"lesion_{label}_vox"] = lesion[idx, :, label - 1, :, :].sum(axis=(1, 2, 3))

    dev["male"] = (dev["性别"] == "男").astype(int)
    val["male"] = (val["性别"] == "男").astype(int)
    dev["impaired"] = (dev["MOCA"] <= 22).astype(int)
    val["impaired"] = (val["MOCA"] <= 22).astype(int)

    rows = []
    for name in ["age", "fazekas", "education", "nihss", "moca"]:
        col_map = {
            "age": "年龄",
            "fazekas": "Fazekas",
            "education": "受教育年限",
            "nihss": "NIHSS",
            "moca": "MOCA",
        }
        a = pd.to_numeric(dev[col_map[name]], errors="coerce")
        b = pd.to_numeric(val[col_map[name]], errors="coerce")
        a_test = a.dropna()
        b_test = b.dropna()
        rows.append(
            {
                "variable": name,
                "development": format_median_iqr(a),
                "validation": format_median_iqr(b),
                "p": stats.mannwhitneyu(a_test, b_test, alternative="two-sided").pvalue,
                "smd": smd_continuous(a_test, b_test),
            }
        )

    for name, a, b in [
        ("male", dev["male"], val["male"]),
        ("impaired", dev["impaired"], val["impaired"]),
    ]:
        table = pd.crosstab(
            pd.Series(np.r_[np.zeros(len(a)), np.ones(len(b))], dtype=int),
            pd.Series(np.r_[a.values, b.values], dtype=int),
        )
        if table.shape == (2, 2):
            p = stats.chi2_contingency(table, correction=False).pvalue
        else:
            p = np.nan
        rows.append(
            {
                "variable": name,
                "development": f"{int(a.sum())}/{len(a)}",
                "validation": f"{int(b.sum())}/{len(b)}",
                "p": p,
                "smd": smd_binary(a, b),
            }
        )

    for label in (1, 2, 3):
        a = dev[f"lesion_{label}_vox"]
        b = val[f"lesion_{label}_vox"]
        rows.append(
            {
                "variable": f"lesion_{label}_vox",
                "development": format_median_iqr(a),
                "validation": format_median_iqr(b),
                "p": stats.mannwhitneyu(a, b, alternative="two-sided").pvalue,
                "smd": smd_continuous(a, b),
            }
        )

    baseline = pd.DataFrame(rows)
    baseline["p"] = baseline["p"].map(lambda v: f"{v:.4f}")
    baseline["smd"] = baseline["smd"].map(lambda v: f"{v:.3f}")
    baseline.to_csv(TEMPORAL_OUT / "baseline_comparison.csv", index=False)
    print(baseline.to_string(index=False))

    with open(TEMPORAL_OUT / "temporal_split_summary.md", "w", encoding="utf-8") as f:
        f.write("# 时间分组与基线比较\n\n")
        f.write(f"- 开发队列：{len(dev)} 例\n")
        f.write(f"- 独立验证队列：{len(val)} 例\n")
        f.write(f"- 时间边界：{boundary_date.date()}\n\n")
        f.write(baseline.to_string(index=False))

    print("wrote", TEMPORAL_OUT / "temporal_split.csv", TEMPORAL_OUT / "baseline_comparison.csv")


# Copied from train_resnet50_manual_cv.py :: seed_everything
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Copied from train_resnet50_manual_cv.py :: MaskEncoder
class MaskEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.float())


# Copied from train_resnet50_manual_cv.py :: DualBranchModel
class DualBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        self.mask_features = MaskEncoder()
        self.attention = nn.Sequential(
            nn.Linear(2304, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2304),
            nn.Dropout(0.4),
            nn.Linear(2304, 1),
        )

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = images.shape
        im = images.view(B * S, C, H, W)
        image_feat = self.image_features(im)
        image_feat = torch.flatten(image_feat, 1)
        image_feat = image_feat.view(B, S, -1)

        mask = masks.view(B * S, 3, H, W)
        mask_feat = self.mask_features(mask)
        mask_feat = mask_feat.view(B, S, -1)

        feat = torch.cat([image_feat, mask_feat], dim=-1)
        attn = self.attention(feat).squeeze(-1)
        attn = torch.softmax(attn, dim=1)
        pooled = (feat * attn.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


# Copied from train_resnet50_manual_cv.py :: augment_batch
def augment_batch(images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out_images = images.clone()
    out_masks = masks.clone()
    B, S = images.shape[:2]
    for b in range(B):
        angle = random.uniform(-6.0, 6.0)
        translate = (random.uniform(-6.0, 6.0), random.uniform(-6.0, 6.0))
        scale = random.uniform(0.96, 1.04)
        for s in range(S):
            out_images[b, s] = affine(
                images[b, s],
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            out_masks[b, s] = affine(
                masks[b, s].float(),
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).byte()

    jitter = 0.9 + 0.2 * torch.rand(B, 1, 1, 1, 1, device=out_images.device)
    out_images = out_images * jitter
    if random.random() < 0.5:
        out_images = out_images + torch.randn_like(out_images) * 0.008
    out_images = out_images.clamp(0.0, 1.0)
    return out_images, out_masks


# Copied from train_resnet50_manual_cv.py :: predict_proba
@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: torch.Tensor,
    M: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(DEVICE)
        mb = M[i : i + batch_size].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            outs.append(torch.sigmoid(model(xb, mb)).float().cpu().numpy())
    return np.concatenate(outs)


# Copied from train_resnet50_manual_cv.py :: train_fold
def train_full_fold(
    Xtr: torch.Tensor,
    Mtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    Mva: torch.Tensor,
    yva: torch.Tensor,
    fold: int,
) -> Model:
    model = DualBranchModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_features.parameters(), "lr": 1e-4},
            {
                "params": list(model.mask_features.parameters())
                + list(model.attention.parameters())
                + list(model.head.parameters()),
                "lr": 5e-4,
            },
        ],
        weight_decay=1e-4,
    )
    n_pos = int((ytr > 0.5).sum())
    n_neg = len(ytr) - n_pos
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 0.5)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_auc, best_state, patience = 0.0, None, 0

    for epoch in range(1, 31):
        t0 = time.time()
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
        epoch_loss /= len(Xtr)

        val_pred = predict_proba(model, Xva, Mva)
        val_auc = float(roc_auc_score(yva.numpy(), val_pred))
        if val_auc > best_auc:
            best_auc, best_state, patience = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if epoch % 3 == 0 or patience >= 8:
            print(
                f"fold {fold} epoch {epoch:02d} loss={epoch_loss:.4f} "
                f"val_auc={val_auc:.4f} best={best_auc:.4f} time={time.time()-t0:.1f}s",
                flush=True,
            )
        if patience >= 8:
            break

    model.load_state_dict(best_state)
    return model


# Copied from train_resnet50_manual_cv.py :: main
def train_full_cv_main() -> None:
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(CACHE)
    images = torch.from_numpy(data["images"].astype(np.float32))
    masks = torch.from_numpy(data["masks"].astype(np.uint8))
    labels = data["labels"].astype(np.int64)
    mrs = [str(x) for x in data["mrs"]]

    print("data", images.shape, masks.shape, "pos", int(labels.sum()), "device", DEVICE, flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(labels), dtype=np.float32)
    fold_aucs = []
    fold_start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        ytr = torch.from_numpy(labels[tr_idx].astype(np.float32))
        yva = torch.from_numpy(labels[va_idx].astype(np.float32))
        model = train_full_fold(
            images[tr_idx],
            masks[tr_idx],
            ytr,
            images[va_idx],
            masks[va_idx],
            yva,
            fold,
        )
        pred = predict_proba(model, images[va_idx], masks[va_idx])
        oof[va_idx] = pred
        fold_auc = roc_auc_score(labels[va_idx], pred)
        fold_aucs.append(fold_auc)
        torch.save(model.state_dict(), OUT_DIR / f"fold_{fold}.pt")
        print(f"fold {fold} AUC={fold_auc:.4f}", flush=True)

    pooled_auc = roc_auc_score(labels, oof)
    fpr, tpr, thr = roc_curve(labels, oof)
    youden = thr[np.argmax(tpr - fpr)]
    pred = (oof >= youden).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    acc = (tp + tn) / len(labels)

    print("\n" + "=" * 60)
    print(f"fold AUCs={[round(a, 4) for a in fold_aucs]}")
    print(f"mean AUC={np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"pooled AUC={pooled_auc:.4f} acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} youden={youden:.4f}")
    print(f"elapsed={time.time()-fold_start:.1f}s")
    print("=" * 60)

    pd.DataFrame(
        {
            "mr": mrs,
            "label": labels,
            "resnet50_prob": oof,
        }
    ).to_csv(OUT_DIR / "resnet50_manual_oof.csv", index=False)
    print("saved OOF to", OUT_DIR / "resnet50_manual_oof.csv")


# Copied from train_resnet50_temporal_development.py :: make_optimizer
def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.image_features.parameters(), "lr": 1e-4},
            {
                "params": list(model.mask_features.parameters())
                + list(model.attention.parameters())
                + list(model.head.parameters()),
                "lr": 5e-4,
            },
        ],
        weight_decay=1e-4,
    )


# Copied from train_resnet50_temporal_development.py :: make_criterion
def make_criterion(y: np.ndarray | torch.Tensor) -> nn.BCEWithLogitsLoss:
    y_np = np.asarray(y)
    n_pos = int((y_np > 0.5).sum())
    n_neg = len(y_np) - n_pos
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 0.5)], device=DEVICE)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# Copied from train_resnet50_temporal_development.py :: train_fold
def train_development_fold(
    Xtr: torch.Tensor,
    Mtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    Mva: torch.Tensor,
    yva: torch.Tensor,
    fold: int,
) -> tuple[DualBranchModel, int]:
    model = DualBranchModel().to(DEVICE)
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


# Copied from train_resnet50_temporal_development.py :: train_final
def train_final(
    X: torch.Tensor,
    M: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
) -> DualBranchModel:
    seed_everything(SEED + 100)
    model = DualBranchModel().to(DEVICE)
    optimizer = make_optimizer(model)
    criterion = make_criterion(y.numpy())
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        for i in range(0, len(X), BATCH_SIZE):
            bi = perm[i : i + BATCH_SIZE]
            xb, mb = X[bi], M[bi]
            xb, mb = augment_batch(xb, mb)
            xb = xb.to(DEVICE)
            mb = mb.to(DEVICE)
            yb = y[bi].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                loss = criterion(model(xb, mb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach()) * len(bi)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"final epoch {epoch:02d}/{epochs:02d} loss={epoch_loss/len(X):.4f}", flush=True)

    return model


# Copied from train_resnet50_temporal_development.py :: main
def train_development_main() -> None:
    TEMPORAL_OUT.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE, allow_pickle=True)
    cache_mrs = np.asarray([str(x).upper() for x in cache["mrs"]])
    cache_index = {mr: i for i, mr in enumerate(cache_mrs)}
    split = pd.read_csv(TEMPORAL_SPLIT)
    split["mr"] = split["mr"].astype(str).str.upper()
    dev = split[split["split"] == "development"].reset_index(drop=True)
    idx = np.asarray([cache_index[mr] for mr in dev["mr"]], dtype=int)

    X = torch.from_numpy(cache["images"][idx].astype(np.float32))
    M = torch.from_numpy(cache["masks"][idx].astype(np.uint8))
    y = cache["labels"][idx].astype(np.int64)
    mrs = dev["mr"].tolist()
    print("development", X.shape, "pos", int(y.sum()), flush=True)

    seed_everything(SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []
    best_epochs = []
    start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        ytr = torch.from_numpy(y[tr_idx].astype(np.float32))
        yva = torch.from_numpy(y[va_idx].astype(np.float32))
        model, best_epoch = train_development_fold(
            X[tr_idx], M[tr_idx], ytr, X[va_idx], M[va_idx], yva, fold
        )
        pred = predict_proba(model, X[va_idx], M[va_idx])
        oof[va_idx] = pred
        fold_aucs.append(roc_auc_score(y[va_idx], pred))
        best_epochs.append(best_epoch)
        torch.save(model.state_dict(), TEMPORAL_OUT / f"dev_fold_{fold}.pt")
        print(f"development fold {fold} AUC={fold_aucs[-1]:.4f} best_epoch={best_epoch}", flush=True)

    fpr, tpr, thr = roc_curve(y, oof)
    youden = float(thr[np.argmax(tpr - fpr)])
    pooled_auc = roc_auc_score(y, oof)
    pred = (oof >= youden).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    metrics = {
        "n": len(y),
        "n_positive": int(y.sum()),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "pooled_auc": float(pooled_auc),
        "youden_threshold": youden,
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "accuracy": float((tp + tn) / len(y)),
        "best_epochs": [int(x) for x in best_epochs],
    }
    with open(TEMPORAL_OUT / "development_cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"mr": mrs, "label": y, "dev_oof_prob": oof}).to_csv(
        TEMPORAL_OUT / "development_oof.csv", index=False
    )
    print("development CV", metrics, flush=True)

    final_epochs = max(1, int(np.median(best_epochs)))
    final_model = train_final(X, M, torch.from_numpy(y.astype(np.float32)), final_epochs)
    torch.save(final_model.state_dict(), TEMPORAL_OUT / "final_development_model.pt")
    print("saved final model, epochs", final_epochs, "elapsed", time.time() - start, flush=True)


# Copied from fusion_resnet50_manual_clinical5.py :: make_pipe
def make_pipe(C: float) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=3000, solver="liblinear")),
        ]
    )


# Copied from fusion_resnet50_manual_clinical5.py :: fit_fold_with_inner_cv
def fit_fold_with_inner_cv(Xtr: np.ndarray, ytr: np.ndarray) -> Pipeline:
    best_auc, best_C = -1.0, 1.0
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
        aucs = []
        for itr, iva in inner.split(Xtr, ytr):
            pipe = make_pipe(C)
            pipe.fit(Xtr[itr], ytr[itr])
            proba = pipe.predict_proba(Xtr[iva])[:, 1]
            if len(np.unique(ytr[iva])) > 1:
                aucs.append(roc_auc_score(ytr[iva], proba))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_C = np.mean(aucs), C
    pipe = make_pipe(best_C)
    pipe.fit(Xtr, ytr)
    return pipe


# Copied from fusion_resnet50_manual_clinical5.py :: evaluate
def evaluate(name: str, X: np.ndarray, y: np.ndarray, skf: StratifiedKFold) -> pd.Series:
    oof = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        pipe = fit_fold_with_inner_cv(X[tr], y[tr])
        oof[va] = pipe.predict_proba(X[va])[:, 1]
        fold_aucs.append(roc_auc_score(y[va], oof[va]))

    pooled = roc_auc_score(y, oof)
    fpr, tpr, thr = roc_curve(y, oof)
    youden = thr[np.argmax(tpr - fpr)]
    pred = (oof >= youden).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    result = pd.Series(
        {
            "model": name,
            "fold_aucs": [round(a, 4) for a in fold_aucs],
            "mean_auc": float(np.mean(fold_aucs)),
            "std_auc": float(np.std(fold_aucs)),
            "pooled_auc": float(pooled),
            "accuracy": float((tp + tn) / len(y)),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "youden": float(youden),
        }
    )
    print(result.to_string())
    return result, oof


# Copied from fusion_resnet50_manual_clinical5.py :: direct_evaluate
def direct_evaluate(name: str, prob: np.ndarray, y: np.ndarray) -> pd.Series:
    pooled = roc_auc_score(y, prob)
    fpr, tpr, thr = roc_curve(y, prob)
    youden = thr[np.argmax(tpr - fpr)]
    pred = (prob >= youden).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return pd.Series(
        {
            "model": name,
            "fold_aucs": np.nan,
            "mean_auc": np.nan,
            "std_auc": np.nan,
            "pooled_auc": float(pooled),
            "accuracy": float((tp + tn) / len(y)),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "youden": float(youden),
        }
    )


# Copied from train_resnet50_single_sequence_manual_cv.py :: SingleSequenceDualBranchModel
class SingleSequenceDualBranchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        old_conv = backbone.conv1
        conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = conv1
        self.image_features = nn.Sequential(*list(backbone.children())[:-1])
        self.mask_features = MaskEncoder()
        self.attention = nn.Sequential(
            nn.Linear(2304, 256),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2304),
            nn.Dropout(0.4),
            nn.Linear(2304, 1),
        )

    def forward(self, images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        B, S, C, H, W = images.shape
        im = images.view(B * S, C, H, W)
        image_feat = self.image_features(im)
        image_feat = torch.flatten(image_feat, 1)
        image_feat = image_feat.view(B, S, -1)

        mask = masks.view(B * S, 3, H, W)
        mask_feat = self.mask_features(mask)
        mask_feat = mask_feat.view(B, S, -1)

        feat = torch.cat([image_feat, mask_feat], dim=-1)
        attn = self.attention(feat).squeeze(-1)
        attn = torch.softmax(attn, dim=1)
        pooled = (feat * attn.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


# Copied from train_resnet50_single_sequence_manual_cv.py :: train_fold
def train_single_sequence_fold(
    Xtr: torch.Tensor,
    Mtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    Mva: torch.Tensor,
    yva: torch.Tensor,
    fold: int,
) -> SingleSequenceDualBranchModel:
    model = SingleSequenceDualBranchModel().to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_features.parameters(), "lr": 1e-4},
            {
                "params": list(model.mask_features.parameters())
                + list(model.attention.parameters())
                + list(model.head.parameters()),
                "lr": 5e-4,
            },
        ],
        weight_decay=1e-4,
    )
    n_pos = int((ytr > 0.5).sum())
    n_neg = len(ytr) - n_pos
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 0.5)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    best_auc, best_state, patience = 0.0, None, 0

    for epoch in range(1, 31):
        t0 = time.time()
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
        epoch_loss /= len(Xtr)

        val_pred = predict_proba(model, Xva, Mva)
        val_auc = float(roc_auc_score(yva.numpy(), val_pred))
        if val_auc > best_auc:
            best_auc, best_state, patience = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if epoch % 3 == 0 or patience >= 8:
            print(
                f"fold {fold} epoch {epoch:02d} loss={epoch_loss:.4f} "
                f"val_auc={val_auc:.4f} best={best_auc:.4f} time={time.time()-t0:.1f}s",
                flush=True,
            )
        if patience >= 8:
            break

    model.load_state_dict(best_state)
    return model
