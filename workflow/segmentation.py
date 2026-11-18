"""New nnU-Net dataset/command integration and native-space evaluation."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .project import SEQUENCES, digest, dump_json, fresh, paths, read_cohort, record, require_native_images


def locations(cfg):
    root = paths(cfg)["segmentation"]
    dataset_id = int(cfg["segmentation"]["dataset_id"])
    if not 100 <= dataset_id <= 999:
        raise ValueError("本项目使用100～999的分割数据集ID。")
    name = f"Dataset{dataset_id:03d}_PSCI"
    return {"root": root, "name": name, "raw": root / "nnUNet_raw" / name,
            "preprocessed": root / "nnUNet_preprocessed" / name,
            "results": root / "nnUNet_results", "validation": root / "validation_predictions",
            "oof": root / "development_oof_predictions"}


def _volumes(patient, predicted_path=None):
    import nibabel as nib
    image = nib.load(str(patient.t1))
    truth = nib.load(str(patient.manual_mask))
    others = [nib.load(str(patient[s])) for s in SEQUENCES[1:]] + [truth]
    if predicted_path is not None:
        others.append(nib.load(str(predicted_path)))
    if any(v.shape != image.shape or not np.allclose(v.affine, image.affine, atol=1e-4) for v in others):
        raise ValueError(f"{patient.mr}的影像、真值或预测几何不一致；不自动重采样修正。")
    for volume in others[-2:] if predicted_path is not None else [truth]:
        values = np.unique(np.asarray(volume.dataobj))
        if not np.isfinite(values).all() or not set(values) <= {0, 1, 2, 3}:
            raise ValueError(f"{patient.mr}标签不是0/1/2/3。")
    prediction = others[-1] if predicted_path is not None else None
    return image, truth, prediction


def export_dataset(cfg):
    require_native_images(cfg)
    if cfg["annotation_scope"] != "full_volume":
        raise ValueError("分割基线当前要求完整三维标注；先核对annotation_scope，部分标注不能直接当全背景。")
    names = cfg["label_names"]
    if set(names) != {"1", "2", "3"} or any(not isinstance(v, str) or not v.strip() for v in names.values()):
        raise ValueError("请按真实标注填写三个label_names，不能猜测病灶类型。")
    if len(set(names.values()) | {"background"}) != 4:
        raise ValueError("前景标签名称应唯一且不能命名为background。")
    loc, cohort = locations(cfg), read_cohort(cfg)
    if "annotation_scope" not in cohort or not cohort.annotation_scope.eq("full_volume").all():
        raise ValueError("患者清单中存在局部标注或未确认的病例；完整体积导出不能把未标注层当作背景。")
    fresh(loc["raw"])
    for _, patient in cohort.iterrows():
        _volumes(patient)
    train_images, train_labels, test_images = [loc["raw"] / s for s in ("imagesTr", "labelsTr", "imagesTs")]
    for folder in (train_images, train_labels, test_images):
        folder.mkdir(parents=True)
    for _, patient in cohort.iterrows():
        training = patient.split == "development"
        target = train_images if training else test_images
        for i, sequence in enumerate(SEQUENCES):
            shutil.copy2(patient[sequence], target / f"{patient.mr}_{i:04d}.nii.gz")
        if training:
            shutil.copy2(patient.manual_mask, train_labels / f"{patient.mr}.nii.gz")
    dev = cohort.query("split == 'development'")
    splits = []
    for fold in range(1, 6):
        valid_mrs = dev.loc[dev.development_fold == fold, "mr"].tolist()
        training_mrs = dev.loc[dev.development_fold != fold, "mr"].tolist()
        splits.append({"train": training_mrs, "val": valid_mrs})
        inputs = loc["root"] / "oof_inputs" / f"fold_{fold - 1}"
        inputs.mkdir(parents=True)
        for mr in valid_mrs:
            for i in range(3):
                filename = f"{mr}_{i:04d}.nii.gz"
                shutil.copy2(train_images / filename, inputs / filename)
    dump_json(loc["raw"] / "dataset.json", {"channel_names": {"0": "T1-FLAIR", "1": "T2", "2": "DWI"},
        "labels": {"background": 0, **{value: int(key) for key, value in names.items()}},
        "numTraining": len(dev), "file_ending": ".nii.gz"})
    dump_json(loc["root"] / "splits_final.json", splits)
    dump_json(loc["root"] / "dataset_provenance.json", {"cohort_sha256": digest(paths(cfg)["cohort"]),
        "validation_labels_in_training": False, "annotation_scope": cfg["annotation_scope"],
        "configuration": cfg["segmentation"]["configuration"], "label_names": names})
    record(cfg, "segmentation_dataset")
    return loc


def verify_dataset(cfg):
    loc, cohort = locations(cfg), read_cohort(cfg)
    meta = json.loads((loc["root"] / "dataset_provenance.json").read_text(encoding="utf-8"))
    if meta["cohort_sha256"] != digest(paths(cfg)["cohort"]):
        raise ValueError("分割导出后病例清单发生变化。")
    if meta["configuration"] != cfg["segmentation"]["configuration"] or meta["label_names"] != cfg["label_names"]:
        raise ValueError("分割配置或标签定义发生变化，请使用新的run_id导出。")
    expected = set(cohort.loc[cohort.split == "development", "mr"])
    actual = {p.name.removesuffix(".nii.gz") for p in (loc["raw"] / "labelsTr").glob("*.nii.gz")}
    if actual != expected:
        raise ValueError("labelsTr与开发队列不完全一致，禁止继续训练。")


def command_plan(cfg):
    loc = locations(cfg)
    settings = cfg["segmentation"]
    dataset, configuration = str(settings["dataset_id"]), settings["configuration"]
    if configuration not in ("2d", "3d_fullres"):
        raise ValueError("首版只提供2d或3d_fullres基线。")
    plan = {"preprocess": [["nnUNetv2_plan_and_preprocess", "-d", dataset, "--verify_dataset_integrity"]],
            "train": [], "predict_oof": [], "predict_validation": []}
    for fold in range(5):
        plan["train"].append(["nnUNetv2_train", dataset, configuration, str(fold), "--npz", "-device", settings["device"]])
        plan["predict_oof"].append(["nnUNetv2_predict", "-i", str(loc["root"] / "oof_inputs" / f"fold_{fold}"),
            "-o", str(loc["oof"] / f"fold_{fold}"), "-d", dataset, "-c", configuration, "-f", str(fold), "-device", settings["device"]])
    plan["predict_validation"] = [["nnUNetv2_predict", "-i", str(loc["raw"] / "imagesTs"),
        "-o", str(loc["validation"]), "-d", dataset, "-c", configuration, "-f", "0", "1", "2", "3", "4", "-device", settings["device"]]]
    return plan


def install_splits(cfg):
    verify_dataset(cfg)
    loc = locations(cfg)
    if not loc["preprocessed"].is_dir():
        raise FileNotFoundError("先在指定服务器完成preprocess，再安装固定splits_final.json。")
    source = loc["root"] / "splits_final.json"
    destination = loc["preprocessed"] / "splits_final.json"
    if destination.exists():
        if json.loads(destination.read_text()) != json.loads(source.read_text()):
            raise ValueError("已有nnU-Net划分与固定开发折不一致；不覆盖。")
    else:
        shutil.copy2(source, destination)
    return destination


def run_phase(cfg, phase):
    """Called only by an explicitly enabled server notebook cell; no shell=True."""
    verify_dataset(cfg)
    loc = locations(cfg)
    if phase not in command_plan(cfg):
        raise ValueError("未知分割阶段。")
    if phase != "preprocess":
        install_splits(cfg)
    env = os.environ.copy()
    env.update(nnUNet_raw=str(loc["raw"].parent), nnUNet_preprocessed=str(loc["preprocessed"].parent),
               nnUNet_results=str(loc["results"]))
    for folder in (loc["preprocessed"].parent, loc["results"]):
        folder.mkdir(parents=True, exist_ok=True)
    for command in command_plan(cfg)[phase]:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    record(cfg, "segmentation_" + phase)


def evaluate_masks(cfg, scope="validation"):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    verify_dataset(cfg)
    loc, cohort = locations(cfg), read_cohort(cfg)
    if scope not in ("validation", "development_oof"):
        raise ValueError("scope只能是validation或development_oof。")
    cases = cohort.query("split == 'validation'" if scope == "validation" else "split == 'development'")
    rows = []
    for _, patient in cases.iterrows():
        directory = loc["validation"] if scope == "validation" else loc["oof"] / f"fold_{int(patient.development_fold) - 1}"
        image, truth, prediction = _volumes(patient, directory / f"{patient.mr}.nii.gz")
        if image.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"{patient.mr}的NIfTI空间单位不是明确的mm；请核查后再报告HD95_mm。")
        true_values, pred_values = np.asarray(truth.dataobj), np.asarray(prediction.dataobj)
        for label in (1, 2, 3):
            a, b = true_values == label, pred_values == label
            na, nb = int(a.sum()), int(b.sum())
            row = {"mr": patient.mr, "scope": scope, "label": label, "true_voxels": na, "predicted_voxels": nb}
            if not na and not nb:
                row.update(dice=np.nan, hd95_mm=np.nan, status="both_empty_excluded_from_mean")
            elif not na or not nb:
                row.update(dice=0.0, hd95_mm=np.nan, status="one_empty_hd95_undefined")
            else:
                surface_a = a ^ binary_erosion(a, border_value=0)
                surface_b = b ^ binary_erosion(b, border_value=0)
                distances = np.concatenate([distance_transform_edt(~surface_b, sampling=image.header.get_zooms()[:3])[surface_a],
                                            distance_transform_edt(~surface_a, sampling=image.header.get_zooms()[:3])[surface_b]])
                row.update(dice=float(2 * (a & b).sum() / (na + nb)), hd95_mm=float(np.percentile(distances, 95)), status="ok")
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(loc["root"] / f"{scope}_segmentation_metrics.csv", index=False)
    summary = table.groupby("label").agg(n_cases=("mr", "size"), n_dice_defined=("dice", "count"),
        mean_dice=("dice", "mean"), n_hd95_defined=("hd95_mm", "count"), mean_hd95_mm=("hd95_mm", "mean"))
    summary.to_csv(loc["root"] / f"{scope}_segmentation_summary.csv")
    return table, summary


def build_validation_mask_cache(cfg):
    from scipy.ndimage import zoom
    verify_dataset(cfg)
    loc, p = locations(cfg), paths(cfg)
    target = loc["root"] / "validation_masks.npz"
    fresh(target)
    cases = read_cohort(cfg).query("split == 'validation'").reset_index(drop=True)
    result = np.zeros((len(cases), 4, 3, 224, 224), dtype=np.uint8)
    with np.load(p["cache"], allow_pickle=False) as cache:
        layers = cache["max_layers"]
        for i, patient in cases.iterrows():
            image, _, predicted = _volumes(patient, loc["validation"] / f"{patient.mr}.nii.gz")
            labels = np.asarray(predicted.dataobj)
            z0 = int(layers[int(patient.cache_index)]) - 1
            if z0 < 0 or z0 + 3 >= image.shape[2]:
                raise ValueError("自动掩模四层取层越界。")
            for s in range(4):
                for label in (1, 2, 3):
                    result[i, s, label - 1] = zoom((labels[:, :, z0 + s] == label).astype(np.uint8),
                        (224 / image.shape[0], 224 / image.shape[1]), order=0, prefilter=False)
    np.savez_compressed(target, mrs=cases.mr.to_numpy(dtype=str), masks=result)
    dump_json(loc["root"] / "automatic_input_note.json", {
        "scope": "validation_only", "experiment": "manual_model_auto_mask_substitution",
        "manual_slice_selection_still_required": True,
        "mask_cache_sha256": digest(target),
        "predictions_sha256": {mr: digest(loc["validation"] / f"{mr}.nii.gz") for mr in cases.mr},
        "not_implemented": "用嵌套交叉拟合自动掩模重训分类器的完整自动联合开发流程"})
    return target


def verify_validation_mask_cache(cfg):
    verify_dataset(cfg)
    loc = locations(cfg)
    metadata = json.loads((loc["root"] / "automatic_input_note.json").read_text(encoding="utf-8"))
    if digest(loc["root"] / "validation_masks.npz") != metadata["mask_cache_sha256"]:
        raise ValueError("自动分类掩模缓存发生变化。")
    expected = set(read_cohort(cfg).query("split == 'validation'").mr)
    if set(metadata["predictions_sha256"]) != expected:
        raise ValueError("自动掩模缓存的验证患者不一致。")
    for mr, checksum in metadata["predictions_sha256"].items():
        if digest(loc["validation"] / f"{mr}.nii.gz") != checksum:
            raise ValueError(f"{mr}的原生自动预测与四层缓存不一致，请使用新实验版本。")
