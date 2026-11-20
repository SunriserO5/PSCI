"""Patient-grouped 2D nnU-Net on the four annotated slices, with portable artifacts."""
from __future__ import annotations
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .project import digest, dump_json, fresh, paths, read_cohort, record, require_native_images, cohort_identity, export_transfer, verify_transfer
from .selected_slices import selected_patient, selected_indices


def locations(cfg):
    root = paths(cfg)["segmentation"]
    dataset_id = int(cfg["segmentation"]["dataset_id"])
    if not 100 <= dataset_id <= 999 or cfg["segmentation"]["configuration"] != "2d":
        raise ValueError("四层标注使用100～999的数据集ID和2d配置。")
    name = f"Dataset{dataset_id:03d}_PSCISelected2D"
    return {"root": root, "name": name, "raw": root / "nnUNet_raw" / name,
        "preprocessed": root / "nnUNet_preprocessed" / name, "results": root / "nnUNet_results",
        "validation": root / "validation_predictions", "oof": root / "development_oof_predictions"}


def segmentation_label_names(cfg):
    # Numeric placeholders preserve IDs without inventing a clinical interpretation.
    names = {str(i): cfg.get("label_names", {}).get(str(i)) or f"label_{i}" for i in (1, 2, 3)}
    if len(set(names.values()) | {"background"}) != 4:
        raise ValueError("标签名称须唯一且不能命名为background。")
    return names


def export_dataset(cfg):
    import nibabel as nib
    require_native_images(cfg)
    loc, cohort = locations(cfg), read_cohort(cfg)
    fresh(loc["root"])
    names = segmentation_label_names(cfg)
    for directory in ("imagesTr", "labelsTr", "imagesTs"):
        (loc["raw"] / directory).mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    # Artificial working geometry: one slice, original in-plane resolution, distances in pixels.
    affine = np.diag([1.0, 1.0, 999.0, 1.0])
    for _, patient in cohort.iterrows():
        images, truth, source = selected_patient(cfg, patient)
        sources.append(source)
        training = patient.split == "development"
        folder = loc["raw"] / ("imagesTr" if training else "imagesTs")
        for index, layer in enumerate(source["slices_1based"]):
            case = f"{patient.mr}_s{layer:03d}"
            rows.append({"case_id": case, "mr": patient.mr, "slice_1based": layer,
                         "slice_offset": index, "split": patient.split, "development_fold": int(patient.development_fold)})
            for channel, volume in enumerate(images):
                nib.save(nib.Nifti1Image(volume[:, :, index:index + 1].astype(np.float32), affine), folder / f"{case}_{channel:04d}.nii.gz")
            if training:
                nib.save(nib.Nifti1Image(truth[:, :, index:index + 1], affine), loc["raw"] / "labelsTr" / f"{case}.nii.gz")
    cases = pd.DataFrame(rows)
    cases.to_csv(loc["root"] / "case_index.csv", index=False)
    dev = cases.query("split == 'development'")
    if set(dev.development_fold) != {1, 2, 3, 4, 5}:
        raise ValueError("开发患者必须沿用00中的五折。")
    splits = []
    for fold in range(1, 6):
        valid = dev.loc[dev.development_fold == fold, "case_id"].tolist()
        train = dev.loc[dev.development_fold != fold, "case_id"].tolist()
        splits.append({"train": train, "val": valid})
        inputs = loc["root"] / "oof_inputs" / f"fold_{fold - 1}"
        inputs.mkdir(parents=True)
        for case in valid:
            for channel in range(3):
                name = f"{case}_{channel:04d}.nii.gz"
                shutil.copy2(loc["raw"] / "imagesTr" / name, inputs / name)
    dump_json(loc["raw"] / "dataset.json", {"channel_names": {"0": "T1-FLAIR", "1": "T2", "2": "DWI"},
        "labels": {"background": 0, **{name: int(label) for label, name in names.items()}},
        "numTraining": len(dev), "file_ending": ".nii.gz"})
    dump_json(loc["root"] / "splits_final.json", splits)
    tracked = [file for directory in (loc["raw"], loc["root"] / "oof_inputs") for file in directory.rglob("*") if file.is_file()]
    tracked += [loc["root"] / "case_index.csv", loc["root"] / "splits_final.json"]
    dump_json(loc["root"] / "dataset_provenance.json", {"cohort_identity": cohort_identity(cfg),
        "validation_labels_in_training": False, "annotation_scope": "selected_slices", "configuration": "2d",
        "protocol": "selected_four_slices_2d_pixels_v1", "label_names": names, "sources": sources,
        "artifacts": {file.relative_to(loc["root"]).as_posix(): digest(file) for file in tracked}})
    record(cfg, "segmentation_dataset", {"protocol": "selected_four_slices_2d_pixels_v1"})
    return loc


def verify_dataset(cfg):
    loc, cohort = locations(cfg), read_cohort(cfg)
    meta = json.loads((loc["root"] / "dataset_provenance.json").read_text(encoding="utf-8"))
    if meta["cohort_identity"] != cohort_identity(cfg) or meta["annotation_scope"] != cfg["annotation_scope"]:
        raise ValueError("分割导出后患者、划分、取层或标注协议发生变化。")
    if meta["configuration"] != cfg["segmentation"]["configuration"] or meta["label_names"] != segmentation_label_names(cfg):
        raise ValueError("分割配置或标签定义发生变化，请更换 segmentation.experiment。")
    for name, checksum in meta["artifacts"].items():
        if digest(loc["root"] / name) != checksum:
            raise ValueError(f"分割数据文件变化：{name}")
    dev = cohort.query("split == 'development'")
    expected = {f"{patient.mr}_s{layer:03d}" for _, patient in dev.iterrows()
                for layer in range(int(patient.max_layer), int(patient.max_layer) + 4)}
    actual = {file.name.removesuffix(".nii.gz") for file in (loc["raw"] / "labelsTr").glob("*.nii.gz")}
    if actual != expected:
        raise ValueError("labelsTr与开发患者四层清单不一致。")


def command_plan(cfg):
    loc, settings = locations(cfg), cfg["segmentation"]
    dataset = str(settings["dataset_id"])
    plan = {"preprocess": [["nnUNetv2_plan_and_preprocess", "-d", dataset, "-c", "2d", "--verify_dataset_integrity"]],
            "train": [], "predict_oof": [], "predict_validation": []}
    for fold in range(5):
        plan["train"].append(["nnUNetv2_train", dataset, "2d", str(fold), "--npz", "-device", settings["device"]])
        plan["predict_oof"].append(["nnUNetv2_predict", "-i", str(loc["root"] / "oof_inputs" / f"fold_{fold}"),
            "-o", str(loc["oof"] / f"fold_{fold}"), "-d", dataset, "-c", "2d", "-f", str(fold), "-device", settings["device"]])
    plan["predict_validation"] = [["nnUNetv2_predict", "-i", str(loc["raw"] / "imagesTs"),
        "-o", str(loc["validation"]), "-d", dataset, "-c", "2d", "-f", "0", "1", "2", "3", "4", "-device", settings["device"]]]
    return plan


def install_splits(cfg):
    verify_dataset(cfg)
    loc = locations(cfg)
    if not loc["preprocessed"].is_dir():
        raise FileNotFoundError("先完成预处理，再安装固定的患者划分。")
    source, target = loc["root"] / "splits_final.json", loc["preprocessed"] / "splits_final.json"
    if target.exists():
        if json.loads(target.read_text()) != json.loads(source.read_text()):
            raise ValueError("已有nnU-Net划分与固定患者折不同，不覆盖。")
    else:
        shutil.copy2(source, target)
    return target


def run_phase(cfg, phase):
    verify_dataset(cfg)
    loc = locations(cfg)
    if phase not in command_plan(cfg):
        raise ValueError("未知分割阶段。")
    if phase != "preprocess":
        install_splits(cfg)
    env = os.environ.copy()
    env.update(nnUNet_raw=str(loc["raw"].parent), nnUNet_preprocessed=str(loc["preprocessed"].parent), nnUNet_results=str(loc["results"]))
    for folder in (loc["preprocessed"].parent, loc["results"]):
        folder.mkdir(parents=True, exist_ok=True)
    for command in command_plan(cfg)[phase]:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True, env=env)
    record(cfg, "segmentation_" + phase, {"protocol": "selected_four_slices_2d_pixels_v1"})


def predicted_slices(cfg, patient, scope, shape):
    import nibabel as nib
    loc = locations(cfg)
    directory = loc["validation"] if scope == "validation" else loc["oof"] / f"fold_{int(patient.development_fold) - 1}"
    arrays, checksums = [], {}
    for layer in range(int(patient.max_layer), int(patient.max_layer) + 4):
        name = f"{patient.mr}_s{layer:03d}.nii.gz"
        volume = nib.load(str(directory / name))
        values = np.asarray(volume.dataobj)
        if values.shape != (*shape, 1) or not np.allclose(volume.affine, np.diag([1., 1., 999., 1.]), rtol=0, atol=1e-4):
            raise ValueError(f"{name} 预测工作网格不一致。")
        if not np.isfinite(values).all() or not set(np.unique(values)) <= {0, 1, 2, 3}:
            raise ValueError(f"{name} 预测标签不是0/1/2/3。")
        arrays.append(values[:, :, 0].astype(np.uint8))
        checksums[name] = digest(directory / name)
    return np.stack(arrays, axis=2), checksums


def evaluate_masks(cfg, scope="validation"):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    require_native_images(cfg)
    verify_dataset(cfg)
    if scope not in ("validation", "development_oof"):
        raise ValueError("scope只能是validation或development_oof。")
    loc, cohort = locations(cfg), read_cohort(cfg)
    cases = cohort.query("split == 'validation'" if scope == "validation" else "split == 'development'")
    rows, prediction_hashes = [], {}
    for _, patient in cases.iterrows():
        _, truth, _ = selected_patient(cfg, patient)
        prediction, hashes = predicted_slices(cfg, patient, scope, truth.shape[:2])
        prediction_hashes.update(hashes)
        for label in (1, 2, 3):
            a, b = truth == label, prediction == label
            na, nb = int(a.sum()), int(b.sum())
            distances, undefined = [], False
            for index in range(4):
                ai, bi = a[:, :, index], b[:, :, index]
                if not ai.any() and not bi.any():
                    continue
                if not ai.any() or not bi.any():
                    undefined = True
                    continue
                sa, sb = ai ^ binary_erosion(ai, border_value=0), bi ^ binary_erosion(bi, border_value=0)
                distances.extend([distance_transform_edt(~sb)[sa], distance_transform_edt(~sa)[sb]])
            rows.append({"mr": patient.mr, "scope": scope, "label": label, "true_pixels": na, "predicted_pixels": nb,
                "dice": float(2 * (a & b).sum() / (na + nb)) if na + nb else np.nan,
                "hd95_pixels": float(np.percentile(np.concatenate(distances), 95)) if distances and not undefined else np.nan,
                "status": "both_empty" if not na + nb else "empty_slice_hd95_undefined" if undefined else "ok"})
    table = pd.DataFrame(rows)
    table.to_csv(loc["root"] / f"{scope}_segmentation_metrics.csv", index=False)
    summary = table.groupby("label").agg(n_cases=("mr", "size"), n_dice_defined=("dice", "count"), mean_dice=("dice", "mean"),
        n_hd95_defined=("hd95_pixels", "count"), mean_hd95_pixels=("hd95_pixels", "mean"))
    summary.to_csv(loc["root"] / f"{scope}_segmentation_summary.csv")
    dump_json(loc["root"] / f"{scope}_evaluation_provenance.json", {
        "cohort_identity": cohort_identity(cfg), "scope": scope, "predictions_sha256": prediction_hashes,
        "artifacts": {name: digest(loc["root"] / name) for name in
            (f"{scope}_segmentation_metrics.csv", f"{scope}_segmentation_summary.csv")}})
    return table, summary


def build_validation_mask_cache(cfg):
    from scipy.ndimage import zoom
    require_native_images(cfg)
    verify_dataset(cfg)
    loc = locations(cfg)
    target = loc["root"] / "validation_masks.npz"
    fresh(target)
    cases = read_cohort(cfg).query("split == 'validation'").reset_index(drop=True)
    result = np.zeros((len(cases), 4, 3, 224, 224), dtype=np.uint8)
    hashes = {}
    for i, patient in cases.iterrows():
        _, truth, _ = selected_patient(cfg, patient)
        labels, checksums = predicted_slices(cfg, patient, "validation", truth.shape[:2])
        hashes.update(checksums)
        for index in range(4):
            for label in (1, 2, 3):
                result[i, index, label - 1] = zoom((labels[:, :, index] == label).astype(np.uint8),
                    (224 / labels.shape[0], 224 / labels.shape[1]), order=0, prefilter=False)
    np.savez_compressed(target, mrs=cases.mr.to_numpy(dtype=str), masks=result)
    dump_json(loc["root"] / "automatic_input_note.json", {"cohort_identity": cohort_identity(cfg),
        "scope": "validation_only", "experiment": "manual_model_auto_mask_substitution",
        "manual_slice_selection_still_required": True, "protocol": "selected_four_slices_2d_pixels_v1",
        "mask_cache_sha256": digest(target), "predictions_sha256": hashes,
        "note": "只替换已训练分类器的病灶输入；不代表自动取层或自动掩模训练的联合模型。"})
    return target


def verify_validation_mask_cache(cfg):
    loc = locations(cfg)
    if (loc["root"] / "transfer_manifest.json").exists():
        verify_transfer(cfg, loc["root"], "segmentation_results")
    metadata = json.loads((loc["root"] / "automatic_input_note.json").read_text(encoding="utf-8"))
    if metadata["cohort_identity"] != cohort_identity(cfg) or metadata["protocol"] != "selected_four_slices_2d_pixels_v1":
        raise ValueError("自动掩模与当前患者或取层协议不一致。")
    if digest(loc["root"] / "validation_masks.npz") != metadata["mask_cache_sha256"]:
        raise ValueError("自动分类掩模缓存发生变化。")
    cases = read_cohort(cfg).query("split == 'validation'")
    with np.load(loc["root"] / "validation_masks.npz", allow_pickle=False) as cache:
        if cache["mrs"].astype(str).tolist() != cases.mr.tolist() or cache["masks"].shape != (len(cases), 4, 3, 224, 224):
            raise ValueError("自动掩模患者顺序或形状不一致。")
        if not set(np.unique(cache["masks"])) <= {0, 1} or np.any(cache["masks"].sum(axis=2) > 1):
            raise ValueError("自动掩模必须为互斥的三通道二值标签。")
    if not (loc["root"] / "transfer_manifest.json").exists():
        for name, checksum in metadata["predictions_sha256"].items():
            if digest(loc["validation"] / name) != checksum:
                raise ValueError(f"原预测发生变化：{name}")


def export_segmentation_results(cfg, destination):
    verify_validation_mask_cache(cfg)
    for scope in ("validation", "development_oof"):
        verify_segmentation_evaluation(cfg, scope)
    files = ["validation_masks.npz", "automatic_input_note.json", "dataset_provenance.json", "case_index.csv", "splits_final.json",
        "validation_segmentation_metrics.csv", "validation_segmentation_summary.csv",
        "development_oof_segmentation_metrics.csv", "development_oof_segmentation_summary.csv",
        "validation_evaluation_provenance.json", "development_oof_evaluation_provenance.json"]
    return export_transfer(cfg, locations(cfg)["root"], files, destination, "segmentation_results", "selected_four_slices_2d_pixels_v1")


def export_segmentation_weights(cfg, destination):
    from importlib.metadata import version
    loc = locations(cfg)
    verify_dataset(cfg)
    model = loc["results"] / loc["name"] / "nnUNetTrainer__nnUNetPlans__2d"
    files = [model / "dataset.json", model / "plans.json"]
    if (model / "dataset_fingerprint.json").is_file():
        files.append(model / "dataset_fingerprint.json")
    for fold in range(5):
        folder = model / f"fold_{fold}"
        candidates = [folder / name for name in ("checkpoint_final.pth", "checkpoint_final.safetensors") if (folder / name).is_file()]
        if len(candidates) != 1:
            raise ValueError(f"fold_{fold} 需要唯一的最终权重，当前找到{len(candidates)}个。")
        files += candidates
        files += [file for file in folder.iterdir() if file.suffix in (".json", ".pkl")]
    environment = loc["root"] / "model_environment.json"
    dump_json(environment, {"nnunetv2": version("nnunetv2"), "torch": version("torch"),
        "model_relative_path": model.relative_to(loc["root"]).as_posix(),
        "input": "native-resolution selected slices exported by this notebook; not the 224x224 classifier cache"})
    files += [environment, loc["root"] / "dataset_provenance.json", loc["root"] / "splits_final.json"]
    return export_transfer(cfg, loc["root"], [file.relative_to(loc["root"]).as_posix() for file in files],
                           destination, "segmentation_weights", "selected_four_slices_2d_pixels_v1")


def imported_segmentation_inference_plan(cfg, input_folder, output_folder):
    root = paths(cfg)["segmentation"].with_name(paths(cfg)["segmentation"].name + "_imported_model")
    verify_transfer(cfg, root, "segmentation_weights")
    meta = json.loads((root / "model_environment.json").read_text(encoding="utf-8"))
    from importlib.metadata import version
    if version("nnunetv2") != meta["nnunetv2"]:
        raise ValueError("恢复推理时请使用导出记录中的 nnunetv2 版本：" + meta["nnunetv2"])
    return ["nnUNetv2_predict_from_modelfolder", "-i", str(input_folder), "-o", str(output_folder),
            "-m", str(root / meta["model_relative_path"]), "-f", "0", "1", "2", "3", "4", "-device", cfg["segmentation"]["device"]]


def verify_segmentation_evaluation(cfg, scope):
    loc = locations(cfg)
    meta = json.loads((loc["root"] / f"{scope}_evaluation_provenance.json").read_text(encoding="utf-8"))
    if meta["cohort_identity"] != cohort_identity(cfg) or meta["scope"] != scope:
        raise ValueError("分割评价与当前患者或评价范围不同。")
    for name, checksum in meta["artifacts"].items():
        if digest(loc["root"] / name) != checksum:
            raise ValueError(f"分割指标发生变化：{name}")
    cases = read_cohort(cfg).set_index("mr")
    for name, checksum in meta["predictions_sha256"].items():
        mr = name.split("_s", 1)[0]
        directory = loc["validation"] if scope == "validation" else loc["oof"] / f"fold_{int(cases.loc[mr, 'development_fold']) - 1}"
        if digest(directory / name) != checksum:
            raise ValueError(f"{name}在评价后发生变化，请重新评价。")
