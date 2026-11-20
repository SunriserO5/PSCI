"""New radiomics baseline, independent of the unchanged ResNet implementation."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .project import CLINICAL, SEED, SEQUENCES, digest, dump_json, fresh, paths, read_cohort, record, require_native_images, cohort_identity, export_transfer, verify_transfer
from .selected_slices import selected_patient
from .reporting import classification_metrics, threshold_from_development


def extract_features(cfg):
    require_native_images(cfg)
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor
    p, cohort = paths(cfg), read_cohort(cfg)
    destination = p["radiomics"] / "features.csv"
    fresh(destination, p["radiomics"] / "extraction_config.json")
    settings = dict(cfg["radiomics"]["settings"])
    settings.update(force2D=True, force2Ddimension=0, minimumROIDimensions=2)
    if "shape" in cfg["radiomics"]["feature_classes"]:
        raise ValueError("四层标注使用 shape2D，不能使用三维 shape。")
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for feature_class in cfg["radiomics"]["feature_classes"]:
        extractor.enableFeatureClassByName(feature_class)
    p["radiomics"].mkdir(parents=True, exist_ok=True)
    records, failures, skipped, provenance = [], [], [], []
    for _, patient in cohort.iterrows():
        images, masks, source = selected_patient(cfg, patient)
        provenance.append(source)
        row = {"mr": patient.mr, "label": int(patient.label), "split": patient.split}
        collected = {}
        for label in (1, 2, 3):
            present = [(masks[:, :, index] == label).any() for index in range(4)]
            row[f"roi{label}__present_fraction"] = float(np.mean(present))
            row[f"roi{label}__pixel_count"] = int((masks == label).sum())
            valid_slices = 0
            for index in range(4):
                roi = masks[:, :, index] == label
                coordinates = np.argwhere(roi)
                if not len(coordinates):
                    continue
                if len(coordinates) < 2 or (np.ptp(coordinates, axis=0) < 1).any():
                    skipped.append({"mr": patient.mr, "slice": source["slices_1based"][index],
                                    "label": label, "reason": "ROI不足二维范围；计数保留，纹理记缺失"})
                    continue
                valid_slices += 1
                # Singleton z in SimpleITK order (z, y, x); unit working grid is pixels.
                mask = sitk.GetImageFromArray(masks[:, :, index].T[None].astype(np.uint8))
                mask.SetSpacing((1.0, 1.0, 1.0))
                for channel, sequence in enumerate(SEQUENCES):
                    image = sitk.GetImageFromArray(images[channel][:, :, index].T[None].astype(np.float32))
                    image.CopyInformation(mask)
                    try:
                        features = extractor.execute(image, mask, label=label)
                        for name, value in features.items():
                            if not name.startswith("original_") or ("_shape2D_" in name and sequence != "t1"):
                                continue
                            value = float(value)
                            if np.isfinite(value):
                                collected.setdefault(f"{sequence}__roi{label}__{name}", []).append(value)
                    except (ValueError, RuntimeError) as exc:
                        failures.append({"mr": patient.mr, "slice": source["slices_1based"][index],
                                         "sequence": sequence, "label": label, "error": str(exc)})
            row[f"roi{label}__valid_2d_fraction"] = valid_slices / 4.0
        for feature, values in collected.items():
            for suffix, operation in (("mean", np.mean), ("std", np.std), ("max", np.max)):
                row[f"{feature}__{suffix}"] = float(operation(values))
        records.append(row)
    pd.DataFrame(failures, columns=["mr", "slice", "sequence", "label", "error"]).to_csv(p["radiomics"] / "extraction_errors.csv", index=False)
    pd.DataFrame(skipped, columns=["mr", "slice", "label", "reason"]).to_csv(p["radiomics"] / "small_roi_notes.csv", index=False)
    if failures:
        raise ValueError("提取失败已记录 extraction_errors.csv；处理后再重新提取。")
    table = pd.DataFrame(records)
    table.to_csv(destination, index=False)
    dump_json(p["radiomics"] / "extraction_config.json", {
        "parameters": cfg["radiomics"], "pyradiomics_version": radiomics.__version__,
        "roi_scope": "selected_four_slices_2d_pixels_v1", "cohort_identity": cohort_identity(cfg),
        "features_sha256": digest(destination), "sources": provenance,
        "aggregation": "patient-wise mean/std/max over nonempty valid annotated slices; absent features imputed inside training folds",
        "shape_units": "pixels / pixels squared, not millimetres or whole-lesion volume"})
    record(cfg, "radiomics_features", {"protocol": "selected_four_slices_2d_pixels_v1"})
    return table


def _search(cfg, radiomic_columns, clinical_columns):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rad = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("variance", VarianceThreshold()), ("scale", StandardScaler()),
                    ("select", SelectPercentile(f_classif))])
    transformers = [("rad", rad, radiomic_columns)]
    if clinical_columns:
        clinical = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
        transformers.append(("clinical", clinical, clinical_columns))
    pipe = Pipeline([("features", ColumnTransformer(transformers)),
                     ("classifier", LogisticRegression(solver="liblinear", max_iter=3000))])
    return GridSearchCV(pipe, {
        "features__rad__select__percentile": cfg["radiomics"]["percentiles"],
        "classifier__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
        scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=1, error_score="raise")


def _selected_features(model, columns):
    rad = model.named_steps["features"].named_transformers_["rad"]
    retained = np.asarray(columns)[rad.named_steps["variance"].get_support()]
    return retained[rad.named_steps["select"].get_support()].tolist()


def fit_radiomics_models(cfg):
    import joblib
    p, cohort = paths(cfg), read_cohort(cfg)
    fresh(p["radiomics"] / "models.joblib", p["radiomics"] / "model_meta.json")
    extraction = json.loads((p["radiomics"] / "extraction_config.json").read_text(encoding="utf-8"))
    if extraction["cohort_identity"] != cohort_identity(cfg) or extraction["features_sha256"] != digest(p["radiomics"] / "features.csv"):
        raise ValueError("提取特征后病例清单发生变化。")
    features = pd.read_csv(p["radiomics"] / "features.csv", dtype={"mr": str})
    if not features.mr.is_unique or set(features.mr) != set(cohort.mr):
        raise ValueError("影像组学特征与固定队列不一一对应。")
    matched = features.set_index("mr").loc[cohort.mr]
    if not np.array_equal(matched.label, cohort.label) or matched.split.tolist() != cohort.split.tolist():
        raise ValueError("特征表的标签或时间划分与队列不一致。")
    if extraction["roi_scope"] != "selected_four_slices_2d_pixels_v1":
        raise ValueError("特征提取协议与当前四层二维流程不同。")
    table = cohort.merge(features.drop(columns=["label", "split"]), on="mr", validate="one_to_one")
    table = table.query("split == 'development'").reset_index(drop=True)
    lookup = features.set_index("mr")
    if not np.array_equal(lookup.loc[table.mr, "label"].to_numpy(), table.label.to_numpy()):
        raise ValueError("影像组学结局与队列不一致。")
    radiomic_columns = [c for c in features if c not in ("mr", "label", "split")]
    if not radiomic_columns:
        raise ValueError("没有影像组学特征。")
    y = table.label.to_numpy(dtype=int)
    fitted, metadata, predictions, selection_rows = {}, {}, [], []
    for name, clinical_columns in (("Radiomics", []), ("Radiomics_Clinical5", CLINICAL)):
        columns = radiomic_columns + clinical_columns
        X = table[columns].replace([np.inf, -np.inf], np.nan)
        oof, fold_auc = np.zeros(len(y), dtype=float), []
        for fold in range(1, 6):
            valid = table.development_fold.to_numpy() == fold
            search = _search(cfg, radiomic_columns, clinical_columns)
            search.fit(X.loc[~valid], y[~valid])
            oof[valid] = search.predict_proba(X.loc[valid])[:, 1]
            from sklearn.metrics import roc_auc_score
            fold_auc.append(float(roc_auc_score(y[valid], oof[valid])))
            for feature in _selected_features(search.best_estimator_, radiomic_columns):
                selection_rows.append({"model": name, "fold": fold, "feature": feature})
        search = _search(cfg, radiomic_columns, clinical_columns)
        search.fit(X, y)
        fitted[name] = search.best_estimator_
        threshold = threshold_from_development(y, oof)
        if not np.isfinite(threshold):
            raise ValueError(f"{name}未形成有限Youden阈值；先检查开发预测。")
        metadata[name] = {"features": columns, "threshold": threshold, "best_params": search.best_params_,
                          "selected_features": _selected_features(search.best_estimator_, radiomic_columns),
                          "mean_fold_auc": float(np.mean(fold_auc)), "fold_aucs": fold_auc,
                          "development": classification_metrics(y, oof, threshold, n_bootstrap=0)}
        prediction = table[["mr", "label", "development_fold"]].copy()
        prediction["model"], prediction["probability"] = name, oof
        predictions.append(prediction)
    joblib.dump(fitted, p["radiomics"] / "models.joblib")
    pd.concat(predictions).to_csv(p["radiomics"] / "development_oof.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(p["radiomics"] / "selected_features_by_fold.csv", index=False)
    dump_json(p["radiomics"] / "model_meta.json", metadata)
    dump_json(p["radiomics"] / "frozen_artifacts.json", {"artifacts": {
        filename: digest(p["radiomics"] / filename) for filename in ("models.joblib", "model_meta.json", "features.csv", "extraction_config.json")},
        "cohort_identity": cohort_identity(cfg)})
    record(cfg, "radiomics_models", {"feature_selection": "inside_inner_cv"})
    return metadata


def predict_radiomics_validation(cfg):
    p, cohort = paths(cfg), read_cohort(cfg)
    receipt = p["radiomics"] / "transfer_manifest.json"
    if receipt.exists():
        manifest = verify_transfer(cfg, p["radiomics"])
        if manifest["kind"] == "radiomics_complete":
            return imported_radiomics_predictions(cfg)
    import joblib
    lock = json.loads((p["radiomics"] / "frozen_artifacts.json").read_text(encoding="utf-8"))
    if lock["cohort_identity"] != cohort_identity(cfg):
        raise ValueError("固定队列发生变化。")
    for name, expected in lock["artifacts"].items():
        if digest(p["radiomics"] / name) != expected:
            raise ValueError(f"影像组学固定产物发生变化：{name}")
    feature_table = pd.read_csv(p["radiomics"] / "features.csv", dtype={"mr": str})
    val = cohort.query("split == 'validation'").merge(feature_table.drop(columns=["label", "split"]), on="mr", validate="one_to_one")
    models = joblib.load(p["radiomics"] / "models.joblib")
    meta = json.loads((p["radiomics"] / "model_meta.json").read_text(encoding="utf-8"))
    predictions, summaries = [], []
    for name, model in models.items():
        prob = model.predict_proba(val[meta[name]["features"]].replace([np.inf, -np.inf], np.nan))[:, 1]
        threshold = meta[name]["threshold"]
        table = val[["mr", "label"]].copy()
        table["model"], table["probability"], table["threshold"] = name, prob, threshold
        table["prediction"] = (prob >= threshold).astype(int)
        table["protocol"] = "radiomics_selected_four_slices_nested_feature_selection"
        predictions.append(table)
        summaries.append({"model": name, **classification_metrics(val.label, prob, threshold)})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)


def imported_radiomics_predictions(cfg):
    root = paths(cfg)["radiomics"]
    verify_transfer(cfg, root, "radiomics_complete")
    table = pd.read_csv(root / "validation_predictions.csv", dtype={"mr": str})
    expected = read_cohort(cfg).query("split == 'validation'").set_index("mr")
    meta = json.loads((root / "model_meta.json").read_text(encoding="utf-8"))
    if set(table.model) != {"Radiomics", "Radiomics_Clinical5"}:
        raise ValueError("影像组学预测缺少对照组。")
    summaries = []
    for name, group in table.groupby("model"):
        if not group.mr.is_unique or set(group.mr) != set(expected.index):
            raise ValueError("导入预测的时间验证患者不一致。")
        if not np.array_equal(group.label, expected.loc[group.mr, "label"]):
            raise ValueError("导入预测结局不一致。")
        threshold = float(meta[name]["threshold"])
        if not np.allclose(group.threshold, threshold, rtol=0, atol=1e-12):
            raise ValueError("导入预测阈值与开发阶段不同。")
        if not np.array_equal(group.prediction, (group.probability >= threshold).astype(int)):
            raise ValueError("导入预测分类与概率不一致。")
        summaries.append({"model": name, **classification_metrics(group.label, group.probability, threshold)})
    return table, pd.DataFrame(summaries)


def export_radiomics(cfg, destination, include_models=True):
    root = paths(cfg)["radiomics"]
    extraction = json.loads((root / "extraction_config.json").read_text(encoding="utf-8"))
    if extraction["cohort_identity"] != cohort_identity(cfg) or extraction["features_sha256"] != digest(root / "features.csv"):
        raise ValueError("特征表或队列在提取后发生变化。")
    files = ["features.csv", "extraction_config.json", "extraction_errors.csv", "small_roi_notes.csv"]
    kind = "radiomics_features"
    if include_models:
        predictions, _ = predict_radiomics_validation(cfg)
        target = root / "validation_predictions.csv"
        if target.exists():
            existing = pd.read_csv(target, dtype={"mr": str})
            pd.testing.assert_frame_equal(existing, predictions, check_dtype=False, rtol=1e-10, atol=1e-12)
        else:
            predictions.to_csv(target, index=False)
        files += ["models.joblib", "model_meta.json", "frozen_artifacts.json", "development_oof.csv",
                  "selected_features_by_fold.csv", "validation_predictions.csv"]
        kind = "radiomics_complete"
    return export_transfer(cfg, root, files, destination, kind, "selected_four_slices_2d_pixels_v1")
