"""New radiomics baseline, independent of the unchanged ResNet implementation."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .project import CLINICAL, SEED, SEQUENCES, digest, dump_json, fresh, paths, read_cohort, record, require_native_images
from .reporting import classification_metrics, threshold_from_development


def extract_features(cfg):
    require_native_images(cfg)
    import SimpleITK as sitk
    import radiomics
    from radiomics import featureextractor
    if cfg["annotation_scope"] != "full_volume":
        raise ValueError("本版三维影像组学要求先确认annotation_scope=full_volume；部分标注需另定义有效ROI范围。")
    p, cohort = paths(cfg), read_cohort(cfg)
    if "annotation_scope" not in cohort or not cohort.annotation_scope.eq("full_volume").all():
        raise ValueError("患者清单中存在局部标注或未确认的病例，不能作为完整三维 ROI 提取特征。")
    if ("physical_geometry_needs_review" not in cohort
            or cohort.physical_geometry_needs_review.astype(str).str.lower().ne("false").any()):
        raise ValueError("患者空间信息尚未全部核实，请先处理00记录的单位和间距问题，再提取三维物理特征。")
    destination = p["radiomics"] / "features.csv"
    fresh(destination)
    extractor = featureextractor.RadiomicsFeatureExtractor(**cfg["radiomics"]["settings"])
    extractor.disableAllFeatures()
    for feature_class in cfg["radiomics"]["feature_classes"]:
        extractor.enableFeatureClassByName(feature_class)
    p["radiomics"].mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for _, patient in cohort.iterrows():
        row = {"mr": patient.mr, "label": int(patient.label), "split": patient.split}
        mask = sitk.ReadImage(patient.manual_mask)
        values = set(np.unique(sitk.GetArrayViewFromImage(mask)))
        for label in (1, 2, 3):
            row[f"roi_{label}_present"] = float(label in values)
            if label not in values:
                continue
            for sequence in SEQUENCES:
                try:
                    features = extractor.execute(str(patient[sequence]), mask, label=label)
                    for name, value in features.items():
                        if not name.startswith("original_") or ("_shape_" in name and sequence != "t1"):
                            continue
                        value = float(value)
                        row[f"{sequence}__roi{label}__{name}"] = value if np.isfinite(value) else np.nan
                except (ValueError, RuntimeError) as exc:
                    failures.append({"mr": patient.mr, "sequence": sequence, "label": label, "error": str(exc)})
        records.append(row)
    pd.DataFrame(failures, columns=["mr", "sequence", "label", "error"]).to_csv(p["radiomics"] / "extraction_errors.csv", index=False)
    if failures:
        raise ValueError("特征提取存在失败；已记录extraction_errors.csv，不发布不完整特征表。")
    table = pd.DataFrame(records)
    table.to_csv(destination, index=False)
    dump_json(p["radiomics"] / "extraction_config.json", {
        "parameters": cfg["radiomics"], "pyradiomics_version": radiomics.__version__,
        "roi_scope": "full_volume_native_geometry", "cohort_sha256": digest(p["cohort"]),
        "note": "固定逐病例提取；不使用验证患者的群体统计。全三维ROI与四层DL覆盖范围不同。"})
    record(cfg, "radiomics_features")
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
    if extraction["cohort_sha256"] != digest(p["cohort"]):
        raise ValueError("提取特征后病例清单发生变化。")
    features = pd.read_csv(p["radiomics"] / "features.csv", dtype={"mr": str})
    if not features.mr.is_unique or set(features.mr) != set(cohort.mr):
        raise ValueError("影像组学特征与固定队列不一一对应。")
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
        "cohort_sha256": digest(p["cohort"])})
    record(cfg, "radiomics_models", {"feature_selection": "inside_inner_cv"})
    return metadata


def predict_radiomics_validation(cfg):
    import joblib
    p, cohort = paths(cfg), read_cohort(cfg)
    lock = json.loads((p["radiomics"] / "frozen_artifacts.json").read_text(encoding="utf-8"))
    if lock["cohort_sha256"] != digest(p["cohort"]):
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
        table["protocol"] = "radiomics_full_volume_nested_feature_selection"
        predictions.append(table)
        summaries.append({"model": name, **classification_metrics(val.label, prob, threshold)})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(summaries)
