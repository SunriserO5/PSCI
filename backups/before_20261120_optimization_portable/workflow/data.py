"""Patient matching, image audit and portable four-slice classification data."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .project import CLINICAL, IMAGE_FILES, SEED, digest, dump_json, fresh, paths, record, require_input, source_snapshot
from .baseline import configure_baseline, load_volume, normalize_slices, prepare_temporal_main


def preparation_paths(cfg):
    mode = cfg.get("preprocessing", {}).get("mode", "full")
    if mode not in {"full", "sample"}:
        raise ValueError("preprocessing.mode 只能是 full 或 sample。")
    p = paths(cfg)
    root = p["data"] if mode == "full" else p["data"] / "sample"
    return {"root": root, "mode": mode, "matching": root / "patient_matching.csv",
            "audit": root / "volume_audit.csv", "audit_details": root / "volume_audit_details.json",
            "cache": p["cache"] if mode == "full" else root / "resnet50_manual_cache_sample.npz",
            "cache_manifest": root / "cache_manifest.json"}


def preparation_settings(cfg):
    """Only settings which determine patient selection or audit interpretation."""
    settings = cfg.get("preprocessing", {})
    return {"mode": preparation_paths(cfg)["mode"], "sample_mrs": settings.get("sample_mrs", []),
            "expected_orientation": settings.get("expected_orientation", "RAS"),
            "audit_dicom_geometry": settings.get("audit_dicom_geometry", True),
            "annotation_scope": cfg.get("annotation_scope", "unconfirmed"),
            "annotation_scope_by_mr": settings.get("annotation_scope_by_mr", {}),
            "expected_patients": int(cfg["expected_patients"])}


def excel_headers(cfg):
    return pd.read_excel(require_input(cfg, "excel_path"), sheet_name=cfg["sheet_name"], nrows=0).columns.tolist()


def normalized_mr(value) -> str:
    if pd.isna(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError("缺少有效影像号。")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value) or value < 0 or not float(value).is_integer():
            raise ValueError(f"影像号应为非负整数或 MR数字：{value!r}")
        value = str(int(value))
    match = re.fullmatch(r"(?:MR\s*)?(\d+)", str(value).strip().upper())
    if not match:
        raise ValueError(f"影像号格式无法确定：{value!r}；请使用 MR数字 或纯数字。")
    return "MR" + match.group(1)


def four_slice_indices(value, depth=None):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"取层记录不是数值：{value!r}") from None
    if not np.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError(f"起始层必须是从1开始的整数：{value!r}")
    start = int(number)
    if start > np.iinfo(np.int16).max:
        raise ValueError("起始层超出缓存支持范围。")
    zs = list(range(start - 1, start + 3))
    if depth is not None and zs[-1] >= depth:
        raise ValueError(f"所选第{start}～{start+3}层超出总层数{depth}。")
    return zs


def prepare_adapter(cfg):
    if cfg.get("data_access") == "classification_cache_only":
        raise ValueError("当前是服务器分类数据包，请直接使用已有缓存和清单。")
    p = paths(cfg)
    fresh(p["adapter"], p["data"] / "clinical_adapter_meta.json")
    source_path = require_input(cfg, "excel_path")
    source_hash = digest(source_path)
    source = pd.read_excel(source_path, sheet_name=cfg["sheet_name"])
    source.columns = source.columns.map(lambda x: str(x).strip())
    if not source.columns.is_unique:
        raise ValueError("Excel 列名去空格后重复。")
    columns = cfg["columns"]
    missing = [k for k, name in columns.items() if name is None or name not in source.columns]
    if missing:
        raise ValueError(f"列映射与真实表头不符：{missing}；现有列：{source.columns.tolist()}")
    p["data"].mkdir(parents=True, exist_ok=True)
    issues, mrs = [], []
    for i, value in enumerate(source[columns["mr"]]):
        try:
            mrs.append(normalized_mr(value))
        except ValueError as exc:
            mrs.append(None)
            issues.append({"excel_row": i + 2, "field": "mr", "reason": str(exc)})
    duplicate = pd.Series(mrs).notna() & pd.Series(mrs).duplicated(keep=False)
    for i in np.flatnonzero(duplicate):
        issues.append({"excel_row": int(i) + 2, "field": "mr", "reason": f"重复患者编号 {mrs[i]}"})
    moca = pd.to_numeric(source[columns["moca"]], errors="coerce")
    valid_outcome = moca.notna() & np.isfinite(moca) & moca.between(0, 30)
    excluded = pd.DataFrame({"excel_row": np.arange(len(source)) + 2, "mr": mrs,
                             "reason": "missing_or_invalid_moca"}).loc[~valid_outcome]
    excluded.to_csv(p["data"] / "excluded_missing_outcome.csv", index=False)
    date_values = source[columns["check_time"]]
    dates = pd.to_datetime(date_values.astype(str).str.strip().str.replace(".", "-", regex=False),
                           format="mixed", errors="coerce")
    for i in np.flatnonzero(valid_outcome & dates.isna()):
        issues.append({"excel_row": int(i) + 2, "field": "check_time", "reason": "无法解析检查日期"})
    pd.DataFrame(issues, columns=["excel_row", "field", "reason"]).to_csv(p["data"] / "clinical_issues.csv", index=False)
    if issues:
        raise ValueError("临床表存在编号或日期问题，请查看 data/clinical_issues.csv。")
    if not valid_outcome.any():
        raise ValueError("没有可用的 MoCA 结局。")
    names = [f"_unused_{i}" for i in range(33)]
    spec = {0: ("MR", "mr"), 1: ("MR_join", "mr"), 2: ("检查时间", "check_time"),
            3: ("性别", "sex"), 4: ("年龄", "age"), 5: ("Fazekas", "fazekas"),
            6: ("脑室体素最大层", "max_layer"), 13: ("受教育年限", "education"),
            23: ("NIHSS", "nihss"), 24: ("MOCA", "moca"), 25: ("同型半胱氨酸", "hcy"), 32: ("尿酸", "ua")}
    for i, (name, _) in spec.items():
        names[i] = name
    adapter = pd.DataFrame(index=source.index, columns=names)
    numeric = {"age", "fazekas", "education", "nihss", "hcy", "ua", "moca"}
    for _, (name, key) in spec.items():
        value = pd.Series(mrs, index=source.index) if key == "mr" else source[columns[key]]
        adapter[name] = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan) if key in numeric else value
    adapter["检查时间"] = dates.dt.strftime("%Y.%m.%d")
    adapter["_excel_row"] = np.arange(len(source)) + 2
    adapter = adapter.loc[valid_outcome].reset_index(drop=True)
    if digest(source_path) != source_hash:
        raise ValueError("读取期间源 Excel 发生变化，请重新准备临床工作表。")
    adapter.to_excel(p["adapter"], sheet_name="Sheet1", index=False)
    dump_json(p["data"] / "clinical_adapter_meta.json", {"source_excel_sha256": source_hash,
        "sheet_name": cfg["sheet_name"], "columns": columns, "adapter_sha256": digest(p["adapter"]),
        "source_rows": len(source), "included_rows": len(adapter), "excluded_outcome_rows": len(excluded)})
    record(cfg, "adapter", {"rows": len(adapter), "missing_outcome_exclusions": len(excluded)})
    return adapter


def read_adapter(cfg):
    p = paths(cfg)
    meta = json.loads((p["data"] / "clinical_adapter_meta.json").read_text(encoding="utf-8"))
    if (digest(p["adapter"]) != meta["adapter_sha256"] or cfg["columns"] != meta["columns"]
            or cfg["sheet_name"] != meta["sheet_name"]):
        raise ValueError("临床工作表或列映射在准备后发生变化，请核对实验目录。")
    if digest(require_input(cfg, "excel_path")) != meta["source_excel_sha256"]:
        raise ValueError("源 Excel 已更新，请使用新 run_id 重新准备数据。")
    adapter = pd.read_excel(p["adapter"], dtype={"MR": str, "MR_join": str})
    if not adapter.MR.is_unique:
        raise ValueError("临床工作表患者编号重复。")
    return adapter


def patient_directories(cfg):
    result, duplicates = {}, []
    for directory in sorted(require_input(cfg, "patients_root").iterdir()):
        match = re.search(r"MR\s*(\d+)$", directory.name.strip().upper())
        if directory.is_dir() and match:
            mr = normalized_mr("MR" + match.group(1))
            if mr in result:
                duplicates.append({"mr": mr, "first": str(result[mr].parent), "second": str(directory)})
            else:
                result[mr] = directory.resolve() / "配准图像"
    prep = preparation_paths(cfg)
    prep["root"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(duplicates, columns=["mr", "first", "second"]).to_csv(prep["root"] / "duplicate_directories.csv", index=False)
    if duplicates:
        raise ValueError("发现重复 MR 文件夹，请查看 duplicate_directories.csv。")
    return result


def match_patients(cfg):
    adapter, directories = read_adapter(cfg), patient_directories(cfg)
    prep = preparation_paths(cfg)
    selected_ids = set(adapter.MR)
    if prep["mode"] == "sample":
        requested = cfg.get("preprocessing", {}).get("sample_mrs", [])
        if not requested:
            raise ValueError("单例/小样本检查请填写 preprocessing.sample_mrs。")
        selected_ids = {normalized_mr(mr) for mr in requested}
        if not selected_ids <= set(adapter.MR):
            raise ValueError(f"样例编号在临床表中不存在：{sorted(selected_ids-set(adapter.MR))}")
    rows = []
    for _, row in adapter.iterrows():
        mr, directory = row.MR, directories.get(row.MR)
        missing = [name for name in IMAGE_FILES + ["Untitled.nii.gz"] if directory is None or not (directory / name).is_file()]
        reason = "missing_patient_directory" if directory is None else ("missing_files: " + ", ".join(missing) if missing else "")
        rows.append({"mr": mr, "excel_row": int(row["_excel_row"]), "selected": mr in selected_ids,
                     "matched": not missing, "reg_dir": str(directory) if directory else "", "reason": reason})
    table = pd.DataFrame(rows)
    table.to_csv(prep["matching"], index=False)
    pd.DataFrame([{"mr": mr, "reg_dir": str(directory)} for mr, directory in directories.items() if mr not in set(adapter.MR)],
                 columns=["mr", "reg_dir"]).to_csv(prep["root"] / "directories_without_clinical.csv", index=False)
    dump_json(prep["root"] / "matching_meta.json", {"adapter_sha256": digest(paths(cfg)["adapter"]),
        "matching_sha256": digest(prep["matching"]), "settings": preparation_settings(cfg),
        "patients_root": str(require_input(cfg, "patients_root").resolve())})
    return table


def read_matching(cfg):
    prep = preparation_paths(cfg)
    meta = json.loads((prep["root"] / "matching_meta.json").read_text(encoding="utf-8"))
    read_adapter(cfg)
    if (meta["settings"] != preparation_settings(cfg) or meta["adapter_sha256"] != digest(paths(cfg)["adapter"])
            or meta["matching_sha256"] != digest(prep["matching"])
            or meta["patients_root"] != str(require_input(cfg, "patients_root").resolve())):
        raise ValueError("患者匹配依据发生变化，请重新执行患者匹配和影像检查。")
    table = pd.read_csv(prep["matching"], dtype={"mr": str}, keep_default_na=False)
    return table.loc[table.selected.astype(str).str.lower().eq("true")].reset_index(drop=True)


def dicom_geometry(patient_dir):
    result = {}
    folders = {"t1": "T1_FLAIR_AX", "t2": "T2_PROPELLER", "dwi": "DWI_b1000"}
    try:
        import pydicom
    except ImportError:
        return {"status": "not_checked", "reason": "缺少 pydicom，未核对原始 DICOM。", "series": result}
    tags = ["Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
            "ImagePositionPatient", "ImageOrientationPatient", "NumberOfFrames"]
    for sequence, name in folders.items():
        files = sorted((Path(patient_dir) / name).glob("*.[dD][cC][mM]"))
        if not files:
            result[sequence] = {"status": "not_available"}
            continue
        try:
            datasets = [pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=tags) for f in files]
            if any(int(getattr(d, "NumberOfFrames", 1)) != 1 for d in datasets):
                raise ValueError("多帧 DICOM 需要单独解析。")
            orientations = np.asarray([d.ImageOrientationPatient for d in datasets], dtype=float)
            positions = np.asarray([d.ImagePositionPatient for d in datasets], dtype=float)
            if not np.isfinite(orientations).all() or not np.isfinite(positions).all():
                raise ValueError("DICOM 方向或位置包含无效数值。")
            if not np.allclose(orientations, orientations[0], atol=1e-4, rtol=0):
                raise ValueError("同一序列内切片方向不一致。")
            normal = np.cross(orientations[0, :3], orientations[0, 3:])
            if np.linalg.norm(normal) < 1e-8:
                raise ValueError("DICOM 切片方向无效。")
            positions_along_normal = np.sort(positions @ (normal / np.linalg.norm(normal)))
            distance = np.diff(positions_along_normal)
            if not len(distance) or np.any(distance < 1e-3):
                raise ValueError("切片位置重复或不足两层。")
            pixel_spacing = np.asarray(datasets[0].PixelSpacing, dtype=float)
            thickness = float(datasets[0].SliceThickness) if hasattr(datasets[0], "SliceThickness") else None
            declared_spacing = float(datasets[0].SpacingBetweenSlices) if hasattr(datasets[0], "SpacingBetweenSlices") else None
            if pixel_spacing.shape != (2,) or not np.isfinite(pixel_spacing).all() or np.any(pixel_spacing <= 0):
                raise ValueError("DICOM 像素间距无效。")
            if any(value is not None and not np.isfinite(value) for value in (thickness, declared_spacing)):
                raise ValueError("DICOM 层厚或间距包含无效数值。")
            result[sequence] = {"status": "checked", "files": len(files),
                "shape_2d": [int(datasets[0].Rows), int(datasets[0].Columns)],
                "pixel_spacing_mm": pixel_spacing.tolist(),
                "slice_thickness_mm": thickness,
                "declared_slice_spacing_mm": declared_spacing,
                "measured_slice_spacing_mm": float(np.median(distance)),
                "spacing_range_mm": [float(distance.min()), float(distance.max())]}
        except (OSError, EOFError, ValueError, TypeError, AttributeError, KeyError, pydicom.errors.InvalidDicomError) as exc:
            result[sequence] = {"status": "not_checked", "reason": str(exc)}
    return {"status": "complete" if all(v["status"] == "checked" for v in result.values()) else "incomplete", "series": result}


def inspect_patient(cfg, patient, directory):
    import nibabel as nib
    directory = Path(directory)
    names = IMAGE_FILES + ["Untitled.nii.gz"]
    hashes_before = {name: digest(directory / name) for name in names}
    volumes = [nib.load(str(directory / name)) for name in names]
    ref = volumes[0]
    if len(ref.shape) != 3 or min(ref.shape) < 1 or min(ref.shape[:2]) < 2:
        raise ValueError(f"需要三维影像，实际形状为 {ref.shape}。")
    if not np.isfinite(ref.affine).all() or abs(np.linalg.det(ref.affine[:3, :3])) < 1e-12:
        raise ValueError("参考影像的空间变换无效。")
    if any(v.shape != ref.shape or not np.allclose(v.affine, ref.affine, atol=1e-4, rtol=0) for v in volumes[1:]):
        raise ValueError("三序列影像与人工标注的 shape/affine 不一致。")
    orientation = "".join(nib.aff2axcodes(ref.affine))
    expected_orientation = cfg.get("preprocessing", {}).get("expected_orientation", "RAS")
    if expected_orientation and orientation != expected_orientation:
        raise ValueError(f"影像方向为 {orientation}，与配置的 {expected_orientation} 不同；需人工核对，不自动翻转。")
    zs = four_slice_indices(patient["脑室体素最大层"], ref.shape[2])
    arrays = [np.asarray(v.dataobj) for v in volumes]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("影像或标签包含 NaN/Inf。")
    label_values, counts = np.unique(arrays[-1], return_counts=True)
    if not set(label_values) <= {0, 1, 2, 3}:
        raise ValueError(f"人工标签需要0/1/2/3，实际为 {label_values.tolist()}。")
    geometry_warnings, ranges, files = [], {}, {}
    for name, volume, array in zip(names, volumes, arrays):
        units = volume.header.get_xyzt_units()[0]
        spacing = np.asarray(volume.header.get_zooms()[:3], dtype=float)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError(f"{name} 的体素间距无效。")
        if units != "mm":
            geometry_warnings.append(f"{name}:空间单位为 {units}，物理尺度待核对")
        qform, qcode = volume.get_qform(coded=True)
        sform, scode = volume.get_sform(coded=True)
        if qcode and scode and not np.allclose(qform, sform, atol=1e-3, rtol=0):
            geometry_warnings.append(f"{name}:qform 与 sform 不一致")
        affine_spacing = np.linalg.norm(volume.affine[:3, :3], axis=0)
        if not np.allclose(spacing, affine_spacing, atol=1e-4, rtol=1e-4):
            geometry_warnings.append(f"{name}:头文件体素间距与空间变换不一致")
        files[name] = {"sha256": hashes_before[name], "bytes": (directory / name).stat().st_size,
                       "shape": list(volume.shape), "spacing": spacing.tolist(), "spatial_unit": units,
                       "affine": volume.affine.tolist(), "orientation": "".join(nib.aff2axcodes(volume.affine)),
                       "qform_code": int(qcode), "sform_code": int(scode)}
        if name in IMAGE_FILES:
            with np.errstate(over="ignore", invalid="ignore"):
                image = array.astype(np.float32)
            if not np.isfinite(image).all():
                raise ValueError(f"{name} 转为 float32 后出现无效值。")
            lo, hi = np.percentile(image[:, :, zs], [1, 99.5])
            ranges[name] = {"percentile_1": float(lo), "percentile_99_5": float(hi)}
    warnings = [f"{name}:选定四层的强度范围退化，归一化将沿用原代码的微小分母处理"
                for name, limits in ranges.items() if limits["percentile_99_5"] <= limits["percentile_1"]]
    labels = arrays[-1]
    selected_mask = labels[:, :, zs]
    scope = cfg.get("preprocessing", {}).get("annotation_scope_by_mr", {}).get(str(patient.MR), cfg.get("annotation_scope", "unconfirmed"))
    if scope not in {"selected_slices", "full_volume", "unconfirmed"}:
        raise ValueError(f"{patient.MR} 的标注范围配置无效：{scope}")
    outside = int(np.count_nonzero(labels) - np.count_nonzero(selected_mask))
    if scope == "selected_slices" and outside:
        warnings.append("选定四层外仍有标签，请核对标注范围记录")
    if not np.count_nonzero(selected_mask):
        warnings.append("选定四层的人工标签全空，请核对；不据此自动排除患者")
    dicom = dicom_geometry(directory.parent) if cfg.get("preprocessing", {}).get("audit_dicom_geometry", True) else {"status": "not_requested", "series": {}}
    if dicom["status"] != "complete":
        geometry_warnings.append("原始 DICOM 空间信息未完成全部核对")
    for sequence, name in zip(("t1", "t2", "dwi"), IMAGE_FILES):
        geometry = dicom["series"].get(sequence, {})
        if geometry.get("status") == "checked":
            if not np.isclose(geometry["measured_slice_spacing_mm"], files[name]["spacing"][2], atol=0.05, rtol=0):
                geometry_warnings.append(f"{name}:DICOM层中心间距{geometry['measured_slice_spacing_mm']:.4f}，NIfTI第三轴间距{files[name]['spacing'][2]:.4f}，需核查转换/重采样记录")
            if geometry["spacing_range_mm"][1] - geometry["spacing_range_mm"][0] > 0.05:
                geometry_warnings.append(f"{name}:DICOM 各层中心间距不均匀")
    if hashes_before != {name: digest(directory / name) for name in names}:
        raise ValueError("影像检查期间源文件发生变化，请重新检查。")
    return {"mr": str(patient.MR), "files": files, "selected_slices_1based": [z + 1 for z in zs],
            "annotation_scope": scope, "nonzero_label_slices_1based": (np.flatnonzero((labels != 0).any(axis=(0, 1))) + 1).tolist(),
            "label_counts_native": {str(int(v)): int(c) for v, c in zip(label_values, counts)},
            "label_counts_selected": {str(k): int(np.count_nonzero(selected_mask == k)) for k in (1, 2, 3)},
            "foreground_outside_selected": outside, "normalization": ranges,
            "dicom_geometry": dicom, "warnings": geometry_warnings + warnings,
            "physical_geometry_needs_review": bool(geometry_warnings)}


def audit_volumes(cfg):
    import nibabel as nib
    prep, matching = preparation_paths(cfg), read_matching(cfg)
    adapter = read_adapter(cfg).set_index("MR", drop=False)
    rows, details = [], {}
    for _, match in matching.iterrows():
        row = {"mr": match.mr, "eligible": False, "reason": "", "warnings": "",
               "physical_geometry_needs_review": True, "annotation_scope": "unconfirmed"}
        try:
            if str(match.matched).lower() != "true":
                raise ValueError(match.reason)
            info = inspect_patient(cfg, adapter.loc[match.mr], match.reg_dir)
            details[match.mr] = info
            reference = info["files"][IMAGE_FILES[0]]
            row.update(eligible=True, warnings="；".join(info["warnings"]),
                physical_geometry_needs_review=info["physical_geometry_needs_review"], annotation_scope=info["annotation_scope"],
                shape=json.dumps(reference["shape"]), spacing=json.dumps(reference["spacing"]), orientation=reference["orientation"],
                selected_slices_1based=json.dumps(info["selected_slices_1based"]),
                label_1_voxels=info["label_counts_selected"]["1"], label_2_voxels=info["label_counts_selected"]["2"],
                label_3_voxels=info["label_counts_selected"]["3"], foreground_outside_selected=info["foreground_outside_selected"])
        except (OSError, EOFError, ValueError, TypeError, KeyError, nib.filebasedimages.ImageFileError) as exc:
            row["reason"] = str(exc)
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(prep["audit"], index=False)
    dump_json(prep["audit_details"], {"adapter_sha256": digest(paths(cfg)["adapter"]),
        "matching_sha256": digest(prep["matching"]), "audit_sha256": digest(prep["audit"]),
        "settings": preparation_settings(cfg), "notebook_definitions_sha256": source_snapshot(cfg), "patients": details})
    return audit


def checked_inputs(cfg):
    prep, matching = preparation_paths(cfg), read_matching(cfg)
    audit = pd.read_csv(prep["audit"], dtype={"mr": str}, keep_default_na=False)
    details = json.loads(prep["audit_details"].read_text(encoding="utf-8"))
    if (details["adapter_sha256"] != digest(paths(cfg)["adapter"]) or details["matching_sha256"] != digest(prep["matching"])
            or details["audit_sha256"] != digest(prep["audit"]) or details["settings"] != preparation_settings(cfg)
            or details["notebook_definitions_sha256"] != source_snapshot(cfg)):
        raise ValueError("核查记录的输入、设置或代码定义已变化，请重新检查影像。")
    if not audit.mr.is_unique or audit.mr.tolist() != matching.mr.tolist():
        raise ValueError("核查表与本次选定患者不一致。")
    if not audit.eligible.astype(str).str.lower().eq("true").all():
        raise ValueError("存在未通过检查的患者，请先处理 volume_audit.csv 的 reason；不静默跳过。")
    if prep["mode"] == "full" and len(matching) != int(cfg["expected_patients"]):
        raise ValueError(f"全量模式需要{cfg['expected_patients']}例，当前{len(matching)}例；样例检查请使用 sample 模式。")
    if not len(matching) or set(details["patients"]) != set(matching.mr):
        raise ValueError("核查明细缺少患者。")
    for _, row in matching.iterrows():
        for filename, info in details["patients"][row.mr]["files"].items():
            if digest(Path(row.reg_dir) / filename) != info["sha256"]:
                raise ValueError(f"{row.mr}/{filename} 在核查后发生变化，请重新检查。")
    return matching, read_adapter(cfg).set_index("MR", drop=False), details


def cache_patient(directory, max_layer):
    """Same percentile normalization and scipy zoom calls as the original code."""
    import nibabel as nib
    from scipy.ndimage import zoom
    directory = Path(directory)
    volumes = [load_volume(directory / name) for name in IMAGE_FILES]
    label_volume = np.asarray(nib.load(str(directory / "Untitled.nii.gz")).dataobj)
    if any(v.shape != volumes[0].shape for v in volumes) or label_volume.shape != volumes[0].shape:
        raise ValueError("缓存读取时影像形状发生变化。")
    if any(not np.isfinite(v).all() for v in volumes) or not np.isfinite(label_volume).all():
        raise ValueError("缓存读取时发现非有限值。")
    if not set(np.unique(label_volume)) <= {0, 1, 2, 3}:
        raise ValueError("人工标签值不合法。")
    label_volume = label_volume.astype(np.uint8)
    zs = four_slice_indices(max_layer, volumes[0].shape[2])
    norm_volumes = [normalize_slices(volume, zs) for volume in volumes]
    h, w = volumes[0].shape[:2]
    images = np.empty((4, 3, 224, 224), dtype=np.float16)
    masks = np.zeros((4, 3, 224, 224), dtype=np.uint8)
    for s, z in enumerate(zs):
        for c, volume in enumerate(norm_volumes):
            images[s, c] = zoom(volume[:, :, s], (224 / h, 224 / w), order=1, prefilter=False).astype(np.float16)
        for label in range(1, 4):
            masks[s, label - 1] = zoom((label_volume[:, :, z] == label).astype(np.uint8),
                                     (224 / h, 224 / w), order=0, prefilter=False)
    return images, masks


def validate_cache_file(cache_path, expected_mrs=None, expected_labels=None, expected_layers=None):
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"images", "masks", "labels", "mrs", "max_layers"}
        if not required <= set(cache.files):
            raise ValueError(f"缓存缺字段：{sorted(required-set(cache.files))}")
        images, masks = cache["images"], cache["masks"]
        mrs, labels, layers = cache["mrs"].astype(str), cache["labels"], cache["max_layers"]
        if mrs.ndim != 1 or len(set(mrs)) != len(mrs) or any(normalized_mr(x) != x for x in mrs):
            raise ValueError("缓存编号重复、维度错误或格式不统一。")
        shape = (len(mrs), 4, 3, 224, 224)
        if images.shape != shape or masks.shape != shape or labels.shape != (len(mrs),) or layers.shape != (len(mrs),):
            raise ValueError("缓存形状不符合四层三通道格式。")
        if images.dtype != np.float16 or masks.dtype != np.uint8 or labels.dtype != np.int64 or layers.dtype != np.int16:
            raise ValueError("缓存数据类型与原格式不一致。")
        if not np.isfinite(images).all() or np.any((images < 0) | (images > 1)) or not set(np.unique(masks)) <= {0, 1}:
            raise ValueError("缓存影像需在[0,1]，掩模需为二值。")
        if not set(labels) <= {0, 1} or np.any(layers < 1):
            raise ValueError("缓存结局或取层记录无效。")
        for actual, expected, name in ((mrs, expected_mrs, "患者顺序"), (labels, expected_labels, "结局"), (layers, expected_layers, "取层")):
            if expected is not None and not np.array_equal(actual, np.asarray(expected)):
                raise ValueError(f"缓存{name}与当前临床和核查记录不一致。")
        return {"patients": len(mrs), "shape": list(shape), "positive": int(labels.sum()),
                "uncompressed_array_bytes": int(images.nbytes + masks.nbytes), "file_bytes": Path(cache_path).stat().st_size}


def run_original_preprocessing(cfg):
    prep = preparation_paths(cfg)
    if prep["mode"] == "full" and cfg.get("existing_cache"):
        raise ValueError("已配置 existing_cache；请用核查已有缓存单元，不覆盖文件。")
    fresh(prep["cache"], prep["cache_manifest"], prep["cache"].with_suffix(".partial.npz"))
    matching, adapter, details = checked_inputs(cfg)
    images, masks, labels, layers = [], [], [], []
    for i, row in matching.iterrows():
        patient = adapter.loc[row.mr]
        image, mask = cache_patient(row.reg_dir, patient["脑室体素最大层"])
        images.append(image)
        masks.append(mask)
        labels.append(int(float(patient.MOCA) <= 22))
        layers.append(four_slice_indices(patient["脑室体素最大层"])[0] + 1)
        if (i + 1) % 20 == 0 or i + 1 == len(matching):
            print(f"已生成 {i+1}/{len(matching)} 例四层输入", flush=True)
    # Recheck source bytes after reading all images, before publishing a cache.
    checked_inputs(cfg)
    prep["root"].mkdir(parents=True, exist_ok=True)
    partial = prep["cache"].with_suffix(".partial.npz")
    with partial.open("xb") as stream:
        np.savez_compressed(stream, images=np.stack(images), masks=np.stack(masks), labels=np.asarray(labels, dtype=np.int64),
                            mrs=matching.mr.to_numpy(dtype=str), max_layers=np.asarray(layers, dtype=np.int16))
    summary = validate_cache_file(partial, matching.mr.tolist(), labels, layers)
    partial.rename(prep["cache"])
    dump_json(prep["cache_manifest"], {"cache_sha256": digest(prep["cache"]), "audit_details_sha256": digest(prep["audit_details"]),
        "adapter_sha256": digest(paths(cfg)["adapter"]), "settings": preparation_settings(cfg), "summary": summary,
        "method": "original_four_slices_percentile_1_99.5_linear_images_nearest_masks_224",
        "patients": [{"mr": mr, "selected_slices_1based": details["patients"][mr]["selected_slices_1based"],
                       "annotation_scope": details["patients"][mr]["annotation_scope"]} for mr in matching.mr]})
    record(cfg, "preprocessing_" + prep["mode"], summary)
    return summary


def verify_existing_cache(cfg):
    prep = preparation_paths(cfg)
    fresh(prep["cache_manifest"])
    matching, adapter, details = checked_inputs(cfg)
    patients = adapter.loc[matching.mr]
    labels = (patients.MOCA.to_numpy(dtype=float) <= 22).astype(np.int64)
    layers = np.asarray([four_slice_indices(v)[0] + 1 for v in patients["脑室体素最大层"]], dtype=np.int16)
    summary = validate_cache_file(prep["cache"], matching.mr.tolist(), labels, layers)
    with np.load(prep["cache"], allow_pickle=False) as cache:
        images, masks = cache["images"], cache["masks"]
        for i, row in matching.iterrows():
            expected_images, expected_masks = cache_patient(row.reg_dir, layers[i])
            if not np.array_equal(images[i], expected_images) or not np.array_equal(masks[i], expected_masks):
                raise ValueError(f"已有缓存与当前原生影像重新计算的结果不同：{row.mr}。")
    checked_inputs(cfg)
    dump_json(prep["cache_manifest"], {"cache_sha256": digest(prep["cache"]), "audit_details_sha256": digest(prep["audit_details"]),
        "adapter_sha256": digest(paths(cfg)["adapter"]), "settings": preparation_settings(cfg), "summary": summary,
        "method": "existing_cache_compared_patient_by_patient_to_current_source",
        "patients": [{"mr": mr, "selected_slices_1based": details["patients"][mr]["selected_slices_1based"],
                       "annotation_scope": details["patients"][mr]["annotation_scope"]} for mr in matching.mr]})
    return summary


def verify_preparation(cfg):
    prep = preparation_paths(cfg)
    matching, adapter, details = checked_inputs(cfg)
    manifest = json.loads(prep["cache_manifest"].read_text(encoding="utf-8"))
    if (manifest["cache_sha256"] != digest(prep["cache"]) or manifest["audit_details_sha256"] != digest(prep["audit_details"])
            or manifest["adapter_sha256"] != digest(paths(cfg)["adapter"]) or manifest["settings"] != preparation_settings(cfg)):
        raise ValueError("缓存、临床表或核查记录与生成时的版本不同。")
    patients = adapter.loc[matching.mr]
    validate_cache_file(prep["cache"], matching.mr.tolist(), (patients.MOCA.to_numpy(dtype=float) <= 22).astype(int),
                        [four_slice_indices(v)[0] + 1 for v in patients["脑室体素最大层"]])
    return matching, adapter, details


def prepare_split_and_cohort(cfg):
    from sklearn.model_selection import StratifiedKFold
    if preparation_paths(cfg)["mode"] != "full":
        raise ValueError("样例缓存仅用于检查，不生成时间划分或训练队列。")
    p = paths(cfg)
    fresh(p["cohort"], p["temporal"] / "temporal_split.csv", p["data"] / "prepared_data_manifest.json")
    matching, adapter, details = verify_preparation(cfg)
    if len(matching) != 206 or int(cfg["expected_patients"]) != 206:
        raise ValueError("当前保留原206例时间划分，不根据缺失病例重新定义研究队列。")
    with np.load(p["cache"], allow_pickle=False) as cache:
        mrs, y, layers = cache["mrs"].astype(str), cache["labels"].astype(int), cache["max_layers"]
    dates = pd.to_datetime(adapter.loc[mrs, "检查时间"], format="%Y.%m.%d")
    ordered = pd.DataFrame({"mr": mrs, "date": dates.to_numpy()}).sort_values(["date", "mr"], kind="mergesort")
    boundary = str(ordered.iloc[143]["date"].date())
    if boundary != cfg["expected_boundary"]:
        raise ValueError(f"原第144例分界日期为{boundary}，与配置不同，请先核对患者。")
    configure_baseline(p["cache"], p["full_cv"], p["temporal"], p["adapter"], cfg.get("patients_root"))
    prepare_temporal_main()
    cohort = pd.read_csv(p["temporal"] / "temporal_split.csv", dtype={"mr": str})
    cohort["cache_index"] = cohort.mr.map({mr: i for i, mr in enumerate(mrs)})
    cohort["label"] = y[cohort.cache_index.to_numpy(dtype=int)]
    cohort["moca"] = adapter.loc[cohort.mr, "MOCA"].to_numpy(dtype=float)
    cohort["max_layer"] = layers[cohort.cache_index.to_numpy(dtype=int)]
    directories = matching.set_index("mr").reg_dir
    for sequence, filename in zip(("t1", "t2", "dwi", "manual_mask"), IMAGE_FILES + ["Untitled.nii.gz"]):
        cohort[sequence] = [str(Path(directories[mr]) / filename) for mr in cohort.mr]
    clinical_names = {"age": "年龄", "education": "受教育年限", "nihss": "NIHSS", "hcy": "同型半胱氨酸", "ua": "尿酸"}
    for key in CLINICAL:
        cohort[key] = pd.to_numeric(adapter.loc[cohort.mr, clinical_names[key]], errors="coerce").to_numpy()
    cohort["selected_slices_1based"] = [json.dumps(details["patients"][mr]["selected_slices_1based"]) for mr in cohort.mr]
    cohort["annotation_scope"] = [details["patients"][mr]["annotation_scope"] for mr in cohort.mr]
    cohort["physical_geometry_needs_review"] = [details["patients"][mr]["physical_geometry_needs_review"] for mr in cohort.mr]
    cohort["native_images_available"] = True
    cohort["development_fold"] = 0
    dev_index = cohort.index[cohort.split == "development"].to_numpy()
    dev_y = cohort.loc[dev_index, "label"].to_numpy()
    if np.bincount(dev_y, minlength=2).min() < 5:
        raise ValueError("开发队列类别数量不足五折。")
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev_y, dev_y), 1):
        cohort.loc[dev_index[valid], "development_fold"] = fold
    cohort.to_csv(p["cohort"], index=False)
    dump_json(p["data"] / "cohort_summary.json", {"n": len(cohort), "split": cohort.split.value_counts().to_dict(), "boundary": boundary})
    prepared_files = [p["cohort"], p["adapter"], p["temporal"] / "temporal_split.csv", preparation_paths(cfg)["cache_manifest"]]
    dump_json(p["data"] / "prepared_data_manifest.json", {"artifacts": {str(f.relative_to(p["run"])): digest(f) for f in prepared_files},
        "cache_sha256": digest(p["cache"]), "notebook_definitions_sha256": source_snapshot(cfg), "boundary": boundary})
    record(cfg, "cohort")
    return cohort


def export_classification_bundle(cfg):
    import shutil
    p = paths(cfg)
    if preparation_paths(cfg)["mode"] != "full":
        raise ValueError("样例模式不导出完整训练数据包。")
    verify_preparation(cfg)
    prepared = json.loads((p["data"] / "prepared_data_manifest.json").read_text(encoding="utf-8"))
    if prepared["cache_sha256"] != digest(p["cache"]) or prepared["notebook_definitions_sha256"] != source_snapshot(cfg):
        raise ValueError("数据准备完成后缓存或代码定义改变，请核对实验版本。")
    for name, checksum in prepared["artifacts"].items():
        if digest(p["run"] / name) != checksum:
            raise ValueError(f"准备好的数据文件发生变化：{name}")
    notebooks = sorted(cfg["_root"].glob("*.ipynb"))
    if len(notebooks) != 7:
        raise ValueError("请将7本 Notebook 保存在同一目录后导出。")
    destination = cfg["_root"] / "upload_packages" / (cfg["run_id"] + "_classification")
    fresh(destination)
    target_run = destination / "outputs" / cfg["run_id"]
    (target_run / "data").mkdir(parents=True)
    (target_run / "temporal").mkdir()
    filenames = ["data/clinical_adapter.xlsx", "data/clinical_adapter_meta.json", "data/cohort_summary.json",
        "data/patient_matching.csv", "data/matching_meta.json", "data/volume_audit.csv", "data/volume_audit_details.json",
        "data/cache_manifest.json", "data/excluded_missing_outcome.csv", "data/clinical_issues.csv",
        "temporal/temporal_split.csv", "temporal/baseline_comparison.csv", "temporal/temporal_split_summary.md"]
    artifacts = {}
    for name in filenames:
        source, target = p["run"] / name, target_run / name
        shutil.copy2(source, target)
        if digest(source) != digest(target):
            raise ValueError(f"复制校验失败：{name}")
        artifacts[str(target.relative_to(destination))] = digest(target)
    # Keep the local preparation manifest as provenance; its cohort contains native paths.
    source_preparation = target_run / "data" / "source_prepared_data_manifest.json"
    shutil.copy2(p["data"] / "prepared_data_manifest.json", source_preparation)
    artifacts[str(source_preparation.relative_to(destination))] = digest(source_preparation)
    target_cache = target_run / "data" / "resnet50_manual_cache.npz"
    shutil.copy2(p["cache"], target_cache)
    if digest(target_cache) != prepared["cache_sha256"]:
        raise ValueError("上传包中的缓存复制不完整。")
    artifacts[str(target_cache.relative_to(destination))] = digest(target_cache)
    cohort = pd.read_csv(p["cohort"], dtype={"mr": str})
    for column in ("t1", "t2", "dwi", "manual_mask"):
        cohort[column] = ""
    cohort["native_images_available"] = False
    cohort_path = target_run / "data" / "cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    artifacts[str(cohort_path.relative_to(destination))] = digest(cohort_path)
    for notebook in notebooks:
        content = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in content["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"], cell["execution_count"] = [], None
                if not cell.get("metadata", {}).get("embedded_definition"):
                    text = "".join(cell["source"])
                    text = re.sub(r"(?m)^((?:RUN_\w+|LOCK_ORIGINAL_MODELS|INSTALL_SPLITS|EVALUATE_DEVELOPMENT_SEGMENTATION|SHOW_CACHE_EXAMPLE|COMPARE_DEVELOPMENT_OOF)\s*=\s*)True\b", r"\g<1>False", text)
                    if notebook.name.startswith("06_"):
                        text = text.replace("INCLUDE_RADIOMICS = True", "INCLUDE_RADIOMICS = False")
                    cell["source"] = text.splitlines(keepends=True)
        dump_json(destination / notebook.name, content)
    config = {k: v for k, v in cfg.items() if not k.startswith("_")}
    config.update(excel_path=None, patients_root=None, existing_cache=None, outputs_root="outputs",
                  data_access="classification_cache_only")
    dump_json(destination / "config.json", config)
    if (cfg["_root"] / "requirements.txt").is_file():
        shutil.copy2(cfg["_root"] / "requirements.txt", destination / "requirements.txt")
    dump_json(destination / "classification_bundle.json", {"run_id": cfg["run_id"], "kind": "classification_cache_only",
        "artifacts": artifacts, "cache": str(target_cache.relative_to(destination)), "cohort": str(cohort_path.relative_to(destination)),
        "source_cohort_sha256": digest(p["cohort"]), "notebook_definitions_sha256": source_snapshot(cfg),
        "patients": len(cohort), "split": cohort.split.value_counts().to_dict(), "boundary": prepared["boundary"],
        "native_images_included": False, "note": "原生路径已清空；本包仅供分类、单序列消融和缓存空间中的解释。"})
    (destination / "使用说明.md").write_text("# 分类数据包\n\n上传整个目录并在服务器解压或打开。配置读取时会核对输入文件校验值。\n\n"
        "无需在服务器重跑00的数据准备，依次使用02训练、03单序列、04热图、06分类时间验证。"
        "所有运行开关默认关闭；06默认不纳入影像组学或自动掩模。\n\n"
        "此包含四层224×224缓存、临床工作表、患者分组和核查记录，不含原生三维影像。"
        "01和05需另行准备原分辨率数据与适合局部标注的流程。核查记录中的本地原始路径仅用于追溯。\n",
        encoding="utf-8")
    return {"directory": str(destination), "patients": len(cohort),
            "bytes": sum(f.stat().st_size for f in destination.rglob("*") if f.is_file()), "input_files": len(artifacts)}
