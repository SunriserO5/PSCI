"""Native inputs for the confirmed four annotated slices; no unannotated background."""
from pathlib import Path
import json
import re
import numpy as np
from .project import IMAGE_FILES, SEQUENCES, digest, require_native_images, paths


def native_patient_files(cfg, patient):
    require_native_images(cfg)
    columns = SEQUENCES + ["manual_mask"]
    recorded = [Path(str(patient.get(c, ""))) for c in columns]
    if all(p.is_file() for p in recorded):
        return recorded
    root = Path(cfg.get("patients_root") or "")
    if not root.is_dir():
        raise FileNotFoundError("在 config.json 填写数据电脑上的 patients_root。")
    if "_native_lookup" not in cfg:
        lookup = {}
        for file in root.rglob(IMAGE_FILES[0]):
            matches = set(re.findall(r"(?<![A-Z0-9])MR\d+(?!\d)", str(file.parent).upper()))
            for mr in matches:
                lookup.setdefault(mr, []).append(file.parent)
        cfg["_native_lookup"] = lookup
    matches = cfg["_native_lookup"].get(str(patient.mr).upper(), [])
    if len(matches) != 1:
        raise ValueError(f"{patient.mr} 未找到唯一的三序列目录；检查 patients_root 和目录中的 MR 编号。")
    files = [matches[0] / name for name in IMAGE_FILES + ["Untitled.nii.gz"]]
    if not all(file.is_file() for file in files):
        raise FileNotFoundError(f"{patient.mr} 缺少三序列影像或人工标注。")
    return files


def selected_indices(patient, depth):
    value = float(patient.max_layer)
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"{patient.mr} 起始层不是整数。")
    start = int(value) - 1
    zs = list(range(start, start + 4))
    if start < 0 or zs[-1] >= depth:
        raise ValueError(f"{patient.mr} 的四层取层越界。")
    recorded = patient.get("selected_slices_1based")
    if isinstance(recorded, str) and recorded.strip() and json.loads(recorded) != [z + 1 for z in zs]:
        raise ValueError(f"{patient.mr} 的缓存取层与标注层记录不一致。")
    return zs


def selected_patient(cfg, patient):
    import nibabel as nib
    if cfg.get("annotation_scope") != "selected_slices":
        raise ValueError("本流程要求确认全队列仅选定四层标注。")
    files = native_patient_files(cfg, patient)
    volumes = [nib.load(str(file)) for file in files]
    ref = volumes[0]
    tolerance = np.array([1e-4, 1e-4, 1e-4, 1e-3])
    if len(ref.shape) != 3 or any(v.shape != ref.shape or not np.allclose(v.affine, ref.affine, rtol=0, atol=tolerance) for v in volumes[1:]):
        raise ValueError(f"{patient.mr} 的 shape/affine 不一致，先回到00核查。")
    zs = selected_indices(patient, ref.shape[2])
    truth = np.asarray(volumes[-1].dataobj)
    if not np.isfinite(truth).all() or not set(np.unique(truth)) <= {0, 1, 2, 3}:
        raise ValueError(f"{patient.mr} 人工标签必须为0/1/2/3。")
    outside = np.ones(ref.shape[2], dtype=bool)
    outside[zs] = False
    if np.any(truth[:, :, outside] != 0):
        raise ValueError(f"{patient.mr} 在指定四层之外发现标注，请核对起始层。")
    images = [np.asarray(v.dataobj, dtype=np.float32)[:, :, zs] for v in volumes[:3]]
    if any(not np.isfinite(v).all() for v in images):
        raise ValueError(f"{patient.mr} 影像包含非有限值。")
    checksums = {file.name: digest(file) for file in files}
    audit_file = paths(cfg)["data"] / "volume_audit_details.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    expected = audit["patients"][str(patient.mr)]["files"]
    if any(checksums[name] != expected[name]["sha256"] for name in checksums):
        raise ValueError(f"{patient.mr} 原始影像或标注与00的缓存来源不同，请先核查数据版本。")
    return images, truth[:, :, zs].astype(np.uint8), {
        "mr": str(patient.mr), "slices_1based": [z + 1 for z in zs], "shape": list(ref.shape),
        "native_affine": ref.affine.tolist(), "working_space": "pixel",
        "files_sha256": checksums}
