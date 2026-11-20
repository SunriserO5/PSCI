"""Configuration, provenance and stage boundaries; standard library only."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SEED = 20260922
CLINICAL = ["age", "education", "nihss", "hcy", "ua"]
SEQUENCES = ["t1", "t2", "dwi"]
IMAGE_FILES = ["t1_flair.nii.gz", "t2.nii.gz", "dwi_b1000.nii.gz"]


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    settings = path.parent / "experiment_settings.json"
    if settings.is_file():
        overlay = json.loads(settings.read_text(encoding="utf-8"))
        allowed = {"annotation_scope", "optimized", "radiomics", "segmentation", "report_version"}
        if set(overlay) - allowed:
            raise ValueError("experiment_settings.json 只放实验参数，机器路径仍在 config.json。")
        for key, value in overlay.items():
            if isinstance(value, dict):
                cfg[key] = {**cfg.get(key, {}), **value}
            else:
                cfg[key] = value
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["run_id"]):
        raise ValueError("run_id 只使用字母、数字、下划线或短横线。")
    cfg["_root"] = path.parent
    for key in ("excel_path", "patients_root", "existing_cache", "outputs_root"):
        if cfg.get(key):
            p = Path(cfg[key]).expanduser()
            cfg[key] = p if p.is_absolute() else path.parent / p
    cfg["_run"] = cfg["outputs_root"] / cfg["run_id"]
    if cfg.get("data_access") == "classification_cache_only":
        verify_classification_bundle(cfg)
    return cfg


def paths(cfg: dict) -> dict[str, Path]:
    root = cfg["_run"]
    return {
        "run": root, "data": root / "data", "cache": Path(cfg.get("existing_cache") or root / "data" / "resnet50_manual_cache.npz"),
        "adapter": root / "data" / "clinical_adapter.xlsx", "cohort": root / "data" / "cohort.csv",
        "temporal": root / "temporal", "full_cv": root / "full_cv", "single": root / "single_sequence",
        "radiomics": root / "radiomics" / experiment_name(cfg, "radiomics", "selected2d_v1"),
        "explain": root / "explain", "segmentation": root / "segmentation" / experiment_name(cfg, "segmentation", "selected2d_v1"),
        "optimized": root / "optimized" / experiment_name(cfg, "optimized", "resnet_v1"),
        "reports": root / "reports" / safe_name(cfg.get("report_version", "comparison_v2")),
    }


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def fresh(*targets: Path) -> None:
    exists = [str(p) for p in targets if p.exists()]
    if exists:
        raise FileExistsError("产物已存在，请加载已有结果，或更换 run_id：" + ", ".join(exists))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_snapshot(cfg: dict) -> dict:
    """Hash the literal definition cells, independently of outputs and execution counts."""
    result = {}
    for path in sorted(cfg["_root"].glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        definitions = ["".join(cell["source"]) for cell in notebook["cells"]
                       if cell["cell_type"] == "code" and cell.get("metadata", {}).get("embedded_definition")]
        result[path.name] = hashlib.sha256("\n\n".join(definitions).encode("utf-8")).hexdigest()
    return result


def record(cfg: dict, stage: str, extra: dict | None = None) -> None:
    obj = {"stage": stage, "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
           "notebook_definitions_sha256": source_snapshot(cfg),
           "protocol": "resnet_baseline_and_optimized_selected_slices", **(extra or {})}
    dump_json(paths(cfg)["run"] / "provenance" / f"{stage}.json", obj)


def read_cohort(cfg: dict):
    import pandas as pd
    p = paths(cfg)["cohort"]
    if not p.exists():
        raise FileNotFoundError("先运行00的数据准备与病例清单单元。")
    table = pd.read_csv(p, dtype={"mr": str})
    if not table["mr"].is_unique or not set(table["split"]) <= {"development", "validation"}:
        raise ValueError("病例清单编号重复或队列标记无效。")
    return table


def development(cfg: dict):
    return read_cohort(cfg).query("split == 'development'").reset_index(drop=True)


def require_input(cfg: dict, key: str) -> Path:
    value = cfg.get(key)
    if not value or not Path(value).exists():
        raise FileNotFoundError(f"请在 config.json 配置有效的 {key}。")
    return Path(value)


def verify_classification_bundle(cfg: dict) -> None:
    """Verify portable data without accessing the original image directories."""
    root = cfg["_root"].resolve()
    manifest_path = root / "classification_bundle.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("分类数据包缺少 classification_bundle.json，请上传完整目录。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "classification_cache_only" or manifest["run_id"] != cfg["run_id"]:
        raise ValueError("分类数据包与当前 run_id 不一致。")
    manifest["artifacts"] = {portable_name(k): v for k, v in manifest["artifacts"].items()}
    for key in ("cache", "cohort"):
        manifest[key] = portable_name(manifest[key])
    p = paths(cfg)
    for key in ("cache", "cohort"):
        expected = (root / manifest[key]).resolve()
        expected.relative_to(root)
        if p[key].resolve() != expected or manifest[key] not in manifest["artifacts"]:
            raise ValueError(f"分类数据包的 {key} 路径或清单被改变。")
    for name, expected_hash in manifest["artifacts"].items():
        file = (root / name).resolve()
        file.relative_to(root)
        if not file.is_file() or digest(file) != expected_hash:
            raise ValueError(f"分类数据包文件缺失或校验失败：{name}")
    # Notebook hashes are provenance, not data identity. Model contracts are checked separately.


def require_native_images(cfg: dict) -> None:
    if cfg.get("data_access") == "classification_cache_only":
        raise ValueError("本操作需要原分辨率影像，请在数据电脑执行；导入结果和生成报告无需原影像。")


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("实验名称只使用字母、数字、下划线或短横线。")
    return value


def experiment_name(cfg, stage, default):
    return safe_name(cfg.get(stage, {}).get("experiment", default))


def portable_name(value):
    from pathlib import PurePosixPath
    name = str(value).replace("\\", "/")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ("..", ".") or ":" in p for p in path.parts):
        raise ValueError(f"数据包必须使用包内相对路径：{value}")
    return path.as_posix()


def contained_file(root, name):
    root = Path(root).resolve()
    target = (root / portable_name(name)).resolve()
    target.relative_to(root)
    return target


def cohort_identity(cfg):
    """Patient identity independent of local directories, CSV formatting and row order."""
    import pandas as pd
    table = read_cohort(cfg)
    columns = ["mr", "label", "split", "development_fold", "max_layer", "moca"] + CLINICAL
    if not set(columns) <= set(table):
        raise ValueError("病例清单缺少患者、结局、分组、取层或临床列。")
    rows = []
    for _, row in table.sort_values("mr").iterrows():
        values = {}
        for key in columns:
            value = row[key]
            values[key] = (None if pd.isna(value) else str(value) if key in ("mr", "split") else format(float(value), ".12g"))
        rows.append(values)
    payload = {"schema": 1, "patients": rows}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def definition_contract(cfg, prefix, names):
    import ast
    files = list(cfg["_root"].glob(prefix + "_*.ipynb"))
    if len(files) != 1:
        raise ValueError(f"找不到唯一的 {prefix} Notebook。")
    notebook = json.loads(files[0].read_text(encoding="utf-8"))
    definitions = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                if node.name in definitions:
                    raise ValueError(f"重复定义：{node.name}")
                definitions[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if set(definitions) != set(names):
        raise ValueError("模型或处理流程定义不齐全。")
    return definitions


def verify_transfer(cfg, directory, kind=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "transfer_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("kind") not in ("radiomics_features", "radiomics_complete", "segmentation_results", "segmentation_weights"):
        raise ValueError("不支持的数据包类型或版本。")
    if kind and manifest["kind"] != kind:
        raise ValueError("数据包类型不匹配。")
    if manifest["cohort_identity"] != cohort_identity(cfg):
        raise ValueError("两台机器的患者、标签、临床数据、取层或开发折不一致。")
    if manifest["native_source_identity"] != native_source_identity(cfg):
        raise ValueError("数据包与分类缓存记录的原始影像或人工标注不是同一版本。")
    if manifest["annotation_scope"] != "selected_slices" or cfg["annotation_scope"] != "selected_slices":
        raise ValueError("当前流程只接受四层标注协议。")
    for name, checksum in manifest["artifacts"].items():
        file = contained_file(directory, name)
        if not file.is_file() or digest(file) != checksum:
            raise ValueError(f"数据包文件缺失或传输后发生变化：{name}")
    return manifest


def export_transfer(cfg, source, filenames, destination, kind, protocol):
    import shutil
    source, destination = Path(source).resolve(), Path(destination).resolve()
    fresh(destination)
    mapping = {}
    for name in filenames:
        name = portable_name(name)
        file = contained_file(source, name)
        if not file.is_file():
            raise FileNotFoundError(file)
        mapping[name] = digest(file)
    destination.mkdir(parents=True)
    for name in mapping:
        target = contained_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained_file(source, name), target)
    dump_json(destination / "transfer_manifest.json", {
        "schema": 1, "kind": kind, "cohort_identity": cohort_identity(cfg), "native_source_identity": native_source_identity(cfg),
        "annotation_scope": cfg["annotation_scope"], "protocol": protocol,
        "artifacts": mapping, "source_run_id": cfg["run_id"],
        "notebook_definitions_sha256": source_snapshot(cfg)})
    verify_transfer(cfg, destination, kind)
    return destination


def import_transfer(cfg, source, stage):
    import shutil
    source = Path(source).resolve()
    manifest = verify_transfer(cfg, source)
    allowed = {"radiomics": ("radiomics_features", "radiomics_complete"),
               "segmentation": ("segmentation_results",), "segmentation_weights": ("segmentation_weights",)}
    if stage not in allowed or manifest["kind"] not in allowed[stage]:
        raise ValueError("数据包与接收阶段不匹配。")
    expected_protocol = "selected_four_slices_2d_pixels_v1"
    if manifest["protocol"] != expected_protocol:
        raise ValueError("数据包的二维取层/像素空间处理协议不匹配。")
    target = paths(cfg)[stage] if stage != "segmentation_weights" else paths(cfg)["segmentation"].with_name(paths(cfg)["segmentation"].name + "_imported_model")
    fresh(target)
    target.mkdir(parents=True)
    for name in manifest["artifacts"]:
        dest = contained_file(target, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(contained_file(source, name), dest)
    shutil.copy2(source / "transfer_manifest.json", target / "transfer_manifest.json")
    verify_transfer(cfg, target)
    return target


def native_source_identity(cfg):
    audit_file = paths(cfg)["data"] / "volume_audit_details.json"
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    patients = read_cohort(cfg).mr.tolist()
    files = IMAGE_FILES + ["Untitled.nii.gz"]
    selected = {mr: {name: audit["patients"][mr]["files"][name]["sha256"] for name in files} for mr in patients}
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
