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
        "radiomics": root / "radiomics", "explain": root / "explain", "segmentation": root / "segmentation",
        "reports": root / "reports",
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
           "protocol": "original_resnet_integration_no_architecture_optimization", **(extra or {})}
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
    if manifest["notebook_definitions_sha256"] != source_snapshot(cfg):
        raise ValueError("Notebook 定义与数据包导出时不同，请使用对应版本或重新导出。")


def require_native_images(cfg: dict) -> None:
    if cfg.get("data_access") == "classification_cache_only":
        raise ValueError("本数据包只有分类缓存，没有原分辨率影像；影像组学和分割需另外准备原生数据。")
