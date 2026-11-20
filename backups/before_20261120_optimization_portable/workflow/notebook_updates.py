"""Register the reviewed 03-only update without changing data or fitted models."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .project import digest, source_snapshot


def register_notebook03_update(root):
    root = Path(root).resolve()
    notebook_name = "03_任务3_原单序列消融整合.ipynb"
    notebook = json.loads((root / notebook_name).read_text(encoding="utf-8"))
    release = notebook["metadata"]["psci"]["structure_update"]
    current = source_snapshot({"_root": root})
    if current.get(notebook_name) != release["new_definitions_sha256"]:
        raise ValueError("03的代码定义与本次更新版本不同，请先保存并使用完整的更新文件。")
    previous = {**current, notebook_name: release["previous_definitions_sha256"]}
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["run_id"]):
        raise ValueError("run_id格式无效。")
    output_root = Path(config["outputs_root"]).expanduser()
    output_root = output_root if output_root.is_absolute() else root / output_root
    run = output_root / config["run_id"]
    if config.get("existing_cache"):
        cache = Path(config["existing_cache"]).expanduser()
        cache = cache if cache.is_absolute() else root / cache
    else:
        cache = run / "data" / "resnet50_manual_cache.npz"
    candidates = []
    bundle_path = root / "classification_bundle.json"
    if config.get("data_access") == "classification_cache_only":
        if not bundle_path.is_file():
            raise FileNotFoundError("缺少classification_bundle.json，请保留原上传包的配置和数据目录。")
        candidates.append((bundle_path, root, "bundle"))
    lock = run / "temporal" / "frozen_artifacts.json"
    if lock.is_file():
        candidates.append((lock, run, "model_lock"))
    if not candidates:
        print("当前目录没有需要更新的分类包或模型锁定记录。")
        return
    plans = []
    for path, base, kind in candidates:
        raw = path.read_bytes()
        metadata = json.loads(raw)
        stored = metadata.get("notebook_definitions_sha256")
        if stored != current and stored != previous:
            changed = sorted(name for name in set(current) | set(stored or {})
                             if (stored or {}).get(name) != current.get(name))
            raise ValueError(f"{path.name}不是已知的03单独更新，差异涉及{changed}。请核对其余Notebook，不能自动登记。")
        if kind == "bundle":
            if metadata.get("kind") != "classification_cache_only" or metadata["run_id"] != config["run_id"]:
                raise ValueError("上传包与配置的run_id或用途不同。")
            expected_paths = {"cache": cache, "cohort": run / "data" / "cohort.csv"}
            for key, expected in expected_paths.items():
                if (root / metadata[key]).resolve() != expected.resolve() or metadata[key] not in metadata["artifacts"]:
                    raise ValueError(f"上传包的{key}路径不同；请先核对配置及JSON中的斜杠方向。")
        if stored == current:
            continue
        for relative, checksum in metadata["artifacts"].items():
            artifact = (base / relative).resolve()
            artifact.relative_to(base.resolve())
            if not artifact.is_file() or digest(artifact) != checksum:
                raise ValueError(f"{path.name}记录的数据或模型发生变化：{relative}。停止登记，保留已有记录。")
        if kind == "model_lock" and digest(cache) != metadata["cache_sha256"]:
            raise ValueError("缓存与原模型锁定版本不同。")
        updated = {**metadata, "notebook_definitions_sha256": current}
        plans.append((path, raw, updated))
    if not plans:
        print("03更新已登记，可继续读取配置。")
        return
    backup_dir = run / "provenance" / release["update_id"]
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Validate all candidates before touching any manifest, and keep original bytes.
    for path, raw, updated in plans:
        if path.read_bytes() != raw:
            raise ValueError(f"登记期间{path.name}发生变化，请重试。")
        backup = backup_dir / (path.name + ".before-update.json")
        if backup.exists():
            if backup.read_bytes() != raw:
                raise ValueError(f"已有不同的更新备份：{backup}；请保留并核对。")
        else:
            with backup.open("xb") as stream:
                stream.write(raw)
    for path, raw, updated in plans:
        if path.read_bytes() != raw:
            raise ValueError(f"登记期间{path.name}发生变化，请重试。")
        temporary = path.with_name(path.name + ".03-update.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    receipt = {"update_id": release["update_id"], "notebook": notebook_name,
        "previous_definitions_sha256": release["previous_definitions_sha256"],
        "new_definitions_sha256": release["new_definitions_sha256"],
        "updated_manifests": [str(path) for path, _, _ in plans],
        "unchanged_notebooks": {k: v for k, v in current.items() if k != notebook_name},
        "note": "只登记新增结构消融的03版本；缓存、分组、02模型文件及其校验值保持不变。"}
    receipt_path = backup_dir / "update_receipt.json"
    if not receipt_path.exists():
        with receipt_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print("已登记03更新；原清单备份在：", backup_dir)
