"""读取原始NIfTI，生成待复核清单和叠图；不修改影像、标注、队列或训练缓存。

在完整数据电脑运行：python 07_数据筛查.py
全部患者的影像质量检查：python 07_数据筛查.py --scope all
只生成清单：python 07_数据筛查.py --no-figures
路径沿用同级config.json。默认只检查开发集，模型疑难病例另表输出。
"""

import argparse
import html
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


FILES = {"t1": "t1_flair.nii.gz", "t2": "t2.nii.gz",
         "dwi": "dwi_b1000.nii.gz", "manual_mask": "Untitled.nii.gz"}
AFFINE_TOL = np.array([1e-4, 1e-4, 1e-4, 1e-3])  # 与00一致，按矩阵列比较。
TINY_PIXELS = 3  # 连通域启发式提示，不是病灶大小诊断标准。
COLORS = {1: "red", 2: "lime", 3: "cyan"}  # 未假定1/2/3的临床含义。


def resolve_path(value, root):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def normalize_mr(value):
    match = re.fullmatch(r"(?:MR\s*)?(\d+)(?:\.0+)?", str(value).strip().upper())
    return "MR" + match.group(1) if match else None


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def flag(row, level, message):
    row["_issues"].append((level, message))


def scan_folders(root):
    """只看直接患者目录，不递归匹配备份文件；缺少T1的病例也保留。"""
    folders = {}
    if root.is_dir():
        for patient in sorted(root.iterdir()):
            match = re.search(r"MR\s*(\d+)$", patient.name.strip(), re.IGNORECASE)
            if patient.is_dir() and match:
                directory = (patient / "配准图像").resolve()
                candidates = folders.setdefault("MR" + match.group(1), [])
                if directory not in candidates:
                    candidates.append(directory)
    return folders


def excel_reference(cfg, root, messages):
    path = resolve_path(cfg["excel_path"], root)
    if not path.is_file():
        messages.append("未找到Excel：本次仅检查cohort内部一致性，未核对最新Excel。")
        return None
    try:
        data = pd.read_excel(path, sheet_name=cfg["sheet_name"])
        data.columns = data.columns.astype(str).str.strip()
        columns = cfg["columns"]
        keys = ("mr", "moca", "max_layer")
        if not data.columns.is_unique or any(columns[k] not in data for k in keys):
            raise ValueError("缺少MR、MoCA或取层列，或者列名重复")
        data = pd.DataFrame({key: data[columns[key]] for key in keys})
        data["mr"] = data.mr.map(normalize_mr)
        if data.mr.isna().any():
            messages.append(f"Excel有{int(data.mr.isna().sum())}行MR无法识别，需核对原表。")
        return {mr: group for mr, group in data.dropna(subset=["mr"]).groupby("mr")}
    except Exception as exc:
        messages.append(f"未能读取Excel，未完成Excel核对：{type(exc).__name__}: {exc}")
        return None


def check_clinical(patient, row, references):
    score, outcome = number(patient.get("moca")), number(patient.get("label"))
    if not np.isfinite(score) or not 0 <= score <= 30:
        flag(row, 1, "cohort中MoCA缺失或不在0～30范围，请查原始记录")
    elif outcome not in (0, 1) or outcome != int(score <= 22):
        flag(row, 1, "cohort中的MoCA与二分类结局不一致，请查原始记录")
    if patient.get("split") not in ("development", "validation"):
        flag(row, 1, "患者分组字段无效")
    if patient.get("split") == "development" and number(patient.get("development_fold")) not in (1, 2, 3, 4, 5):
        flag(row, 1, "开发患者折号不是1～5")
    if references is None:
        return
    source = references.get(row["mr"])
    if source is None:
        flag(row, 1, "当前Excel中未找到此MR，请核对患者匹配")
    elif len(source) != 1:
        flag(row, 1, "当前Excel有重复MR，未自动选择其中一行")
    else:
        for key in ("moca", "max_layer"):
            a, b = number(patient.get(key)), number(source.iloc[0][key])
            if not np.isfinite(a) or not np.isfinite(b) or a != b:
                flag(row, 1, f"cohort与当前Excel的{key}不一致，请核对来源")


def locate_files(patient, row, folders, root):
    candidates = folders.get(row["mr"], [])
    if len(candidates) > 1:
        flag(row, 1, "存在多个同MR患者目录，未自动选取：" + " | ".join(map(str, candidates)))
        return {}
    if candidates:
        return {key: candidates[0] / name for key, name in FILES.items()}
    # 换电脑后优先使用config中的患者目录；找不到时才使用清单记录的原路径。
    return {key: resolve_path(patient[key], root) for key in FILES
            if key in patient and pd.notna(patient[key]) and str(patient[key]).strip()}


def valid_geometry(image):
    return (len(image.shape) == 3 and min(image.shape) > 0
            and np.isfinite(image.affine).all()
            and abs(np.linalg.det(image.affine[:3, :3])) > 1e-12
            and np.isfinite(image.header.get_zooms()[:3]).all()
            and min(image.header.get_zooms()[:3]) > 0)


def same_grid(a, b):
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=AFFINE_TOL, rtol=0)


def normalized(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, np.float32), float("nan"), float("nan")
    lo, hi = np.percentile(finite, [1, 99.5])
    clean = np.nan_to_num(values, nan=float(lo), posinf=float(hi), neginf=float(lo))
    return np.clip((clean - lo) / max(float(hi - lo), 1e-6), 0, 1), float(lo), float(hi)


def image_statistics(key, values, row):
    bad = int((~np.isfinite(values)).sum())
    row[f"{key}_invalid_pixels"] = bad
    if bad:
        flag(row, 1, f"{key}选定四层包含NaN或Inf")
    display, lo, hi = normalized(values)
    row[f"{key}_p1"], row[f"{key}_p995"] = lo, hi
    constant = [i + row["first_slice"] for i in range(4)
                if not np.isfinite(values[:, :, i]).any()
                or np.ptp(values[:, :, i][np.isfinite(values[:, :, i])]) <= 1e-6]
    if constant:
        flag(row, 1, f"{key}以下层为常数图或无有效像素：{constant}")
    if np.isfinite(hi - lo) and hi - lo <= 1e-6:
        flag(row, 2, f"{key}四层归一化分位差接近0，请看图核对")
    if min(values.shape[:2]) > 1 and not bad and hi > lo:
        row[f"{key}_gradient"] = float((np.abs(np.diff(display, axis=0)).mean()
                                             + np.abs(np.diff(display, axis=1)).mean()) / 2)
    return display


def mask_statistics(labels, start, row):
    finite = np.isfinite(labels)
    good = np.isin(labels, [0, 1, 2, 3]) & finite
    row["mask_invalid_pixels"] = int((~good).sum())
    if not good.all():
        flag(row, 1, "标注包含0/1/2/3以外的值或NaN/Inf，未自动取整或重编码")
    annotated = np.isin(labels, [1, 2, 3])
    row["annotated_slices"] = ",".join(map(str, np.flatnonzero(annotated.any(axis=(0, 1))) + 1))
    if start is None or start + 4 > labels.shape[2]:
        return
    selected = labels[:, :, start:start + 4]
    outside = int(annotated[:, :, :start].sum() + annotated[:, :, start + 4:].sum())
    row["outside_four_pixels"] = outside
    if outside:
        flag(row, 3, "选定四层之外也有标注，仅提示核对取层，不认定为错标")
    if not np.isin(selected, [1, 2, 3]).any():
        flag(row, 2, "选定四层均无1/2/3标注，请确认确无目标区域还是漏标")
    for label in (1, 2, 3):
        region = selected == label
        row[f"label_{label}_pixels"] = int(region.sum())
        row[f"label_{label}_slice_pixels"] = ",".join(map(str, region.sum(axis=(0, 1))))
        tiny, components, tiny_layers = 0, 0, []
        for z in range(4):
            connected, count = ndimage.label(region[:, :, z], structure=np.ones((3, 3)))
            areas = np.bincount(connected.ravel())[1:]
            components += count
            count_tiny = int(((areas > 0) & (areas <= TINY_PIXELS)).sum())
            tiny += count_tiny
            if count_tiny:
                tiny_layers.append(start + z + 1)
        row[f"label_{label}_components_2d"] = components
        row[f"label_{label}_tiny_components"] = tiny
        if tiny:
            flag(row, 2, f"标签{label}在第{tiny_layers}层有{tiny}个≤{TINY_PIXELS}像素的二维连通域，需区分小目标与零散误点")
        if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
            flag(row, 2, f"标签{label}接触图像外边界，请核对截断或误画")


def draw_contours(ax, labels):
    for label, color in COLORS.items():
        plane = labels.T == label
        if plane.any():
            # 坐标仍对应imshow的像素中心；补零只是让触边区域也能画出闭合轮廓。
            ax.contour(np.arange(-1, plane.shape[1] + 1), np.arange(-1, plane.shape[0] + 1),
                       np.pad(plane.astype(float), 1), levels=[.5], colors=[color], linewidths=.6)


def make_figures(row, volumes, selected, labels, start, output):
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    filename = row["case_id"]
    fig, axes = plt.subplots(6, 4, figsize=(12, 17))
    try:
        for r, key in enumerate(("t1", "t2", "dwi")):
            values = selected.get(key)
            overlay_ok = (values is not None and labels is not None and "manual_mask" in volumes
                          and same_grid(volumes[key], volumes["manual_mask"]))
            for s in range(4):
                for axis in (axes[r, s], axes[r + 3, s]):
                    if values is not None:
                        axis.imshow(values[:, :, s].T, cmap="gray", vmin=0, vmax=1, origin="lower")
                    else:
                        axis.text(.5, .5, "Unavailable", ha="center", va="center", transform=axis.transAxes)
                    axis.set_axis_off()
                level = start + s + 1 if start is not None else "?"
                axes[r, s].set_title(f"{key.upper()} | slice {level}", fontsize=9)
                axes[r + 3, s].set_title("Mask contours" if overlay_ok else "No aligned mask", fontsize=9)
                if overlay_ok:
                    draw_contours(axes[r + 3, s], labels[:, :, start + s])
        fig.suptitle(f"{row['mr']} | labels 1=red, 2=green, 3=cyan\n"
                     "Native voxel axes; same display window per sequence; no resampling", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .96))
        file = figures / f"{filename}_detail.png"
        fig.savefig(file, dpi=130)
        row["detail_image"] = file.relative_to(output).as_posix()
    finally:
        plt.close(fig)

    if "t1" not in volumes or start is None or not valid_geometry(volumes["t1"]):
        return
    image = volumes["t1"]
    depth = image.shape[2]
    # 每次只处理一例；总览一次读取T1，避免逐层反复解压同一个nii.gz。
    overview_volume = np.asarray(image.dataobj, dtype=np.float32)
    # 一般病例约20层，全部展示；更多层时采样，同时保留选中层及邻层。
    layers = set(np.linspace(0, depth - 1, min(depth, 32), dtype=int).tolist())
    layers.update(range(max(0, start - 1), min(depth, start + 5)))
    layers = sorted(layers)
    fig, axes = plt.subplots(math.ceil(len(layers) / 5), 5, figsize=(15, 3 * math.ceil(len(layers) / 5)), squeeze=False)
    try:
        lo, hi = row.get("t1_p1", np.nan), row.get("t1_p995", np.nan)
        overlay_ok = labels is not None and same_grid(image, volumes["manual_mask"])
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, z in zip(axes.flat, layers):
            plane = overview_volume[:, :, z]
            if np.isfinite(lo) and np.isfinite(hi):
                view = np.clip((np.nan_to_num(plane) - lo) / max(hi - lo, 1e-6), 0, 1)
            else:
                view, _, _ = normalized(plane)
            ax.imshow(view.T, cmap="gray", vmin=0, vmax=1, origin="lower")
            if overlay_ok:
                draw_contours(ax, labels[:, :, z])
            is_selected = start <= z < start + 4
            ax.set_title(f"slice {z + 1}" + (" | SELECTED" if is_selected else ""),
                         color="darkorange" if is_selected else "black", fontsize=10)
        fig.suptitle(f"{row['mr']} | T1 slice overview ({len(layers)}/{depth} shown)\n"
                     "Outside selected four: no contour does NOT mean no lesion", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
        file = figures / f"{filename}_overview.png"
        fig.savefig(file, dpi=110)
        row["overview_image"] = file.relative_to(output).as_posix()
    finally:
        plt.close(fig)


def inspect_patient(patient, row, folders, root, references, output, figures):
    check_clinical(patient, row, references)
    files = locate_files(patient, row, folders, root)
    volumes, selected, labels = {}, {}, None
    value = number(patient.get("max_layer"))
    start = int(value) - 1 if np.isfinite(value) and value.is_integer() and value >= 1 else None
    row["first_slice"] = start + 1 if start is not None else ""
    if start is None:
        flag(row, 1, "取层起点无效，应为从1开始的整数")
    for key in FILES:
        file = files.get(key)
        row[f"{key}_file"] = str(file) if file is not None else ""
        if file is None or not file.is_file():
            flag(row, 1, f"缺少{FILES[key]}，请核对目录或文件名")
            continue
        try:
            image = nib.load(str(file))
            row[f"{key}_shape"] = "x".join(map(str, image.shape))
            row[f"{key}_spacing"] = ",".join(f"{v:g}" for v in image.header.get_zooms()[:3])
            row[f"{key}_unit"] = image.header.get_xyzt_units()[0]
            if not valid_geometry(image):
                flag(row, 1, f"{key}不是有效三维影像或空间矩阵/间距异常")
                continue
            volumes[key] = image
            if row[f"{key}_unit"] == "unknown":
                flag(row, 3, f"{key}空间单位未声明，仅记录，不据此认定配准错误")
            if start is not None and start + 4 > image.shape[2]:
                flag(row, 1, f"{key}取层越界：起点{start + 1}，共{image.shape[2]}层")
            if key == "manual_mask":
                labels = np.asarray(image.dataobj)
                mask_statistics(labels, start, row)
            elif start is not None and start + 4 <= image.shape[2]:
                values = np.asarray(image.dataobj[:, :, start:start + 4], dtype=np.float32)
                selected[key] = image_statistics(key, values, row)
        except Exception as exc:
            flag(row, 1, f"{key}读取/检查失败：{type(exc).__name__}: {exc}")
    if "t1" in volumes:
        ref = volumes["t1"]
        row["matrix_hw"] = f"{ref.shape[0]}x{ref.shape[1]}"
        for key, image in volumes.items():
            row[f"{key}_affine_max_diff"] = float(np.max(np.abs(image.affine - ref.affine)))
            if not same_grid(ref, image):
                flag(row, 1, f"{key}与T1的尺寸或空间矩阵不一致，不能直接逐像素对应")
    if figures:
        try:
            make_figures(row, volumes, selected, labels, start, output)
        except Exception as exc:
            flag(row, 2, f"部分预览图未生成：{type(exc).__name__}: {exc}")


def relative_flags(rows):
    """梯度异常只作粗筛，参考分布只用开发患者，不用模型结果。"""
    development = [r for r in rows if r["split"] == "development"]
    shapes = pd.Series([r["matrix_hw"] for r in development if r.get("matrix_hw")], dtype=str)
    dominant = shapes.mode().iloc[0] if len(shapes) else None
    for row in rows:
        if dominant and row.get("matrix_hw") and row["matrix_hw"] != dominant:
            flag(row, 2, f"矩阵尺寸{row['matrix_hw']}不同于开发集常见尺寸{dominant}，请核对")
    for key in ("t1", "t2", "dwi"):
        values = np.array([r.get(f"{key}_gradient", np.nan) for r in development], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) < 20:
            continue
        logs = np.log(values)
        center = float(np.median(logs))
        scale = float(1.4826 * np.median(np.abs(logs - center)))
        if scale < 1e-6:
            continue
        for row in rows:
            value = row.get(f"{key}_gradient", np.nan)
            if np.isfinite(value) and value > 0:
                z = float((np.log(value) - center) / scale)
                row[f"{key}_gradient_robust_z"] = z
                if z < -3.5:
                    flag(row, 2, f"{key}梯度指标明显偏低，请看图排查模糊或覆盖差异；指标不能诊断运动伪影")


def model_review(cases, model_dir, output, messages):
    """仅使用开发集OOF；不加载权重，不读取测试预测，也不影响质量排序。"""
    file = model_dir / "development_oof.csv"
    if not file.is_file():
        messages.append("未找到开发集OOF，已跳过模型疑难病例表；影像筛查照常完成。目录中如有旧model_review.csv，不代表本次结果。")
        return False
    try:
        oof = pd.read_csv(file, dtype={"mr": str})
        dev = cases.loc[cases.split == "development", ["mr", "label", "development_fold"]].copy()
        if not dev.mr.is_unique or not oof.mr.is_unique or set(dev.mr) != set(oof.mr):
            raise ValueError("OOF与当前开发队列的MR不一致或重复")
        joined = dev.merge(oof[["mr", "label", "development_fold", "probability"]], on="mr", suffixes=("", "_oof"), validate="one_to_one")
        if not joined.label.eq(joined.label_oof).all() or not joined.development_fold.eq(joined.development_fold_oof).all():
            raise ValueError("OOF的标签或折号与当前患者清单不同")
        prob = pd.to_numeric(joined.probability, errors="coerce")
        if not np.isfinite(prob).all() or not prob.between(0, 1).all() or not joined.label.isin([0, 1]).all():
            raise ValueError("OOF概率或结局无效")
        joined["probability"] = prob
        joined["disagreement"] = np.abs(joined.label - prob)
        metrics = model_dir / "development_metrics.csv"
        if metrics.is_file():
            threshold = float(pd.read_csv(metrics).iloc[0]["threshold"])
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("开发阈值无效")
            joined["threshold"] = threshold
            joined["prediction"] = (prob >= threshold).astype(int)
            joined["prediction_error"] = joined.prediction != joined.label
        else:
            messages.append("未找到开发阈值，模型表仅按概率与结局的不一致程度排序，不判定分类对错。")
        joined.drop(columns=["label_oof", "development_fold_oof"]).sort_values("disagreement", ascending=False).to_csv(
            output / "model_review.csv", index=False, encoding="utf-8-sig")
        return True
    except Exception as exc:
        messages.append(f"本次未生成模型疑难病例表：{type(exc).__name__}: {exc}。如目录留有旧表，请勿当成本次结果。")
        return False


def save_reports(rows, output, messages, has_model):
    for row in rows:
        issues = row.pop("_issues")
        row["priority"] = min((level for level, _ in issues), default=4)
        row["review_reason"] = "；".join(dict.fromkeys(message for _, message in issues)) or "未触发自动规则，仍需人工看图"
    table = pd.DataFrame(rows).sort_values(["priority", "mr", "cohort_row"], kind="stable")
    front = ["case_id", "mr", "split", "priority", "review_reason", "first_slice", "detail_image", "overview_image"]
    table = table.reindex(columns=front + [c for c in table if c not in front])
    table.to_csv(output / "quality_screening.csv", index=False, encoding="utf-8-sig")
    notes_file = output / "review_notes.csv"
    if not notes_file.exists():
        notes = table[["case_id", "mr", "split"]].copy()
        for column in ("review_status", "confirmed_problem", "action", "reviewer", "notes"):
            notes[column] = ""
        notes.to_csv(notes_file, index=False, encoding="utf-8-sig")
    else:
        messages.append("review_notes.csv已存在，保留原有人工记录，未覆盖；如队列变动，按本次清单补充记录。")
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for row in table.to_dict("records"):
        links = []
        for key, title in (("detail_image", "三序列原图与标注"), ("overview_image", "T1取层总览")):
            name = row.get(key)
            if isinstance(name, str) and name:
                links.append(f'<a href="{quote(name)}" target="_blank">{title}</a>')
        cards.append(f'<tr><td>{esc(row["mr"])}</td><td>P{row["priority"]}</td>'
                     f'<td>{esc(row["review_reason"])}</td><td>{" / ".join(links) or "本次无预览图"}</td></tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>数据复核</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 20px;color:#222}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}a{color:#165eac}li{margin:6px 0}</style>
<h1>影像与标注复核</h1>
<p>先看图和源记录，再决定是否需要重标。P1：文件、网格或来源问题优先核对；P2：可疑特征需人工判断；P3：记录性提示；P4：未触发规则。它们不是数据好坏的结论。</p>
<p>所有当前范围内的患者均列出。小连通域可能是真实小目标，空标注可能正确，未声明空间单位也不等于配准错误。相同shape/affine只说明网格相容，配准、漏标、边界和标签语义仍需人工检查。1/2/3仅按标签编号显示。</p>
<p>图中不显示MoCA、分类结局或预测。上三行是原图，下三行是轮廓叠图；每个序列使用四层共同的1%～99.5%显示窗。图为原始体素轴展示，不能直接当作标准放射学左右方向。矩阵不一致时不叠加mask。四层以外未画轮廓不代表没有病灶。</p>
<p><a href="quality_screening.csv">自动筛查清单</a> · <a href="review_notes.csv">人工复核记录</a>。可在记录中填写“未见问题 / 需重新勾画 / 需重新配准 / 需核对取层 / 需核对源记录”等结论。原影像、标注、缓存及患者分组均未改写。</p>'''
    if messages:
        page += "<ul>" + "".join(f"<li>{esc(m)}</li>" for m in messages) + "</ul>"
    page += "<table><tr><th>MR</th><th>优先级</th><th>待核对内容</th><th>查看影像</th></tr>" + "".join(cards) + "</table>"
    if has_model:
        page += '<p>影像复核完成后，可另看<a href="model_review.csv">开发集模型疑难病例</a>。概率与结局不一致不能证明错标，不按它自动剔除患者或改变结局。该表未参与上面的质量排序。</p>'
    page += "</html>"
    (output / "index.html").write_text(page, encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("config.json"))
    parser.add_argument("--scope", choices=("development", "all"), default="development")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    config = args.config.expanduser().resolve()
    root = config.parent
    cfg = json.loads(config.read_text(encoding="utf-8"))
    cases = pd.read_csv(resolve_path(cfg["cohort_path"], root), dtype={"mr": str})
    required = {"mr", "split", "moca", "label", "max_layer", "development_fold"}
    if not required <= set(cases):
        raise ValueError(f"cohort.csv缺少列：{sorted(required - set(cases))}")
    original_mrs = cases.mr.copy()
    cases["mr"] = cases.mr.map(normalize_mr)
    duplicate = cases.mr.duplicated(keep=False) & cases.mr.notna()
    results = resolve_path(cfg["results_dir"], root)
    output = results / "data_review" / args.scope
    output.mkdir(parents=True, exist_ok=True)
    messages = []
    patient_root = resolve_path(cfg["patients_root"], root)
    folders = scan_folders(patient_root)
    if not patient_root.is_dir():
        messages.append("patients_root不存在，尝试使用cohort中的原路径；换电脑时请先检查config。")
    elif not folders:
        messages.append("未匹配到患者目录，尝试cohort中的原路径；patients_root应直接包含名称以MR编号结尾的患者文件夹。")
    references = excel_reference(cfg, root, messages)
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")  # 直接保存图片，不启动窗口或预览服务。
    include = cases.split.eq("development") | ~cases.split.isin(["development", "validation"])
    chosen = cases if args.scope == "all" else cases.loc[include]
    if chosen.empty:
        raise ValueError("当前范围内没有患者，请核对cohort的split字段。")
    rows = []
    for index, patient in chosen.iterrows():
        mr = patient.mr if pd.notna(patient.mr) else f"INVALID_ROW_{index + 2}"
        row = {"case_id": f"{mr}_row{index + 2}", "mr": mr, "split": patient.split,
               "cohort_row": index + 2, "detail_image": "", "overview_image": "", "_issues": []}
        if pd.isna(patient.mr):
            flag(row, 1, f"MR编号无法识别：{original_mrs.iloc[index]}")
        if duplicate.iloc[index]:
            flag(row, 1, "cohort存在重复MR，请核对是否同一患者或重复检查")
        try:
            inspect_patient(patient, row, folders, root, references, output, not args.no_figures)
        except Exception as exc:
            flag(row, 1, f"该病例检查未完成：{type(exc).__name__}: {exc}")
        rows.append(row)
        print(f"筛查 {len(rows)}/{len(chosen)}：{mr}，记录{len(row['_issues'])}项提示", flush=True)
    relative_flags(rows)
    has_model = model_review(cases, results / "resnet", output, messages)
    table = save_reports(rows, output, messages, has_model)
    print("\n完成。本次未修改原始数据、患者清单或训练缓存。")
    print("优先级统计：", table.priority.value_counts().sort_index().to_dict())
    for message in messages:
        print("提示：", message)
    print("打开复核页面：", output / "index.html")
    print("人工记录：", output / "review_notes.csv")


if __name__ == "__main__":
    main()
