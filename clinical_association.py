"""Clinical7 inputs and development-cohort association analysis for PSCI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


CLINICAL5 = ["age", "education", "nihss", "hcy", "ua"]
CLINICAL7 = CLINICAL5 + ["fazekas", "diabetes"]
EXTRA_COLUMNS = {"fazekas": "Fazekas", "diabetes": "糖尿病"}
# Effects from the adjusted logistic model are reported per increment below.
INCREMENTS = dict(zip(CLINICAL7, [10, 1, 1, 5, 50, 1, 1]))
EFFECT_UNITS = {
    "age": "10 years", "education": "1 year", "nihss": "1 point",
    "hcy": "5 µmol/L", "ua": "50 µmol/L", "fazekas": "1 grade",
    "diabetes": "yes vs no",
}


def normalize_mr(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.upper()
    return "MR" + text.str.extract(r"^(?:MR\s*)?(\d+)(?:\.0+)?$", expand=False)


def add_clinical7(cases: pd.DataFrame, excel_path: Path, config: dict) -> pd.DataFrame:
    """Add or validate the two extra predictors, preserving cohort row order."""
    cases = cases.copy()
    mrs = normalize_mr(cases["mr"])
    if mrs.isna().any() or not mrs.is_unique:
        raise ValueError("患者清单的 MR 缺失、无法识别或重复。")

    missing = [key for key in EXTRA_COLUMNS if key not in cases]
    if missing:
        if not excel_path.is_file():
            raise FileNotFoundError(
                f"患者清单缺少 {', '.join(missing)}，且找不到 {excel_path}。"
                "请提供配置中的 Excel，或使用脚本导出已补齐两列的患者清单。"
            )
        columns = {key: config["columns"].get(key, EXTRA_COLUMNS[key]) for key in missing}
        mr_column = config["columns"]["mr"]
        wanted = {mr_column, *columns.values()}
        source = pd.read_excel(excel_path, sheet_name=config["sheet_name"], dtype=object,
                               usecols=lambda name: str(name).strip() in wanted)
        source.columns = source.columns.astype(str).str.strip()
        absent = wanted.difference(source.columns)
        if absent or not source.columns.is_unique:
            raise ValueError(f"Excel 列名重复或缺少所需列：{', '.join(sorted(absent))}")
        source_mrs = normalize_mr(source[mr_column])
        selected = source_mrs.isin(mrs)
        matched = source.loc[selected].set_index(source_mrs.loc[selected])
        if not matched.index.is_unique:
            raise ValueError("Excel 中存在与患者清单匹配的重复 MR。")
        unmatched = ~mrs.isin(matched.index)
        if unmatched.any():
            raise ValueError(f"患者清单有 {int(unmatched.sum())} 个 MR 在 Excel 中没有匹配。")
        for key, column in columns.items():
            cases[key] = mrs.map(matched[column])

    fazekas = cases["fazekas"].astype("string").str.strip().replace("", pd.NA)
    fazekas_num = pd.to_numeric(fazekas, errors="coerce")
    if (fazekas.notna() & ~fazekas_num.isin([0, 1, 2, 3])).any():
        raise ValueError("Fazekas 应为 0、1、2、3 或缺失，请核对评分口径。")
    diabetes = cases["diabetes"].astype("string").str.strip().str.lower().replace("", pd.NA)
    diabetes_num = diabetes.map({"有": 1, "是": 1, "1": 1, "1.0": 1,
                                 "无": 0, "否": 0, "0": 0, "0.0": 0})
    if (diabetes.notna() & diabetes_num.isna()).any():
        raise ValueError("糖尿病应为 有/无、是/否、1/0 或缺失，请核对编码。")
    cases["fazekas"], cases["diabetes"] = fazekas_num.astype(float), diabetes_num.astype(float)
    return cases


def univariate_associations(dev: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu, spearmanr
    from statsmodels.stats.multitest import multipletests

    rows = []
    for feature in CLINICAL7:
        observed = dev[[feature, "moca", "label"]].dropna()
        x, moca = observed[feature], observed["moca"]
        effect, p_value = np.nan, np.nan
        method = "Spearman rho"
        if feature == "diabetes":
            method = "Mann–Whitney U, rank-biserial r (yes vs no)"
            yes, no = moca.loc[x.eq(1)], moca.loc[x.eq(0)]
            if len(yes) and len(no):
                result = mannwhitneyu(yes, no, alternative="two-sided", method="asymptotic")
                effect, p_value = 2 * result.statistic / (len(yes) * len(no)) - 1, result.pvalue
        elif len(observed) >= 3 and x.nunique() > 1 and moca.nunique() > 1:
            effect, p_value = spearmanr(x, moca)
        n_positive = int(observed["label"].sum())
        rows.append({"feature": feature, "n": len(observed), "missing_n": len(dev) - len(observed),
                     "positive_n": n_positive, "negative_n": len(observed) - n_positive,
                     "method": method, "effect": effect, "p_value": p_value})
    table = pd.DataFrame(rows)
    tested = table["p_value"].notna()
    table["p_fdr_bh"] = np.nan
    if tested.any():
        table.loc[tested, "p_fdr_bh"] = multipletests(table.loc[tested, "p_value"], method="fdr_bh")[1]
    return table


def adjusted_associations(dev: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning

    complete = dev[CLINICAL7 + ["label"]].dropna()
    y = complete["label"]
    x = sm.add_constant(complete[CLINICAL7].div(pd.Series(INCREMENTS)), has_constant="add")
    table = pd.DataFrame({"feature": CLINICAL7, "unit": [EFFECT_UNITS[key] for key in CLINICAL7],
                          "n": len(complete), "positive_n": int(y.sum()),
                          "negative_n": int(y.eq(0).sum()), "excluded_n": len(dev) - len(complete),
                          "adjusted_or": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                          "p_value": np.nan, "status": ""}).set_index("feature")
    if len(complete) <= x.shape[1] or y.nunique() != 2:
        table["status"] = "insufficient complete cases or outcome variation"
    elif np.linalg.matrix_rank(x.to_numpy()) != x.shape[1]:
        table["status"] = "rank-deficient design"
    else:
        try:
            with warnings.catch_warnings(), np.errstate(over="raise"):
                warnings.simplefilter("error", PerfectSeparationWarning)
                fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=100)
                estimates = np.exp(fit.params.loc[CLINICAL7])
                intervals = np.exp(fit.conf_int().loc[CLINICAL7]).to_numpy()
            if (fit.converged and np.isfinite(estimates).all() and np.isfinite(intervals).all()
                    and (intervals > 0).all() and np.isfinite(fit.pvalues.loc[CLINICAL7]).all()):
                table["adjusted_or"] = estimates
                table[["ci_lower", "ci_upper"]] = intervals
                table["p_value"] = fit.pvalues.loc[CLINICAL7]
                table["status"] = "ok"
            else:
                table["status"] = "non-convergence or unbounded estimates"
        except (PerfectSeparationError, PerfectSeparationWarning, FloatingPointError, np.linalg.LinAlgError):
            table["status"] = "separation or unstable estimates"
    return table.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="在固定开发队列分析 Clinical7 与 MoCA 的关联。")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, help="默认写入配置结果目录下的 clinical_association。")
    parser.add_argument("--export-cohort", type=Path, help="另存一份带新增两列的清单，供只有缓存的环境使用。")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.config.resolve().parent
    cohort_path = root / Path(config["cohort_path"]).expanduser()
    excel_path = root / Path(config["excel_path"]).expanduser()
    output = args.output_dir or root / Path(config["results_dir"]).expanduser() / "clinical_association"

    cases = pd.read_csv(cohort_path, dtype={"mr": str})
    required = {"mr", "split", "moca", "label", *CLINICAL5}
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"患者清单缺列：{', '.join(sorted(missing))}")
    if not set(cases["split"]) <= {"development", "validation"}:
        raise ValueError("患者清单分组有误。")
    dev = cases.loc[cases["split"].eq("development")].copy()
    dev["moca"] = pd.to_numeric(dev["moca"], errors="coerce")
    if (dev.empty or not dev["moca"].between(0, 30).all() or set(dev["label"]) != {0, 1}
            or not dev["moca"].le(22).eq(dev["label"]).all()):
        raise ValueError("开发队列为空、缺少某类结局、MoCA 无效或与 MoCA ≤22 的标签不一致。")

    if args.export_cohort:
        if args.export_cohort.exists() or args.export_cohort.suffix.lower() != ".csv":
            parser.error("--export-cohort 请提供一个尚不存在的 CSV 路径。")
        enriched = add_clinical7(cases, excel_path, config)
        for key in EXTRA_COLUMNS:
            dev[key] = enriched.loc[dev.index, key]
    else:
        dev = add_clinical7(dev, excel_path, config)
    for key in CLINICAL5:
        dev[key] = pd.to_numeric(dev[key], errors="coerce").replace([np.inf, -np.inf], np.nan)

    univariate = univariate_associations(dev)
    adjusted = adjusted_associations(dev)
    output.mkdir(parents=True, exist_ok=True)
    univariate.to_csv(output / "univariate_moca.csv", index=False, encoding="utf-8-sig")
    adjusted.to_csv(output / "adjusted_odds_ratios.csv", index=False, encoding="utf-8-sig")
    print(f"开发队列 {len(dev)} 例，MoCA ≤22 为 {int(dev['label'].sum())} 例。")
    print(f"调整模型完整病例 {adjusted['n'].iloc[0]} 例，状态 {adjusted['status'].iloc[0]}。")
    print(f"关联分析结果：{output}")
    if args.export_cohort:
        args.export_cohort.parent.mkdir(parents=True, exist_ok=True)
        with args.export_cohort.open("x", encoding="utf-8", newline="") as file:
            enriched.to_csv(file, index=False)
        print(f"补齐临床字段的患者清单：{args.export_cohort}")


if __name__ == "__main__":
    main()
