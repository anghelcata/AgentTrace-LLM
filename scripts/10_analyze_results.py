from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


ANALYSIS_VERSION = "agenttrace-analysis-1.0"
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 42

CONDITIONS = [
    "FULL",
    "NO_PROMPT_HISTORY",
    "NO_EXPLICIT_PROVENANCE",
    "NO_RAW_MESSAGES",
    "PARTIAL_LOG_LOSS",
    "TAMPERED_RECORD",
]

COMPONENTS = {
    "affected_context": "affected_context_score",
    "first_integrity_anomaly": "first_integrity_anomaly_score",
    "root_event": "root_event_score",
    "timeline": "timeline_reconstruction_score",
    "causal_propagation_path": "causal_propagation_path_score",
    "provenance": "provenance_reconstruction_score",
    "correction_containment": "correction_containment_score",
    "final_impact": "final_impact_score",
    "evidence_integrity": "evidence_integrity_detection_score",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("bootstrap_ci expects a one-dimensional array")
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    idx = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    boots = np.nanmean(values[idx], axis=1)
    return (
        float(np.nanpercentile(boots, 2.5)),
        float(np.nanpercentile(boots, 97.5)),
    )


def macro_condition_table(
    scored: pd.DataFrame,
    full_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    metrics = ["FRS_C", "EGR", "UIA"]

    inc_cond = (
        scored.groupby(["incident_id", "evidence_condition"], observed=True)[metrics]
        .mean()
        .reset_index()
    )

    # FRR-S uses all planned reconstructions, including the one output-level failure
    # already encoded as FRR_S=0 by the frozen batch-scoring contract.
    frr_inc_cond = (
        full_df.groupby(["incident_id", "evidence_condition"], observed=True)["FRR_S"]
        .mean()
        .reset_index()
    )

    incidents = sorted(full_df["incident_id"].unique())
    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        row: dict[str, Any] = {
            "evidence_condition": condition,
            "planned_n": int((full_df["evidence_condition"] == condition).sum()),
            "scored_n": int((scored["evidence_condition"] == condition).sum()),
        }

        c = (
            inc_cond[inc_cond["evidence_condition"] == condition]
            .set_index("incident_id")
            .reindex(incidents)
        )
        for metric in metrics:
            vals = c[metric].to_numpy(float)
            lo, hi = bootstrap_ci(vals, rng)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi

        f = (
            frr_inc_cond[frr_inc_cond["evidence_condition"] == condition]
            .set_index("incident_id")
            .reindex(incidents)
        )
        vals = f["FRR_S"].to_numpy(float)
        lo, hi = bootstrap_ci(vals, rng)
        row["FRR_S_mean"] = float(np.nanmean(vals))
        row["FRR_S_ci95_low"] = lo
        row["FRR_S_ci95_high"] = hi
        rows.append(row)

    return pd.DataFrame(rows)


def fit_frs_c_gee(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = scored.copy()
    work["evidence_condition"] = pd.Categorical(
        work["evidence_condition"],
        categories=CONDITIONS,
        ordered=True,
    )

    formula = (
        'FRS_C ~ C(evidence_condition, Treatment(reference="FULL")) + C(model)'
    )
    model = smf.gee(
        formula=formula,
        groups="incident_id",
        data=work,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    result = model.fit()

    terms = [t for t in result.params.index if "evidence_condition" in t]
    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []

    for term in terms:
        condition = term.split("[T.", 1)[1].rstrip("]")
        ci = result.conf_int().loc[term]
        p = float(result.pvalues[term])
        raw_p.append(p)
        rows.append(
            {
                "evidence_condition": condition,
                "reference_condition": "FULL",
                "adjusted_difference_modified_minus_FULL": float(result.params[term]),
                "robust_se": float(result.bse[term]),
                "ci95_low": float(ci.iloc[0]),
                "ci95_high": float(ci.iloc[1]),
                "p_raw": p,
            }
        )

    holm = multipletests(raw_p, method="holm")[1]
    for row, p_adj in zip(rows, holm):
        row["p_holm"] = float(p_adj)
        row["holm_significant_0_05"] = bool(p_adj < 0.05)

    metadata = {
        "formula": formula,
        "family": "Gaussian",
        "link": "identity",
        "working_correlation": "Exchangeable",
        "covariance": "robust sandwich",
        "n_observations": int(result.nobs),
        "n_clusters": int(len(set(work["incident_id"]))),
        "iterations": int(result.fit_history.get("iteration", []).__len__()),
        "scale": float(result.scale),
    }
    return pd.DataFrame(rows), metadata


def paired_frs_c_degradation(
    scored: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    wide = scored.pivot_table(
        index=["incident_id", "model"],
        columns="evidence_condition",
        values="FRS_C",
        aggfunc="first",
        observed=True,
    )
    incidents = sorted(scored["incident_id"].unique())
    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS[1:]:
        pair = wide[["FULL", condition]].dropna()
        delta = pair["FULL"] - pair[condition]
        incident_delta = (
            delta.groupby(level="incident_id").mean().reindex(incidents)
        )
        vals = incident_delta.to_numpy(float)
        lo, hi = bootstrap_ci(vals, rng)
        rows.append(
            {
                "evidence_condition": condition,
                "reference_condition": "FULL",
                "paired_degradation_FULL_minus_modified": float(np.nanmean(vals)),
                "ci95_low": lo,
                "ci95_high": hi,
                "paired_incident_model_n": int(len(pair)),
                "contributing_incident_n": int(np.sum(~np.isnan(vals))),
            }
        )

    return pd.DataFrame(rows)


def component_condition_summary(
    scored: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    incidents = sorted(scored["incident_id"].unique())
    rows: list[dict[str, Any]] = []

    for component, column in COMPONENTS.items():
        inc_cond = (
            scored.groupby(["incident_id", "evidence_condition"], observed=True)[column]
            .mean()
            .reset_index()
        )
        for condition in CONDITIONS:
            sub = (
                inc_cond[inc_cond["evidence_condition"] == condition]
                .set_index("incident_id")
                .reindex(incidents)
            )
            vals = sub[column].to_numpy(float)
            lo, hi = bootstrap_ci(vals, rng)
            rows.append(
                {
                    "component": component,
                    "evidence_condition": condition,
                    "mean_score": float(np.nanmean(vals)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "scored_reconstruction_n": int(
                        (
                            (scored["evidence_condition"] == condition)
                            & scored[column].notna()
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def component_paired_degradation(
    scored: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    incidents = sorted(scored["incident_id"].unique())
    rows: list[dict[str, Any]] = []

    for component, column in COMPONENTS.items():
        wide = scored.pivot_table(
            index=["incident_id", "model"],
            columns="evidence_condition",
            values=column,
            aggfunc="first",
            observed=True,
        )
        for condition in CONDITIONS[1:]:
            pair = wide[["FULL", condition]].dropna()
            delta = pair["FULL"] - pair[condition]
            incident_delta = (
                delta.groupby(level="incident_id").mean().reindex(incidents)
            )
            vals = incident_delta.to_numpy(float)
            lo, hi = bootstrap_ci(vals, rng)
            rows.append(
                {
                    "component": component,
                    "evidence_condition": condition,
                    "paired_degradation_FULL_minus_modified": float(np.nanmean(vals)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "paired_incident_model_n": int(len(pair)),
                    "contributing_incident_n": int(np.sum(~np.isnan(vals))),
                }
            )

    return pd.DataFrame(rows)


def model_descriptive(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(scored["model"].unique()):
        sub = scored[scored["model"] == model]
        rows.append(
            {
                "model": model,
                "evidence_condition": "ALL",
                "mean_FRS_C": float(sub["FRS_C"].mean()),
                "n": int(sub["FRS_C"].notna().sum()),
            }
        )
        for condition in CONDITIONS:
            c = sub[sub["evidence_condition"] == condition]
            rows.append(
                {
                    "model": model,
                    "evidence_condition": condition,
                    "mean_FRS_C": float(c["FRS_C"].mean()),
                    "n": int(c["FRS_C"].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def strict_pass_summary(scores_dir: Path) -> pd.DataFrame:
    score_files = sorted(scores_dir.glob("*.json"))
    if len(score_files) != 3599:
        raise ValueError(
            f"Expected 3599 per-task score JSON files, found {len(score_files)}"
        )

    counts: dict[str, int] = {}
    all_pass = 0
    for path in score_files:
        score = load_json(path)
        strict = score.get("strict_contract") or {}
        passes = strict.get("passes") or {}
        for key, value in passes.items():
            counts[key] = counts.get(key, 0) + int(bool(value))
        all_pass += int(bool(strict.get("all_pass")))

    rows = [
        {
            "strict_element": key,
            "pass_count": count,
            "denominator_scored": len(score_files),
            "pass_rate": count / len(score_files),
        }
        for key, count in sorted(
            counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]
    rows.append(
        {
            "strict_element": "ALL_STRICT_REQUIREMENTS",
            "pass_count": all_pass,
            "denominator_scored": len(score_files),
            "pass_rate": all_pass / len(score_files),
        }
    )
    return pd.DataFrame(rows)


def completion_summary(full_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for status, n in full_df["scoring_status"].value_counts(dropna=False).items():
        rows.append(
            {
                "category": "scoring_status",
                "value": str(status),
                "count": int(n),
                "proportion_of_3600": float(n / 3600),
            }
        )
    for source, n in full_df["source_status"].value_counts(dropna=False).items():
        rows.append(
            {
                "category": "source_status",
                "value": str(source),
                "count": int(n),
                "proportion_of_3600": float(n / 3600),
            }
        )
    return pd.DataFrame(rows)


def report_text(
    condition_summary: pd.DataFrame,
    gee: pd.DataFrame,
    paired: pd.DataFrame,
    strict: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("AGENTTRACE-LLM FINAL STATISTICAL ANALYSIS")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Condition-level FRS-C (incident-macro mean, 95% cluster bootstrap CI):")
    for _, r in condition_summary.iterrows():
        lines.append(
            f"  {r['evidence_condition']}: "
            f"{r['FRS_C_mean']:.6f} "
            f"[{r['FRS_C_ci95_low']:.6f}, {r['FRS_C_ci95_high']:.6f}] "
            f"(scored n={int(r['scored_n'])})"
        )

    lines.append("")
    lines.append("Primary GEE condition contrasts (modified minus FULL; Holm adjusted):")
    for _, r in gee.iterrows():
        lines.append(
            f"  {r['evidence_condition']}: "
            f"{r['adjusted_difference_modified_minus_FULL']:+.6f} "
            f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}], "
            f"p_Holm={r['p_holm']:.6g}"
        )

    lines.append("")
    lines.append("Paired FRS-C degradation (FULL minus modified; positive = loss):")
    for _, r in paired.iterrows():
        lines.append(
            f"  {r['evidence_condition']}: "
            f"{r['paired_degradation_FULL_minus_modified']:+.6f} "
            f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}], "
            f"pairs={int(r['paired_incident_model_n'])}"
        )

    lines.append("")
    lines.append("Strict forensic recovery:")
    lines.append("  FRR-S = 0 for all 3600 planned reconstructions.")
    lines.append(
        "  Binomial GEE for FRR-S was NOT estimated because the outcome has zero variance."
    )
    lines.append("  Strict-element pass rates among the 3599 scored reconstructions:")
    for _, r in strict.iterrows():
        if r["strict_element"] == "ALL_STRICT_REQUIREMENTS":
            continue
        lines.append(
            f"    {r['strict_element']}: "
            f"{int(r['pass_count'])}/{int(r['denominator_scored'])} "
            f"({100*r['pass_rate']:.2f}%)"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Final statistical analysis for AgentTrace-LLM scoring_v1_1."
    )
    ap.add_argument(
        "--scoring-dir",
        required=True,
        type=Path,
        help="Path to results/full_experiment/scoring_v1_1",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: sibling results/full_experiment/analysis_v1_0",
    )
    ap.add_argument(
        "--run",
        action="store_true",
        help="Write analysis outputs. Without --run, validation only.",
    )
    args = ap.parse_args()

    scoring_dir = args.scoring_dir.resolve()
    summary_path = scoring_dir / "scoring_summary.csv"
    manifest_path = scoring_dir / "scoring_manifest.json"
    scores_dir = scoring_dir / "scores"

    if not summary_path.exists():
        raise SystemExit(f"Missing: {summary_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing: {manifest_path}")
    if not scores_dir.exists():
        raise SystemExit(f"Missing: {scores_dir}")

    manifest = load_json(manifest_path)
    if manifest.get("task_count") != 3600:
        raise SystemExit(f"Expected task_count=3600, got {manifest.get('task_count')}")
    if manifest.get("scored_count") != 3599:
        raise SystemExit(f"Expected scored_count=3599, got {manifest.get('scored_count')}")
    if manifest.get("recovered_for_scoring_count") != 4:
        raise SystemExit(
            "Expected recovered_for_scoring_count=4, got "
            f"{manifest.get('recovered_for_scoring_count')}"
        )
    if manifest.get("unscored_truncated_count") != 1:
        raise SystemExit(
            "Expected unscored_truncated_count=1, got "
            f"{manifest.get('unscored_truncated_count')}"
        )
    if manifest.get("failed_scoring_count") != 0:
        raise SystemExit(
            f"Expected failed_scoring_count=0, got {manifest.get('failed_scoring_count')}"
        )

    df = pd.read_csv(summary_path)
    if len(df) != 3600:
        raise SystemExit(f"Expected 3600 rows in scoring_summary.csv, found {len(df)}")

    scored = df[df["scoring_status"] == "SCORED"].copy()
    unscored = df[df["scoring_status"] != "SCORED"].copy()
    if len(scored) != 3599 or len(unscored) != 1:
        raise SystemExit(
            f"Expected 3599 scored + 1 unscored, found {len(scored)} + {len(unscored)}"
        )

    u = unscored.iloc[0]
    expected_unscored = (
        u["task_id"] == "M05-C098-PLL"
        and u["incident_id"] == "AT-INC-098"
        and u["evidence_condition"] == "PARTIAL_LOG_LOSS"
        and u["source_status"] == "TRUNCATED"
    )
    if not expected_unscored:
        raise SystemExit(
            "Unexpected identity of the single unscored task: "
            f"{u['task_id']} | {u['incident_id']} | "
            f"{u['evidence_condition']} | {u['source_status']}"
        )

    if set(df["evidence_condition"]) != set(CONDITIONS):
        raise SystemExit("Evidence-condition set does not match the frozen six-condition design")
    if df["incident_id"].nunique() != 100:
        raise SystemExit(f"Expected 100 incidents, found {df['incident_id'].nunique()}")
    if df["model"].nunique() != 6:
        raise SystemExit(f"Expected 6 models, found {df['model'].nunique()}")
    if not (df["FRR_S"].fillna(0) == 0).all():
        raise SystemExit("FRR-S is not constant zero; analysis contract must be reviewed")

    print("AGENTTRACE-LLM FINAL ANALYSIS - VALIDATION")
    print(f"Summary: {summary_path}")
    print(f"Summary SHA256: {sha256_file(summary_path)}")
    print(f"Scoring manifest: {manifest_path}")
    print(f"Scorer version: {manifest.get('scorer_version')}")
    print(f"Metric contract: {manifest.get('metric_contract_version')}")
    print("Tasks: 3600")
    print("Scored: 3599")
    print("Unscored: 1 (M05-C098-PLL)")
    print("Incidents: 100")
    print("Models: 6")
    print("Conditions: 6")
    print("FRR-S variation: none (all 0)")
    print(f"Bootstrap: {BOOTSTRAP_REPS} incident-cluster resamples, seed={BOOTSTRAP_SEED}")

    if not args.run:
        print("\nDRY RUN ONLY. No analysis files written.")
        print("Re-run with --run to perform the final statistical analysis.")
        return 0

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (scoring_dir.parent / "analysis_v1_0").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    completion = completion_summary(df)
    condition = macro_condition_table(scored, df, rng)
    gee, gee_meta = fit_frs_c_gee(scored)
    paired = paired_frs_c_degradation(scored, rng)
    component_cond = component_condition_summary(scored, rng)
    component_delta = component_paired_degradation(scored, rng)
    models = model_descriptive(scored)
    strict = strict_pass_summary(scores_dir)

    completion.to_csv(output_dir / "01_completion_summary.csv", index=False)
    condition.to_csv(output_dir / "02_condition_summary.csv", index=False)
    gee.to_csv(output_dir / "03_frs_c_gee_condition_contrasts.csv", index=False)
    paired.to_csv(output_dir / "04_paired_frs_c_degradation.csv", index=False)
    component_cond.to_csv(output_dir / "05_component_condition_summary.csv", index=False)
    component_delta.to_csv(output_dir / "06_component_paired_degradation.csv", index=False)
    models.to_csv(output_dir / "07_model_descriptive_frs_c.csv", index=False)
    strict.to_csv(output_dir / "08_strict_pass_rates.csv", index=False)

    report = report_text(condition, gee, paired, strict)
    (output_dir / "analysis_report.txt").write_text(report, encoding="utf-8")

    analysis_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scoring_dir": str(scoring_dir),
        "source_scoring_summary": str(summary_path),
        "source_scoring_summary_sha256": sha256_file(summary_path),
        "source_scoring_manifest_sha256": sha256_file(manifest_path),
        "source_scorer_sha256": manifest.get("scorer_sha256"),
        "source_scorer_version": manifest.get("scorer_version"),
        "source_metric_contract_version": manifest.get("metric_contract_version"),
        "task_count": 3600,
        "scored_count": 3599,
        "unscored_count": 1,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "frs_c_primary_model": gee_meta,
        "frr_s_analysis": {
            "observed_value_all_tasks": 0.0,
            "binomial_gee_estimated": False,
            "reason": "FRR-S has zero variance (all 3600 values are 0).",
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    write_json(output_dir / "analysis_manifest.json", analysis_manifest)

    print("\nFINAL ANALYSIS COMPLETE")
    print(f"Output: {output_dir}")
    print("")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
