"""Post-hoc held-out risk--coverage comparison at equal decision coverage.

The joint evidence score is based only on thresholds learned on validation:
max(normalized width / dual ACT width threshold,
    dual explanation stability threshold / observed stability).
Lower scores indicate stronger evidence. At each predeclared test coverage,
select the same number of cases for uncertainty-only (width ranking) and dual
(joint-score ranking), then compare LightGBM test errors. Test outcomes are
never used to rank/select cases.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "matched_coverage_test.csv"
OUT_JSON = RESULTS / "matched_coverage_test.json"
OUT_MD = ROOT / "reports" / "matched_coverage_test.md"


def main():
    decision = pd.read_csv(RESULTS / "decision_log.csv")
    forecasts = pd.read_csv(RESULTS / "test_forecasts.csv")
    decision["origin"] = pd.to_datetime(decision["origin"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    windows = (forecasts.groupby(["origin", "household_id_code"], sort=False)
        .agg(actual=("actual_kw", list), prediction=("lightgbm_kw", list))
        .reset_index().rename(columns={"household_id_code": "household_code"}))
    windows["origin"] = pd.to_datetime(windows["origin"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    data = decision.merge(windows, on=["origin", "household_code"], validate="one_to_one")
    thresholds = json.loads((RESULTS / "decision_thresholds.json").read_text())
    u_max = float(thresholds["dual_evidence"]["uncertainty_act_max"])
    s_min = float(thresholds["dual_evidence"]["explanation_act_min"])
    data["uncertainty_score"] = data["normalized_interval_width"].astype(float)
    data["dual_score"] = np.maximum(
        data["uncertainty_score"] / u_max,
        s_min / np.maximum(data["explanation_stability"].astype(float), 1e-12),
    )

    def evaluate(indices):
        errors = np.concatenate([
            np.asarray(data.iloc[i]["prediction"], dtype=float)
            - np.asarray(data.iloc[i]["actual"], dtype=float)
            for i in indices
        ])
        return {
            "mae_kw": float(np.mean(np.abs(errors))),
            "rmse_kw": float(np.sqrt(np.mean(errors**2))),
            "bias_kw": float(np.mean(errors)),
        }

    rows = []
    selected_at_20 = {}
    for coverage in (0.10, 0.20, 0.30, 0.40, 0.50):
        n = max(1, int(round(coverage * len(data))))
        uncertainty = data.nsmallest(n, "uncertainty_score").index.to_numpy()
        dual = data.nsmallest(n, "dual_score").index.to_numpy()
        mu, md = evaluate(uncertainty), evaluate(dual)
        rows.extend([
            {"nominal_coverage": coverage, "origins": n, "method": "uncertainty_only", **mu},
            {"nominal_coverage": coverage, "origins": n, "method": "uncertainty_plus_shap", **md},
        ])
        if coverage == .20:
            selected_at_20 = {"uncertainty": uncertainty, "dual": dual}
    output = pd.DataFrame(rows)
    output.to_csv(OUT_CSV, index=False)

    # Household-cluster bootstrap of the 20% ΔMAE, conditional on frozen selections.
    uids, dids = selected_at_20["uncertainty"], selected_at_20["dual"]
    homes = data["household_code"].unique()
    def household_error_sums(indices):
        sums, counts = {}, {}
        for i in indices:
            household = int(data.iloc[i]["household_code"])
            err = np.abs(np.asarray(data.iloc[i]["prediction"], dtype=float)
                         - np.asarray(data.iloc[i]["actual"], dtype=float))
            sums[household] = sums.get(household, 0.0) + float(err.sum())
            counts[household] = counts.get(household, 0) + len(err)
        return sums, counts
    du_s, du_n = household_error_sums(dids)
    un_s, un_n = household_error_sums(uids)
    rng = np.random.default_rng(42)
    diffs = np.empty(5000)
    for b in range(len(diffs)):
        sampled = rng.choice(homes, size=len(homes), replace=True)
        weights = {int(h): int(np.sum(sampled == h)) for h in homes}
        dual_mae = sum(weights[h] * du_s.get(h, 0.0) for h in weights) / sum(weights[h] * du_n.get(h, 0) for h in weights)
        unc_mae = sum(weights[h] * un_s.get(h, 0.0) for h in weights) / sum(weights[h] * un_n.get(h, 0) for h in weights)
        diffs[b] = dual_mae - unc_mae
    common = len(set(uids) & set(dids))
    matched = {
        "test_windows": int(len(data)),
        "coverage_target": 0.20,
        "selected_origins_per_method": int(len(uids)),
        "realized_coverage": float(len(uids) / len(data)),
        "overlapping_origins": int(common),
        "dual_minus_uncertainty_mae_95pct_household_cluster_bootstrap_ci_kw": [float(x) for x in np.quantile(diffs, [.025, .975])],
        "bootstrap_replicates": int(len(diffs)),
        "bootstrap_seed": 42,
        "households_bootstrapped": int(len(homes)),
        "bootstrap_interpretation": "conditional on the test-set score rankings and selected sets",
        "thresholds_frozen_from_validation": True,
        "test_outcomes_used_for_selection": False,
        "interpretation": "Post-hoc risk-coverage diagnostic. The equal-coverage comparison ranks held-out evidence scores at the stated coverage; it does not replace the frozen-threshold operational test comparison.",
        "dual_validation_thresholds": {"uncertainty_act_max": u_max, "explanation_act_min": s_min},
    }
    for method, ids in selected_at_20.items():
        matched[method] = evaluate(ids)
    OUT_JSON.write_text(json.dumps(matched, indent=2), encoding="utf-8")
    md = [
        "# Matched-Coverage Test Comparison", "",
        "This is a post-hoc risk--coverage comparison on the held-out test origins. The uncertainty-only policy ranks cases by normalized interval width. The dual-evidence policy ranks by the weaker normalized margin across interval width and SHAP stability, using ACT thresholds fixed on validation. For each predeclared coverage, both methods select the same number of test origins. Test outcomes are not used for ranking or selection.", "",
        "| Test coverage | Selected origins per method | Uncertainty-only MAE (kW) | Dual-evidence MAE (kW) | Uncertainty-only RMSE (kW) | Dual-evidence RMSE (kW) |", "|---:|---:|---:|---:|---:|---:|"
    ]
    for coverage in (0.10, 0.20, 0.30, 0.40, 0.50):
        a = output[(output.nominal_coverage == coverage) & (output.method == "uncertainty_only")].iloc[0]
        b = output[(output.nominal_coverage == coverage) & (output.method == "uncertainty_plus_shap")].iloc[0]
        md.append(f"| {coverage:.0%} | {int(a.origins)} | {a.mae_kw:.4f} | {b.mae_kw:.4f} | {a.rmse_kw:.4f} | {b.rmse_kw:.4f} |")
    u, d = matched["uncertainty"], matched["dual"]
    md += ["", f"At 20% target coverage ({matched['selected_origins_per_method']} origins; realized {matched['realized_coverage']:.2%}), dual-evidence MAE was {d['mae_kw']:.4f} kW versus {u['mae_kw']:.4f} kW for uncertainty-only. The dual-selected and uncertainty-selected sets overlap on {common} origins. Conditional household-cluster bootstrap (n={matched['households_bootstrapped']} households) 95% interval for the difference (dual minus uncertainty MAE) was [{matched['dual_minus_uncertainty_mae_95pct_household_cluster_bootstrap_ci_kw'][0]:.4f}, {matched['dual_minus_uncertainty_mae_95pct_household_cluster_bootstrap_ci_kw'][1]:.4f}] kW.", "", "This is evidence that the joint reliability score ranked lower-error forecasts more effectively at matched coverage in this test set. It is not a causal estimate of SHAP’s value, does not establish operational savings, and should be presented as a retrospective risk--coverage diagnostic alongside the frozen-threshold results.", ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_JSON)
    print(json.dumps(matched, indent=2))

if __name__ == "__main__":
    main()
