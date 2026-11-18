"""Run the complete four-model, dual-evidence research evaluation."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from dual_evidence.controller.decision import decide, select_thresholds, thresholds_as_dict
from dual_evidence.evaluation.metrics import forecast_metrics, selective_metrics
from dual_evidence.evaluation.sensitivity import evaluate_zero_run_sensitivity
from dual_evidence.explainability.stability import explain_and_measure_stability
from dual_evidence.forecasting.chronos2 import Chronos2Forecaster
from dual_evidence.forecasting.classical import LightGBMForecaster, seasonal_naive
from dual_evidence.forecasting.patchtst import PatchTSTForecaster
from dual_evidence.forecasting.windows import build_windows
from dual_evidence.optimization.demand_response import optimize_day_ahead
from dual_evidence.uncertainty.conformal import conformal_quantile, interval_metrics, prediction_interval
from dual_evidence.verification.secondary import verify_forecast
from dual_evidence.agents.orchestrator import configured_coordinator
from dual_evidence.logging_utils import configure_logging


def _save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.item()), encoding="utf-8")


def _model_outputs(windows, train, validation):
    Path("results").mkdir(exist_ok=True)
    prediction_splits = ("validation", "calibration", "test")
    def load_cache(model):
        path = Path(f"results/forecast_cache_{model}.npz")
        if not path.exists():
            return None
        with np.load(path) as cached:
            values = {name: cached[name].copy() for name in prediction_splits}
        if any(values[name].shape != (len(windows[name]), 48) for name in prediction_splits):
            return None
        values["train"] = np.empty((0, 48), dtype=np.float32)
        logging.info("Resuming cached %s forecasts", model)
        return values
    def save_cache(model, values):
        np.savez_compressed(f"results/forecast_cache_{model}.npz",
                            **{name: values[name] for name in prediction_splits})

    naive_cache = load_cache("seasonal_naive")
    outputs = {"seasonal_naive": naive_cache or
               {name: seasonal_naive(windows[name]) for name in prediction_splits}}
    outputs["seasonal_naive"]["train"] = np.empty((0, 48), dtype=np.float32)
    if naive_cache is None: save_cache("seasonal_naive", outputs["seasonal_naive"])
    statuses = {"seasonal_naive": {"status": "complete", "type": "baseline"}}
    cfg = yaml.safe_load(Path("configs/models.yaml").read_text())
    Path("models").mkdir(exist_ok=True)
    lgb_cache = load_cache("lightgbm")
    model_path = Path("models/lightgbm_pooled.joblib")
    if model_path.exists():
        lgb = joblib.load(model_path)
        logging.info("Resuming saved pooled LightGBM model")
    else:
        lgb = LightGBMForecaster(**{k: cfg["lightgbm"][k] for k in
            ("n_estimators", "learning_rate", "num_leaves", "max_depth", "random_state", "n_jobs")})
        logging.info("Fitting pooled 48-horizon LightGBM on %d windows", len(train))
        lgb.fit(train)
        joblib.dump(lgb, model_path)
    outputs["lightgbm"] = lgb_cache or {name: lgb.predict(windows[name]) for name in prediction_splits}
    outputs["lightgbm"]["train"] = np.empty((0, 48), dtype=np.float32)
    if lgb_cache is None: save_cache("lightgbm", outputs["lightgbm"])
    statuses["lightgbm"] = {"status": "complete", "n_estimators_per_horizon": lgb.n_estimators,
                             "training_windows": len(train)}

    patch = PatchTSTForecaster(epochs=int(cfg.get("patchtst", {}).get("epochs", 8)),
                                batch_size=int(cfg.get("patchtst", {}).get("batch_size", 128)))
    try:
        import torch
        patch_cache = load_cache("patchtst")
        patch_path = Path("models/patchtst_state.pt")
        if patch_path.exists():
            patch.network = patch._build()[1]
            patch.network.load_state_dict(torch.load(patch_path, map_location=patch.device, weights_only=True))
            logging.info("Resuming saved PatchTST model")
        else:
            logging.info("Training CPU PatchTST")
            patch.fit(train, validation)
            torch.save(patch.network.state_dict(), patch_path)
        outputs["patchtst"] = patch_cache or {name: patch.predict(windows[name]) for name in prediction_splits}
        outputs["patchtst"]["train"] = np.empty((0, 48), dtype=np.float32)
        if patch_cache is None: save_cache("patchtst", outputs["patchtst"])
        statuses["patchtst"] = {"status": "complete", "epochs": patch.epochs,
                                  "training_windows": len(train), "device": patch.device}
    except Exception as exc:
        logging.exception("PatchTST execution failed")
        outputs["patchtst"] = None
        statuses["patchtst"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    chrono = Chronos2Forecaster(model_id=cfg.get("chronos2", {}).get("model_id", "amazon/chronos-2"),
                                context_length=336, batch_size=32)
    try:
        chrono_cache = load_cache("chronos2")
        if chrono_cache is None:
            logging.info("Loading and evaluating zero-shot Chronos-2")
            chrono.load()
            outputs["chronos2"] = {name: chrono.predict(windows[name]) for name in prediction_splits}
            outputs["chronos2"]["train"] = np.empty((0, 48), dtype=np.float32)
            save_cache("chronos2", outputs["chronos2"])
        else:
            outputs["chronos2"] = chrono_cache
        statuses["chronos2"] = {"status": "complete", "model_id": chrono.model_id,
                                  "mode": "zero-shot", "training_windows": 0}
    except Exception as exc:
        logging.exception("Chronos-2 execution failed")
        outputs["chronos2"] = None
        statuses["chronos2"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return lgb, outputs, statuses


def _rows_for_windows(model_outputs, windows):
    data = {"origin": np.repeat(pd.to_datetime(windows.origin).astype(str), 48),
            "household_id_code": np.repeat(windows.household, 48),
            "horizon_step": np.tile(np.arange(1, 49), len(windows)),
            "actual_kw": windows.y.reshape(-1)}
    for model, predictions in model_outputs.items():
        if predictions is not None:
            data[f"{model}_kw"] = predictions.reshape(-1)
    return pd.DataFrame(data)


def _plot_results(metrics, configs):
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    names = [row["model"] for row in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, [row["mae_kw"] for row in metrics], color="#4263eb")
    axes[0].set_ylabel("MAE (kW)"); axes[0].set_title("Test forecast accuracy")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, [row.get("picp", float("nan")) for row in metrics], color="#2f9e44")
    axes[1].axhline(.9, color="#d9480f", linestyle="--", label="90% target")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("PICP"); axes[1].set_title("90% conformal interval coverage")
    axes[1].tick_params(axis="x", rotation=20); axes[1].legend()
    fig.tight_layout(); fig.savefig("reports/figures/model_comparison.png", dpi=160); plt.close(fig)
    keys = list(configs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(keys, [configs[k]["automation_coverage"] for k in keys], color="#7048e8")
    axes[0].set_ylim(0, 1); axes[0].set_title("Automation coverage")
    x = np.arange(len(keys)); width = .36
    axes[1].bar(x-width/2, [configs[k].get("cost_increase_rate", 0) for k in keys],
                width, label="Cost increase", color="#e8590c")
    axes[1].bar(x+width/2, [configs[k].get("peak_increase_rate", 0) for k in keys],
                width, label="Peak increase", color="#f08c00")
    axes[1].set_xticks(x, keys); axes[1].set_ylim(0, 1)
    axes[1].set_title("Cost / peak increase among ACTs"); axes[1].legend()
    for ax in axes: ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig("reports/figures/controller_comparison.png", dpi=160); plt.close(fig)


def run():
    load_dotenv()
    configure_logging("logs")
    Path("results").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    ecfg = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    tool_calls = []
    coordinator = configured_coordinator({
        "forecast": _model_outputs,
        "conformal": conformal_quantile,
        "explanation_stability": explain_and_measure_stability,
        "controller": decide,
        "verification": verify_forecast,
        "demand_response": optimize_day_ahead,
    })
    def call_tool(name, *args, **kwargs):
        tool_calls.append({"tool": name, "invocation": len(tool_calls)+1})
        return coordinator.run_tool(name, *args, **kwargs)
    cfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    logging.info("Reading 30-minute panel")
    panel = pd.read_csv(Path(cfg["processed_dir"])/"panel_30min.csv", parse_dates=["timestamp"])
    windows = build_windows(panel, cfg["history_context_steps"], cfg["horizon_steps"],
                            cfg["forecast_origin_stride_steps"], cfg["split_ratios"])
    logging.info("Forecast windows: %s", {k: len(v) for k, v in windows.items()})
    lgb, predictions, statuses = call_tool("forecast", windows, windows["train"], windows["validation"])
    _save_json(statuses, "results/model_status.json")

    calibrated, interval_results, metric_rows = {}, {}, []
    for model, output in predictions.items():
        if output is None:
            continue
        alpha = float(ecfg["conformal_alpha"])
        q = call_tool("conformal", windows["calibration"].y, output["calibration"], alpha=alpha)
        lower, upper = prediction_interval(output["test"], q)
        calibrated[model] = q
        interval_results[model] = (lower, upper)
        scores = {**forecast_metrics(windows["test"].y, output["test"]),
                  **interval_metrics(windows["test"].y, lower, upper),
                  "model": model, "alpha": alpha, "calibration_windows": len(windows["calibration"])}
        metric_rows.append(scores)
    pd.DataFrame(metric_rows).to_csv("results/model_metrics.csv", index=False)
    zero_sensitivity = evaluate_zero_run_sensitivity(
        panel, windows["test"], {name: values["test"] if values is not None else None
                                 for name, values in predictions.items()})
    _save_json(zero_sensitivity, "results/zero_run_sensitivity.json")
    _save_json({key: value.tolist() for key, value in calibrated.items()}, "results/conformal_quantiles.json")
    test_rows = _rows_for_windows({k: v["test"] if v is not None else None for k, v in predictions.items()}, windows["test"])
    test_rows.to_csv("results/test_forecasts.csv", index=False)

    logging.info("Computing TreeSHAP perturbation stability on validation/test origins")
    perturbations = int(ecfg["explanation_perturbations"])
    expl_seed = int(ecfg["explanation_seed"])
    explanation_cache = Path("results/explanation_cache.npz")
    if explanation_cache.exists():
        with np.load(explanation_cache) as cached:
            val_stability = cached["validation"].copy()
            test_stability = cached["test"].copy()
        global_shap = pd.read_csv("results/shap_global_importance.csv").to_dict("records")
        local_shap = json.loads(Path("results/shap_stability.json").read_text())["local_examples"]
        logging.info("Resuming cached TreeSHAP stability vectors")
    else:
        val_stability, global_shap, _ = call_tool("explanation_stability", lgb, windows["validation"], perturbations, expl_seed)
        test_stability, _, local_shap = call_tool("explanation_stability", lgb, windows["test"], perturbations, expl_seed)
        np.savez_compressed(explanation_cache, validation=val_stability, test=test_stability)
        pd.DataFrame(global_shap).to_csv("results/shap_global_importance.csv", index=False)
    _save_json({"method": "LightGBM pred_contrib TreeSHAP",
                "perturbations": f"{perturbations} perturbations: 1% demand; 0.1 standard deviation weather; seed {expl_seed}",
                "mean_validation_spearman_stability": float(val_stability.mean()),
                "mean_test_spearman_stability": float(test_stability.mean()),
                "local_examples": ([{"origin": str(pd.Timestamp(windows['test'].origin[i])),
                                     "top_features": local_shap[i]}
                                    for i in range(min(len(local_shap), 100))]
                                   if len(local_shap) and isinstance(local_shap[0], list)
                                   else local_shap)},
               "results/shap_stability.json")

    q = calibrated["lightgbm"]
    validation_width = np.mean(2*q) / np.maximum(np.mean(np.abs(predictions["lightgbm"]["validation"]), axis=1), .05)
    test_width = np.mean(2*q) / np.maximum(np.mean(np.abs(predictions["lightgbm"]["test"]), axis=1), .05)
    dc = yaml.safe_load(Path("configs/decision_thresholds.yaml").read_text())
    thresholds = select_thresholds(validation_width, val_stability,
        dc["uncertainty_quantile"], dc["explanation_quantile"],
        dc["verify_uncertainty_quantile"], dc["verify_explanation_quantile"])
    dual_validation = ((validation_width <= thresholds.uncertainty_act_max) &
                       (val_stability >= thresholds.explanation_act_min))
    if dual_validation.mean() == 0:
        matched_uncertainty_threshold = float(np.nextafter(np.min(validation_width), -np.inf))
    else:
        matched_uncertainty_threshold = float(np.quantile(
            validation_width, float(dual_validation.mean()) - .5/len(validation_width)))
    matched_uncertainty_coverage = float(np.mean(validation_width <= matched_uncertainty_threshold))
    thresholds = type(thresholds)(thresholds.uncertainty_act_max,
        thresholds.explanation_act_min, thresholds.uncertainty_verify_max,
        thresholds.explanation_verify_min, float(dc.get("verification_relative_tolerance", .25)))
    _save_json({"dual_evidence": thresholds_as_dict(thresholds),
                "uncertainty_only_matched_coverage_threshold": matched_uncertainty_threshold,
                "validation_dual_coverage": float(dual_validation.mean()),
                "validation_uncertainty_only_coverage": matched_uncertainty_coverage,
                "validation_coverage_gap": abs(matched_uncertainty_coverage - float(dual_validation.mean())),
                "threshold_source": "validation origins only; conformal scores from calibration split"},
               "results/decision_thresholds.json")

    records, config_decisions = [], {"ungated": [], "uncertainty_only": [],
                                    "explanation_only": [], "dual_evidence": []}
    secondary_name = "patchtst"
    if predictions.get(secondary_name) is None:
        raise RuntimeError("PatchTST is required for secondary verification but its predictions are unavailable")
    secondary_test = predictions[secondary_name]["test"]
    lgb_test = predictions["lightgbm"]["test"]
    for i in range(len(windows["test"])):
        width, stable = float(test_width[i]), float(test_stability[i])
        unc = {"decision": "ACT" if width <= matched_uncertainty_threshold else "ABSTAIN",
               "reason": "matched validation coverage uncertainty rule"}
        exp = {"decision": "ACT" if stable >= thresholds.explanation_act_min else "ABSTAIN",
               "reason": "validation explanation stability threshold"}
        initial = call_tool("controller", width, stable, thresholds)
        comparison = (call_tool("verification", lgb_test[i], secondary_test[i],
                               thresholds.verification_relative_tolerance)
                      if initial["decision"] == "VERIFY" else None)
        outcome = call_tool("controller", width, stable, thresholds,
                            comparison["agrees"] if comparison is not None else None)
        config_decisions["ungated"].append({"decision": "ACT", "reason": "ungated reference"})
        config_decisions["uncertainty_only"].append(unc)
        config_decisions["explanation_only"].append(exp)
        config_decisions["dual_evidence"].append({**outcome, "initial_decision": initial["decision"],
                                                    "verification": comparison if initial["decision"] == "VERIFY" else None})
        records.append({"origin": str(pd.Timestamp(windows["test"].origin[i])),
            "household_code": int(windows["test"].household[i]), "normalized_interval_width": width,
            "explanation_stability": stable, "initial_dual_decision": initial["decision"],
            "dual_decision": outcome["decision"], "dual_reason": outcome["reason"],
            "verification_relative_difference": comparison["mean_relative_difference"] if comparison else None,
            "verification_agrees": comparison["agrees"] if comparison else None})
    pd.DataFrame(records).to_csv("results/decision_log.csv", index=False)

    # DR is simulated only after each configuration returns ACT. One in every seven
    # test days is sampled per home to keep repeated convex solves manageable.
    dr_summaries = {}
    dr_records = []
    home_day_counter = {}
    for i, origin64 in enumerate(windows["test"].origin):
        key = int(windows["test"].household[i]); day = pd.Timestamp(origin64).date()
        seen_key = (key, day)
        home_day_counter[seen_key] = home_day_counter.get(seen_key, 0) + 1
        if home_day_counter[seen_key] > 1:
            continue
        # Windows are already daily; deterministic one-in-seven sampling by ordinal.
        if day.toordinal() % int(ecfg["dr_evaluation_stride_days"]) != 0:
            continue
        for name, decisions in config_decisions.items():
            decision = decisions[i]
            if decision["decision"] != "ACT":
                continue
            forecast_model = "lightgbm"
            result = call_tool("demand_response", predictions[forecast_model]["test"][i], windows["test"].y[i],
                               flexible_share=float(ecfg["dr_flexible_share"]),
                               peak_penalty_pkr_per_kw=float(ecfg.get("dr_peak_penalty_pkr_per_kw", 50.0)))
            dr_records.append({"origin": str(pd.Timestamp(origin64)), "config": name, **{
                k: v for k, v in result.items() if k not in ("scheduled_flexible_kw", "optimized_actual_kw")}})
            dr_summaries.setdefault(name, []).append(result)
    pd.DataFrame(dr_records).to_csv("results/demand_response_decisions.csv", index=False)
    for name in config_decisions:
        ds = config_decisions[name]
        # Risk and utility are based on the ACT subset where CVXPY was actually run.
        dmetrics = {}
        dr = dr_summaries.get(name, [])
        dmetrics["automation_coverage"] = float(np.mean([x["decision"] == "ACT" for x in ds]))
        dmetrics["dr_evaluated_actions"] = len(dr)
        dmetrics["cost_increase_rate"] = float(np.mean([x["cost_increase"] for x in dr])) if dr else 0.0
        dmetrics["peak_increase_rate"] = float(np.mean([x["peak_increase"] for x in dr])) if dr else 0.0
        dmetrics["mean_cost_saving_pkr"] = float(np.mean([x["cost_saving_pkr"] for x in dr])) if dr else 0.0
        dmetrics["mean_cost_saving_pct"] = float(np.mean([x["cost_saving_pct"] for x in dr])) if dr else 0.0
        dmetrics["mean_peak_reduction_kw"] = float(np.mean([x["peak_reduction_kw"] for x in dr])) if dr else 0.0
        dr_summaries[name] = dmetrics
    _save_json(dr_summaries, "results/controller_metrics.json")

    # Forecast and interval metrics are genuine test outputs from each executed model.
    _plot_results(metric_rows, dr_summaries)
    lines = ["# Experimental Results", "",
      "All rows below are calculated from the downloaded REWD-P source data. Test forecasts are 48 half-hour steps; conformal intervals use a separate calibration split. The source archive contains 59 unique households; the synopsis says 60. Long zero runs are preserved and flagged.", "",
      "## Forecasting and interval results", "",
      "| Model | MAE kW | RMSE kW | WAPE % | PICP | Mean width kW |", "|---|---:|---:|---:|---:|---:|"]
    for r in metric_rows:
        lines.append(f"| {r['model']} | {r['mae_kw']:.4f} | {r['rmse_kw']:.4f} | {r['wape_pct']:.2f} | {r['picp']:.3f} | {r['mean_interval_width']:.4f} |")
    lines += ["", "## Long-zero-run sensitivity", "",
      f"{zero_sensitivity['test_windows_intersecting_flagged_zero_run']} of {zero_sensitivity['test_windows_total']} test windows intersect a long-zero flag. The table compares all test windows with the subset excluding those windows; this is an evaluation-only sensitivity check, with the training policy unchanged.", "",
      "| Model | All-test MAE | Excluding flagged-window MAE | Flagged-window MAE |", "|---|---:|---:|---:|"]
    for name, values in zero_sensitivity["models"].items():
        clean = values["exclude_flagged_zero_windows"]
        flagged_metrics = values["flagged_zero_windows_only"]
        lines.append(f"| {name} | {values['overall']['mae_kw']:.4f} | "
                     f"{clean['mae_kw'] if clean else float('nan'):.4f} | "
                     f"{flagged_metrics['mae_kw'] if flagged_metrics else float('nan'):.4f} |")
    lines += ["", "## Selective controller and DR comparison", "",
      "| Configuration | Automation | Cost-increase rate | Peak increase rate | Cost saved PKR | Peak reduction kW | DR cases |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in dr_summaries.items():
        lines.append(f"| {name} | {r['automation_coverage']:.3f} | {r.get('cost_increase_rate', 0.0):.3f} | {r.get('peak_increase_rate', 0.0):.3f} | {r['mean_cost_saving_pkr']:.2f} | {r['mean_peak_reduction_kw']:.4f} | {r['dr_evaluated_actions']} |")
    lines += ["", "## Execution status", "", "| Model | Status |", "|---|---|"]
    lines += [f"| {name} | {value['status']}: {value.get('error', value.get('model_id', 'executed'))} |" for name, value in statuses.items()]
    lines += ["", "The DR objective combines illustrative energy cost with a configurable peak-demand penalty. Cost and peak outcomes are reported separately; increases are possible under forecast error. Tariffs and flexible-load share are simulation assumptions, not a utility tariff or measured appliance telemetry. DR optimization executes only for ACT decisions; no real appliance commands are issued. Numeric regret is withheld because the sampled forecast schedule and a fully aligned oracle were not evaluated under a common feasible set.", "",
      "Primary frozen-threshold comparison is Dual-Evidence versus Uncertainty-Only. The Uncertainty-Only threshold is chosen on validation data to match Dual-Evidence validation coverage; test coverage is reported as observed. A separate post-hoc matched-coverage test risk--coverage diagnostic is in `reports/matched_coverage_test.md`.", ""]
    Path("reports/experiment_results.md").write_text("\n".join(lines), encoding="utf-8")
    _save_json({"models": statuses, "metrics": metric_rows, "controller": dr_summaries,
                "zero_run_sensitivity": zero_sensitivity,
                "n_windows": {k: len(v) for k, v in windows.items()},
                "scope": "59 unique households; chronological 70/10/10/10; test untouched during fitting",
                "secondary_verifier": secondary_name,
                "experiment_config": ecfg},
               "results/experiment_summary.json")
    _save_json(tool_calls, "logs/agent_tool_calls.json")
    try:
        narrative = coordinator.summarize({"models": statuses, "metrics": metric_rows,
                                           "controller": dr_summaries})
        _save_json({"status": "complete" if narrative else "disabled"}, "results/llm_status.json")
    except Exception as exc:
        logging.exception("Optional LLM narrative failed; deterministic reports remain complete")
        narrative = None
        _save_json({"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                   "results/llm_status.json")
    if narrative:
        Path("reports/llm_summary.md").write_text(
            "# Optional LLM-Generated Narrative\n\nThe narrative was generated after Python computed the metrics. "
            "The numeric tables in `experiment_results.md` remain authoritative.\n\n" + narrative,
            encoding="utf-8")
    failed = [name for name, status in statuses.items() if status["status"] != "complete"]
    if failed:
        raise RuntimeError("Required forecasting model(s) failed: " + ", ".join(failed) +
                           ". See results/model_status.json and reports/experiment_results.md")
    logging.info("Research experiment completed; see reports/experiment_results.md")


if __name__ == "__main__":
    run()
