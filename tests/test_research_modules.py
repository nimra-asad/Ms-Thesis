import unittest

import numpy as np
import pandas as pd

from dual_evidence.controller.decision import Thresholds, decide, select_thresholds
from dual_evidence.optimization.demand_response import optimize_day_ahead
from dual_evidence.forecasting.classical import seasonal_naive
from dual_evidence.forecasting.windows import build_windows, time_boundaries
from dual_evidence.forecasting.windows import WindowSet
from dual_evidence.evaluation.sensitivity import zero_run_window_mask
from dual_evidence.uncertainty.conformal import conformal_quantile, interval_metrics, prediction_interval
from dual_evidence.verification.secondary import verify_forecast


class ResearchModuleTests(unittest.TestCase):
    def test_split_conformal_finite_sample_quantile_and_coverage(self):
        truth = np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])
        pred = truth - 1
        q = conformal_quantile(truth, pred, alpha=.25)
        self.assertEqual(q.tolist(), [1., 1.])
        lower, upper = prediction_interval(pred, q)
        self.assertEqual(interval_metrics(truth, lower, upper)["picp"], 1.0)

    def test_controller_act_verify_and_abstain_are_deterministic(self):
        thresholds = Thresholds(.2, .7, .5, .4, .25)
        self.assertEqual(decide(.1, .8, thresholds)["decision"], "ACT")
        self.assertEqual(decide(.1, .5, thresholds)["decision"], "VERIFY")
        self.assertEqual(decide(.1, .5, thresholds, True)["decision"], "ACT")
        self.assertEqual(decide(.6, .2, thresholds)["decision"], "ABSTAIN")

    def test_thresholds_use_only_the_passed_validation_signals(self):
        selected = select_thresholds(np.arange(10.), np.linspace(0, 1, 10))
        self.assertAlmostEqual(selected.uncertainty_act_max, 6.3)
        self.assertAlmostEqual(selected.explanation_act_min, .30)

    def test_secondary_verification_reports_relative_difference(self):
        result = verify_forecast(np.array([1., 2.]), np.array([1.1, 2.1]), .1)
        self.assertTrue(result["agrees"])
        self.assertLess(result["mean_relative_difference"], .1)

    def test_dr_preserves_flexible_energy_and_reports_actual_cost(self):
        actual = np.full(48, .4)
        forecast = actual.copy()
        actual[34:44] = 2.0
        forecast[34:44] = 2.0
        forecast[0] = -0.01  # Tree regressors may stray below the physical floor.
        result = optimize_day_ahead(forecast, actual)
        self.assertIn(result["status"], {"optimal", "optimal_inaccurate"})
        self.assertGreater(result["flex_energy_kwh"], 0)
        self.assertAlmostEqual(sum(result["scheduled_flexible_kw"])*.5,
                               result["flex_energy_kwh"], places=3)
        self.assertAlmostEqual(result["baseline_actual_cost_pkr"] - result["optimized_actual_cost_pkr"],
                               result["cost_saving_pkr"], places=5)

    def test_windows_do_not_cross_chronological_split_cutoffs(self):
        timestamps = pd.date_range("2025-01-01", periods=1100, freq="30min")
        panel = pd.DataFrame({"timestamp": timestamps, "household_id": "House1", "city": "Lahore",
                              "target_kw": np.linspace(1, 2, len(timestamps))})
        windows = build_windows(panel)
        boundaries = time_boundaries(panel)
        self.assertGreater(len(windows["train"]), 0)
        self.assertLess(pd.Timestamp(windows["train"].origin[-1]) + pd.Timedelta(hours=23.5),
                        boundaries["train_end"])
        self.assertLess(pd.Timestamp(windows["validation"].origin[-1]) + pd.Timedelta(hours=23.5),
                        boundaries["validation_end"])

    def test_seasonal_naive_repeats_the_previous_day(self):
        history = np.arange(336, dtype=np.float32)[None, :]
        window = WindowSet(history, np.empty((1, 0), np.float32), np.zeros((1, 3), np.float32),
            np.zeros(1, int), np.zeros(1, int), np.array(["2025-01-08"], dtype="datetime64[ns]"),
            np.zeros((1, 48), np.float32), ())
        self.assertTrue(np.array_equal(seasonal_naive(window)[0], history[0, -48:]))

    def test_zero_run_sensitivity_marks_only_intersecting_horizons(self):
        timestamps = pd.date_range("2025-01-01", periods=100, freq="30min")
        panel = pd.DataFrame({"timestamp": timestamps, "household_id": "House1",
                              "suspect_zero_run": [False]*50 + [True]*50})
        windows = WindowSet(np.zeros((2, 336), np.float32), np.empty((2, 0), np.float32),
            np.zeros((2, 3), np.float32), np.zeros(2, int), np.zeros(2, int),
            np.array([timestamps[0], timestamps[24]], dtype="datetime64[ns]"),
            np.zeros((2, 48), np.float32), ())
        self.assertEqual(zero_run_window_mask(panel, windows).tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
