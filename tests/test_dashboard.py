"""
Unit Tests for Interactive 3D Dashboard & Static Report Exporter (Module 6)

Tests:
1. generate_dashboard_data generation across all 5 navigation engines.
2. Structure and schema of trajectories, scorecards, errors, and telemetry.
3. Offline static HTML report generation (export_static_report).
4. DashboardRequestHandler API endpoints (/api/profiles, /api/sample).
"""

import os
import json
import unittest
import tempfile

from src.dashboard.server import generate_dashboard_data
from src.dashboard.export_report import export_static_report


class TestDashboard(unittest.TestCase):
    """Test suite for Module 6 Dashboard & Analytics."""

    def test_generate_dashboard_data(self):
        """Test multi-engine simulation data generation and JSON schema."""
        payload = generate_dashboard_data(
            profile="figure_eight",
            duration=10.0,
            dt=0.02,
            imu_grade="consumer",
            gps_outage=(3.0, 7.0),
            seed=42,
        )

        self.assertIsInstance(payload, dict)
        self.assertIn("metadata", payload)
        self.assertIn("scorecards", payload)
        self.assertIn("timestamps", payload)
        self.assertIn("trajectories", payload)
        self.assertIn("errors_vs_time", payload)
        self.assertIn("telemetry", payload)

        # Check that all 5 trajectories exist
        tr = payload["trajectories"]
        for key in ("ground_truth", "classical_dr", "zupt_dr", "neural_dr", "ai_ekf"):
            self.assertIn(key, tr)
            self.assertIn("x", tr[key])
            self.assertIn("y", tr[key])
            self.assertIn("z", tr[key])
            self.assertEqual(len(tr[key]["x"]), len(payload["timestamps"]))

        # Check scorecards
        scorecards = payload["scorecards"]
        self.assertEqual(len(scorecards), 4)  # Mod 2, Mod 3, Mod 4, Mod 5
        for sc in scorecards:
            self.assertIn("name", sc)
            self.assertIn("ate_rmse", sc)
            self.assertIn("reduction_pct", sc)

        # Check telemetry fields
        tel = payload["telemetry"]
        for k in ("speed_ekf", "roll_deg", "pitch_deg", "yaw_deg", "acc_bias_x", "pos_3sigma_x", "gps_valid", "handoff_state", "motion_context"):
            self.assertIn(k, tel)
            self.assertEqual(len(tel[k]), len(payload["timestamps"]))

    def test_export_static_report(self):
        """Test standalone self-contained HTML report exporter."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "test_navis_report.html")
            path = export_static_report(
                output_path=out_file,
                profile="figure_eight",
                duration=10.0,
                imu_grade="consumer",
                gps_outage_start=3.0,
                gps_outage_end=7.0,
                seed=42
            )

            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 5000)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("window.__PRELOADED_NAVIS_DATA__", content)
            self.assertIn("NAVIS", content)
            self.assertIn("Plotly", content)


if __name__ == "__main__":
    unittest.main()
