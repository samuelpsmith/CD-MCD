"""Unit tests for LabVIEW output math calculations."""

import math
import numpy as np
from processing.labview_output_math import calculate_avg_row


def test_calculate_avg_row_r_and_phase():
    x_samples = [1.0, 1.0]
    y_samples = [1.0, 1.0]
    row = calculate_avg_row(350.0, x_samples, y_samples)

    assert row[0] == 350.0
    assert math.isclose(row[1], 1.0, rel_tol=1e-9)
    assert math.isclose(row[2], 1.0, rel_tol=1e-9)
    assert math.isclose(row[3], math.sqrt(2.0), rel_tol=1e-9)
    assert math.isclose(row[4], 45.0, rel_tol=1e-9)
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[7] == 0


def test_calculate_avg_row_accepts_numpy_arrays():
    x_samples = np.array([0.25, 0.75])
    y_samples = np.array([0.50, 1.00])
    row = calculate_avg_row(532.0, x_samples, y_samples)

    assert row[0] == 532.0
    assert math.isclose(row[1], 0.5, rel_tol=1e-9)
    assert math.isclose(row[2], 0.75, rel_tol=1e-9)
    assert math.isclose(row[3], math.hypot(0.5, 0.75), rel_tol=1e-9)
    assert math.isclose(row[4], math.degrees(math.atan2(0.75, 0.5)), rel_tol=1e-9)
    assert row[7] == 0
