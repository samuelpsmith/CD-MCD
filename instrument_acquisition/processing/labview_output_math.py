"""Math utilities for LabVIEW-style CD/MCD output generation.

This module handles the normalization and statistical calculations needed to
match LabVIEW's CD/MCD acquisition output format. The key normalization uses
the SR830 lock-in amplifier sensitivity to scale reference voltage before
dividing X and Y demodulator outputs.

LabVIEW Normalization Formula:
    reference_scaled = reference_aux_voltage × sr830_sensitivity / 10.0
    x_ratio = x_demod / reference_scaled
    y_ratio = y_demod / reference_scaled

Reference channel mapping (LabOne to Python/API):
    LabOne physical "Aux In 1" → Python/API "auxin0" (default)
    LabOne physical "Aux In 2" → Python/API "auxin1" (alternate)
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, Iterable, List, Tuple

import numpy as np


VALID_REFERENCE_CHANNELS = {'auxin0', 'auxin1'}
REFERENCE_CHANNEL_ZERO_THRESHOLD = 1e-12


def validate_reference_channel(channel: str) -> str:
    """Validate that the reference aux channel is allowed."""
    channel_lower = channel.lower()
    if channel_lower not in VALID_REFERENCE_CHANNELS:
        raise ValueError(
            f'Invalid reference_aux_channel: {channel}. Must be one of {VALID_REFERENCE_CHANNELS}.'
        )
    return channel_lower


def normalize_x_y_by_sr830_scaled_reference(
    samples: List[Dict[str, float]],
    reference_channel: str,
    sr830_sensitivity: float,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Normalize X and Y by SR830-scaled reference voltage to match LabVIEW output.

    Formula (LabVIEW equivalent):
        reference_scaled = reference_aux_voltage * sr830_sensitivity / 10.0
        x_ratio = x / reference_scaled
        y_ratio = y / reference_scaled

    Args:
        samples: List of sample dicts from Zurich with 'x', 'y', and reference_channel keys.
        reference_channel: Name of reference channel (e.g., 'auxin0', 'auxin1').
        sr830_sensitivity: SR830 sensitivity in volts (e.g., 0.1, 0.5).
        verbose: If True, print normalization details.

    Returns:
        Tuple of (x_ratio_samples, y_ratio_samples).

    Raises:
        ValueError: If reference_channel is invalid, missing from samples, or reference is too small.
    """
    reference_channel = validate_reference_channel(reference_channel)

    if verbose:
        print(
            f'Normalization: X_ratio = X / (reference_aux * {sr830_sensitivity} / 10.0)\n'
            f'               Y_ratio = Y / (reference_aux * {sr830_sensitivity} / 10.0)\n'
            f'               Reference channel: {reference_channel}\n'
            f'               SR830 sensitivity: {sr830_sensitivity} V'
        )

    x_ratio_samples = []
    y_ratio_samples = []

    for sample_index, sample in enumerate(samples):
        if reference_channel not in sample:
            raise ValueError(
                f'Sample {sample_index} missing {reference_channel} for normalization. '
                f'Available keys: {list(sample.keys())}'
            )

        reference_aux = float(sample[reference_channel])
        reference_scaled = reference_aux * sr830_sensitivity / 10.0

        # Protection against division by zero
        if abs(reference_scaled) < REFERENCE_CHANNEL_ZERO_THRESHOLD:
            raise ValueError(
                f'Sample {sample_index}: scaled reference = {reference_scaled} is too close to zero. '
                f'Cannot normalize (reference_aux={reference_aux}, sr830_sensitivity={sr830_sensitivity}). '
                f'This may indicate a hardware or configuration problem.'
            )

        x = float(sample['x'])
        y = float(sample['y'])
        x_ratio = x / reference_scaled
        y_ratio = y / reference_scaled

        x_ratio_samples.append(x_ratio)
        y_ratio_samples.append(y_ratio)

    return x_ratio_samples, y_ratio_samples


def calculate_avg_row(
    wavelength_nm: float,
    x_samples: Iterable[float],
    y_samples: Iterable[float],
    ddof: int = 1,
) -> List[float]:
    """
    Calculate one average output row for a given wavelength.
    
    Computes mean, standard deviation, magnitude (R), and phase from
    normalized X and Y samples (post-SR830 scaling). Used to populate
    AVG CSV output rows in LabVIEW-compatible format.
    
    Args:
        wavelength_nm: Wavelength in nanometers for this data point.
        x_samples: Normalized X demod samples (post-SR830 scaling).
        y_samples: Normalized Y demod samples (post-SR830 scaling).
        ddof: Delta degrees of freedom for std dev calculation (default 1 for sample std).
    
    Returns:
        List containing: [wavelength, mean_x, mean_y, magnitude_r, phase_deg, std_x, std_y, 0]
    
    Raises:
        ValueError: If samples are empty, mismatched length, or ddof is invalid.
    """
    x_values = np.asarray(x_samples, dtype=float).ravel()
    y_values = np.asarray(y_samples, dtype=float).ravel()

    if x_values.size == 0 or y_values.size == 0:
        raise ValueError('X and Y sample lists must be non-empty')
    if x_values.size != y_values.size:
        raise ValueError('X and Y sample lists must have the same length')

    if ddof < 0:
        raise ValueError('ddof must be non-negative')

    mean_X = statistics.mean(x_values)
    mean_Y = statistics.mean(y_values)

    if len(x_values) == 1:
        std_X = 0.0
    else:
        if ddof >= len(x_values):
            raise ValueError('ddof must be smaller than the number of X samples')
        variance_X = sum((value - mean_X) ** 2 for value in x_values) / (len(x_values) - ddof)
        std_X = math.sqrt(variance_X)

    if len(y_values) == 1:
        std_Y = 0.0
    else:
        if ddof >= len(y_values):
            raise ValueError('ddof must be smaller than the number of Y samples')
        variance_Y = sum((value - mean_Y) ** 2 for value in y_values) / (len(y_values) - ddof)
        std_Y = math.sqrt(variance_Y)

    R = math.hypot(mean_X, mean_Y)
    phase_deg = math.degrees(math.atan2(mean_Y, mean_X)) #NOTE why do we need this if we measure phase?

    return [
        float(wavelength_nm),
        float(mean_X),
        float(mean_Y),
        float(R),
        float(phase_deg),
        float(std_X),
        float(std_Y),
        0,
    ]
