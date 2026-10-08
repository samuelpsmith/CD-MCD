"""Wavelength sequence utilities for offline acquisition simulation."""

from __future__ import annotations

from typing import List


def generate_wavelengths(start_nm: float, end_nm: float, step_nm: float) -> List[float]:
    """Generate an inclusive wavelength scan sequence.

    The ending wavelength is included when it lies exactly on the step sequence
    within floating point tolerance.
    """
    if step_nm == 0:
        raise ValueError('step_nm cannot be zero')

    if start_nm == end_nm:
        return [start_nm]

    increasing = end_nm > start_nm
    if increasing and step_nm < 0:
        raise ValueError('step_nm must be positive for increasing scans')
    if not increasing and step_nm > 0:
        raise ValueError('step_nm must be negative for decreasing scans')

    tolerance = abs(step_nm) * 1e-9 + 1e-12
    wavelengths: List[float] = []
    index = 0

    while True:
        current = start_nm + step_nm * index
        if increasing:
            if current > end_nm + tolerance:
                break
        else:
            if current < end_nm - tolerance:
                break

        wavelengths.append(round(current, 12))
        index += 1

    if wavelengths and abs(wavelengths[-1] - end_nm) <= tolerance:
        wavelengths[-1] = end_nm
    return wavelengths
