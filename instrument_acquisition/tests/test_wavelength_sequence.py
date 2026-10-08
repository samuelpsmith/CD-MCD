"""Unit tests for wavelength sequence generation."""

from acquisition.wavelength_sequence import generate_wavelengths
import pytest


def test_generate_wavelengths_includes_end_point():
    wavelengths = generate_wavelengths(350.0, 750.0, 1.0)
    assert len(wavelengths) == 401
    assert wavelengths[0] == 350.0
    assert wavelengths[-1] == 750.0


def test_generate_wavelengths_small_sequence():
    assert generate_wavelengths(350.0, 355.0, 1.0) == [350.0, 351.0, 352.0, 353.0, 354.0, 355.0]


def test_generate_wavelengths_zero_step_raises():
    with pytest.raises(ValueError):
        generate_wavelengths(350.0, 355.0, 0.0)
