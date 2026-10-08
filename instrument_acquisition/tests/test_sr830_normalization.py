"""Unit tests for SR830-scaled normalization math."""

import math
import pytest
from processing.labview_output_math import (
    normalize_x_y_by_sr830_scaled_reference,
    validate_reference_channel,
    VALID_REFERENCE_CHANNELS,
)


def test_validate_reference_channel():
    """Test reference channel validation."""
    # Valid channels
    assert validate_reference_channel('auxin0') == 'auxin0'
    assert validate_reference_channel('auxin1') == 'auxin1'
    assert validate_reference_channel('AUXIN0') == 'auxin0'
    assert validate_reference_channel('AUXIN1') == 'auxin1'

    # Invalid channels
    with pytest.raises(ValueError, match='Invalid reference_aux_channel'):
        validate_reference_channel('auxin2')
    with pytest.raises(ValueError, match='Invalid reference_aux_channel'):
        validate_reference_channel('invalid')


def test_sr830_normalization_basic():
    """Test LabVIEW-equivalent SR830 normalization with known values.
    
    Test case from requirements:
        x = 2.0
        y = 4.0
        reference_aux_voltage = 5.0
        sr830_sensitivity = 0.1
        
        reference_scaled = 5.0 * 0.1 / 10 = 0.05
        expected x_ratio = 2.0 / 0.05 = 40.0
        expected y_ratio = 4.0 / 0.05 = 80.0
    """
    samples = [
        {'x': 2.0, 'y': 4.0, 'auxin0': 5.0},
        {'x': 2.0, 'y': 4.0, 'auxin0': 5.0},
    ]
    
    x_ratios, y_ratios = normalize_x_y_by_sr830_scaled_reference(
        samples,
        reference_channel='auxin0',
        sr830_sensitivity=0.1,
    )
    
    assert len(x_ratios) == 2
    assert len(y_ratios) == 2
    assert math.isclose(x_ratios[0], 40.0, rel_tol=1e-9)
    assert math.isclose(y_ratios[0], 80.0, rel_tol=1e-9)
    assert math.isclose(x_ratios[1], 40.0, rel_tol=1e-9)
    assert math.isclose(y_ratios[1], 80.0, rel_tol=1e-9)


def test_sr830_normalization_different_sensitivities():
    """Test normalization with different SR830 sensitivity values."""
    samples = [{'x': 1.0, 'y': 2.0, 'auxin0': 10.0}]
    
    # SR830 sensitivity = 0.5 V
    # reference_scaled = 10.0 * 0.5 / 10 = 0.5
    # x_ratio = 1.0 / 0.5 = 2.0
    # y_ratio = 2.0 / 0.5 = 4.0
    x_ratios, y_ratios = normalize_x_y_by_sr830_scaled_reference(
        samples,
        reference_channel='auxin0',
        sr830_sensitivity=0.5,
    )
    assert math.isclose(x_ratios[0], 2.0, rel_tol=1e-9)
    assert math.isclose(y_ratios[0], 4.0, rel_tol=1e-9)


def test_sr830_normalization_auxin1():
    """Test normalization using auxin1 instead of auxin0."""
    samples = [{'x': 2.0, 'y': 4.0, 'auxin1': 5.0}]
    
    x_ratios, y_ratios = normalize_x_y_by_sr830_scaled_reference(
        samples,
        reference_channel='auxin1',
        sr830_sensitivity=0.1,
    )
    
    assert math.isclose(x_ratios[0], 40.0, rel_tol=1e-9)
    assert math.isclose(y_ratios[0], 80.0, rel_tol=1e-9)


def test_sr830_normalization_missing_reference_channel():
    """Test that missing reference channel raises error."""
    samples = [{'x': 1.0, 'y': 2.0}]  # No auxin0 or auxin1
    
    with pytest.raises(ValueError, match='missing auxin0'):
        normalize_x_y_by_sr830_scaled_reference(
            samples,
            reference_channel='auxin0',
            sr830_sensitivity=0.1,
        )


def test_sr830_normalization_zero_reference_protection():
    """Test protection against division by zero."""
    # reference_scaled = 0.001 * 0.1 / 10 = 0.00001, which is above threshold 1e-12
    samples = [{'x': 1.0, 'y': 2.0, 'auxin0': 0.001}]
    
    x_ratios, y_ratios = normalize_x_y_by_sr830_scaled_reference(
        samples,
        reference_channel='auxin0',
        sr830_sensitivity=0.1,
    )
    # Should complete without error
    assert len(x_ratios) == 1


def test_sr830_normalization_extremely_small_reference():
    """Test that extremely small reference values trigger error."""
    # reference_scaled = 1e-15 * 0.1 / 10 = 1e-17, which is below threshold 1e-12
    samples = [{'x': 1.0, 'y': 2.0, 'auxin0': 1e-15}]
    
    with pytest.raises(ValueError, match='too close to zero'):
        normalize_x_y_by_sr830_scaled_reference(
            samples,
            reference_channel='auxin0',
            sr830_sensitivity=0.1,
        )


def test_sr830_normalization_negative_reference():
    """Test normalization with negative reference values (allowed, but produces inverted ratios)."""
    samples = [{'x': 2.0, 'y': 4.0, 'auxin0': -5.0}]
    
    x_ratios, y_ratios = normalize_x_y_by_sr830_scaled_reference(
        samples,
        reference_channel='auxin0',
        sr830_sensitivity=0.1,
    )
    
    # reference_scaled = -5.0 * 0.1 / 10 = -0.05
    # x_ratio = 2.0 / -0.05 = -40.0
    # y_ratio = 4.0 / -0.05 = -80.0
    assert math.isclose(x_ratios[0], -40.0, rel_tol=1e-9)
    assert math.isclose(y_ratios[0], -80.0, rel_tol=1e-9)


def test_sr830_normalization_verbose_output(capsys):
    """Test that verbose mode prints formula."""
    samples = [{'x': 1.0, 'y': 2.0, 'auxin0': 1.0}]
    
    normalize_x_y_by_sr830_scaled_reference(
        samples,
        reference_channel='auxin0',
        sr830_sensitivity=0.5,
        verbose=True,
    )
    
    captured = capsys.readouterr()
    assert 'Normalization:' in captured.out
    assert 'X_ratio' in captured.out
    assert 'Y_ratio' in captured.out
    assert 'reference_aux' in captured.out
    assert 'SR830 sensitivity' in captured.out
