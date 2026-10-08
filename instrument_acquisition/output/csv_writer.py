"""CSV writer for LabVIEW-style acquisition output files."""

from __future__ import annotations

import csv
import os
from typing import Sequence, Tuple


def _validate_scan_type(scan_type: str) -> str:
    normalized = scan_type.lower()
    if normalized not in {'pos', 'neg', 'ref'}:
        raise ValueError('scan_type must be "pos", "neg", or "ref"')
    return normalized


def write_labview_style_csvs(
    base_name: str,
    scan_type: str,
    output_dir: str,
    wavelengths: Sequence[float],
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[Sequence[float]],
    avg_rows: Sequence[Sequence[float]],
) -> Tuple[str, str, str]:
    """Write X, Y and AVG CSV files in LabVIEW-compatible format."""
    scan_type = _validate_scan_type(scan_type)
    os.makedirs(output_dir, exist_ok=True)

    x_path = os.path.join(output_dir, f'{base_name}_{scan_type}_X.csv')
    y_path = os.path.join(output_dir, f'{base_name}_{scan_type}_Y.csv')
    avg_path = os.path.join(output_dir, f'{base_name}_{scan_type}_AVG.csv')

    with open(x_path, 'w', newline='', encoding='utf-8') as x_file:
        writer = csv.writer(x_file, delimiter=',')
        for wavelength, samples in zip(wavelengths, x_rows):
            writer.writerow([wavelength, *samples])

    with open(y_path, 'w', newline='', encoding='utf-8') as y_file:
        writer = csv.writer(y_file, delimiter=',')
        for wavelength, samples in zip(wavelengths, y_rows):
            writer.writerow([wavelength, *samples])

    with open(avg_path, 'w', newline='', encoding='utf-8') as avg_file:
        writer = csv.writer(avg_file, delimiter=',')
        for row in avg_rows:
            writer.writerow(row)

    return x_path, y_path, avg_path


def write_aux_channel_csv(
    base_name: str,
    scan_type: str,
    output_dir: str,
    wavelengths: Sequence[float],
    aux_rows: Sequence[Sequence[float]],
    aux_channel: str = 'auxin0',
) -> str:
    """Write auxiliary channel data in one CSV file per scan."""
    scan_type = _validate_scan_type(scan_type)
    os.makedirs(output_dir, exist_ok=True)
    aux_path = os.path.join(output_dir, f'{base_name}_{scan_type}_{aux_channel}.csv')

    with open(aux_path, 'w', newline='', encoding='utf-8') as aux_file:
        writer = csv.writer(aux_file, delimiter=',')
        for wavelength, samples in zip(wavelengths, aux_rows):
            writer.writerow([wavelength, *samples])

    return aux_path


def write_I_sample_csv(
    base_name: str,
    scan_type: str,
    output_dir: str,
    wavelengths: Sequence[float],
    i_rows: Sequence[Sequence[float]],
) -> str:
    """Write raw detector intensity (I_sample) CSV named <base>_<scan_type>_I_sample.csv.

    Format matches X/Y CSVs: first column wavelength, then one column per sample.
    """
    scan_type = _validate_scan_type(scan_type)
    os.makedirs(output_dir, exist_ok=True)
    i_path = os.path.join(output_dir, f'{base_name}_{scan_type}_I_sample.csv')

    with open(i_path, 'w', newline='', encoding='utf-8') as i_file:
        writer = csv.writer(i_file, delimiter=',')
        for wavelength, samples in zip(wavelengths, i_rows):
            writer.writerow([wavelength, *samples])

    return i_path


def write_I_avg_csv(
    base_name: str,
    scan_type: str,
    output_dir: str,
    wavelengths: Sequence[float],
    i_avg_rows: Sequence[Sequence[float]],
) -> str:
    """Write average intensity CSV named <base>_<scan_type>_I_avg.csv.

    Each row is written as: wavelength, mean_I_sample, std_I_sample
    """
    scan_type = _validate_scan_type(scan_type)
    os.makedirs(output_dir, exist_ok=True)
    i_avg_path = os.path.join(output_dir, f'{base_name}_{scan_type}_I_avg.csv')

    with open(i_avg_path, 'w', newline='', encoding='utf-8') as i_avg_file:
        writer = csv.writer(i_avg_file, delimiter=',')
        for wavelength, values in zip(wavelengths, i_avg_rows):
            writer.writerow([wavelength, *values])

    return i_avg_path
