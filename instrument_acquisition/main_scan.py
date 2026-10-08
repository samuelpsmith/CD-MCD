"""Offline acquisition simulator for LabVIEW-style CD/MCD output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from acquisition.config import load_config, get_scan_defaults
from acquisition.wavelength_sequence import generate_wavelengths
from instruments.cm110 import CM110
from instruments.pem100 import PEM100
from instruments.zurich_mfli import ZurichMFLI
from output.csv_writer import write_labview_style_csvs
from output.notes_writer import write_notes_file
from processing.labview_output_math import calculate_avg_row, normalize_x_y_by_sr830_scaled_reference


def _generate_fake_samples(
    wavelength_nm: float,
    number_of_points: int,
    zurich: ZurichMFLI,
    reference_channel: str,
    sr830_sensitivity: float,
) -> Tuple[List[float], List[float]]:
    """Generate fake X and Y sample arrays for one wavelength using Zurich simulation, normalized by SR830-scaled reference."""
    samples = []

    for _ in range(number_of_points):
        sample = zurich.get_fake_sample(wavelength_nm)
        samples.append(sample)

    x_ratio_samples, y_ratio_samples = normalize_x_y_by_sr830_scaled_reference(
        samples,
        reference_channel=reference_channel,
        sr830_sensitivity=sr830_sensitivity,
        verbose=False,
    )

    return x_ratio_samples, y_ratio_samples


def run_offline_simulation() -> None:
    """Simulate an offline scan and write LabVIEW-style output files."""
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    config = load_config(config_path)
    scan_defaults = get_scan_defaults(config)

    base_name = str(scan_defaults.get('base_name', 'test'))
    scan_type = str(scan_defaults.get('scan_type', 'pos'))
    start_nm = float(scan_defaults.get('start_nm', 350.0))
    end_nm = float(scan_defaults.get('end_nm', 355.0))
    step_nm = float(scan_defaults.get('step_nm', 1.0))
    number_of_points = int(scan_defaults.get('number_of_points', 100))
    output_dir = str(scan_defaults.get('output_dir', 'data'))
    b_field_t = float(scan_defaults.get('b_field_t', 1.0))
    path_length_m = float(scan_defaults.get('path_length_m', 0.002))
    experiment_note = str(scan_defaults.get('experiment_note', 'Offline simulation test'))
    ddof = int(scan_defaults.get('standard_deviation_ddof', 1))
    retardation_waves = float(config.get('pem', {}).get('retardation_waves', 0.250))
    sr830_sensitivity = float(scan_defaults.get('sr830_sensitivity_v', 0.5))
    reference_channel = str(scan_defaults.get('reference_aux_channel', 'auxin0'))

    if number_of_points <= 0:
        raise ValueError('number_of_points must be positive')

    cm110_config = config.get('cm110', {})
    pem_config = config.get('pem', {})
    zurich_config = config.get('zurich', {})

    cm110 = CM110(
        port=str(cm110_config.get('port', 'COM4')),
        baudrate=int(cm110_config.get('baudrate', 9600)),
        bytesize=int(cm110_config.get('bytesize', 8)),
        parity=str(cm110_config.get('parity', 'N')),
        stopbits=int(cm110_config.get('stopbits', 1)),
        timeout_s=float(cm110_config.get('timeout_s', 10.0)),
        dry_run=True,
    )

    pem = PEM100(
        port=str(pem_config.get('port', 'COM1')),
        baudrate=int(pem_config.get('baudrate', 2400)),
        bytesize=int(pem_config.get('bytesize', 8)),
        parity=str(pem_config.get('parity', 'N')),
        stopbits=int(pem_config.get('stopbits', 1)),
        timeout_s=float(pem_config.get('timeout_s', 2.0)),
        terminator=str(pem_config.get('terminator', '\r')),
        dry_run=True,
    )

    zurich = ZurichMFLI(
        device_id=str(zurich_config.get('device_id', 'DEV4388')),
        connection=str(zurich_config.get('connection', 'ethernet')),
        demod_index=int(zurich_config.get('demod_index', 0)),
        sample_node=str(zurich_config.get('sample_node', '/dev4388/demods/0/sample')),
        reference_aux_channel=str(zurich_config.get('reference_aux_channel', 'auxin1')),
        poll_timeout_ms=int(zurich_config.get('poll_timeout_ms', 5)),
        settings=dict(zurich_config.get('settings', {})),
        dry_run=True,
    )

    try:
        zurich.open()
        cm110.set_wavelength(int(round(start_nm)))
        pem.set_wavelength(start_nm)
        pem.set_retardation(retardation_waves)

        wavelengths = generate_wavelengths(start_nm, end_nm, step_nm)
        x_rows: List[List[float]] = []
        y_rows: List[List[float]] = []
        avg_rows: List[List[float]] = []

        for wavelength in wavelengths:
            x_samples, y_samples = _generate_fake_samples(wavelength, number_of_points, zurich, reference_channel, sr830_sensitivity)
            x_rows.append(x_samples)
            y_rows.append(y_samples)
            avg_rows.append(calculate_avg_row(wavelength, x_samples, y_samples, ddof=ddof))

        x_path, y_path, avg_path = write_labview_style_csvs(
            base_name=base_name,
            scan_type=scan_type,
            output_dir=output_dir,
            wavelengths=wavelengths,
            x_rows=x_rows,
            y_rows=y_rows,
            avg_rows=avg_rows,
        )

        notes_path = write_notes_file(
            base_name=base_name,
            scan_type=scan_type,
            output_dir=output_dir,
            b_field_t=b_field_t,
            path_length_m=path_length_m,
            experiment_note=experiment_note,
        )

        print('Created files:')
        print(f'  {x_path}')
        print(f'  {y_path}')
        print(f'  {avg_path}')
        print(f'  {notes_path}')
    finally:
        zurich.close()


if __name__ == '__main__':
    run_offline_simulation()
