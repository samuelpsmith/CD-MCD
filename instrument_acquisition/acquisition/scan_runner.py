"""Scan orchestration for MCD acquisition.

This module keeps the user-facing script small while preserving the confirmed
hardware behavior and LabVIEW-equivalent normalization math.
"""

from __future__ import annotations

import math
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from acquisition.wavelength_sequence import generate_wavelengths
from instruments.cm110 import CM110
from instruments.pem100 import PEM100
from instruments.zurich_mfli import ZurichMFLI
from output.csv_writer import (
    write_I_avg_csv,
    write_I_sample_csv,
    write_labview_style_csvs,
)
from output.notes_writer import write_notes_file
from processing.labview_output_math import (
    REFERENCE_CHANNEL_ZERO_THRESHOLD,
    calculate_avg_row,
    normalize_x_y_by_sr830_scaled_reference,
    validate_reference_channel,
)


SOFTWARE_VERSION = "mcd-python-acquisition"


@dataclass
class ScanRequest:
    live: bool
    base_name: str
    scan_type: str
    start_nm: float
    end_nm: float
    step_nm: float
    points: int
    sr830_sensitivity_v: float
    settle_time_s: float
    output_dir: str
    record_intensity: bool = False
    operator: str = ""
    config_path: str = "config.yaml"
    warnings: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    wavelengths: List[float]
    files_created: List[str]
    avg_rows: List[List[float]]
    i_avg_rows: List[List[float]]
    warnings: List[str]


def get_git_commit() -> str:
    """Return the current git commit when git is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unavailable"
    return result.stdout.strip() or "unavailable"


def validate_scan_request(request: ScanRequest) -> List[float]:
    """Validate scan inputs and return the generated wavelength list."""
    if request.scan_type.lower() not in {"pos", "neg", "ref"}:
        raise ValueError('Invalid scan type. Use "pos", "neg", or "ref".')
    if not request.base_name.strip():
        raise ValueError("Sample/base name cannot be blank.")
    if request.points <= 0:
        raise ValueError("Points per wavelength must be a positive integer.")
    if request.sr830_sensitivity_v <= 0:
        raise ValueError("Invalid SR830 sensitivity. Enter the sensitivity in volts, greater than zero.")
    if request.settle_time_s < 0:
        raise ValueError("Settle time cannot be negative.")
    if request.start_nm <= 0 or request.end_nm <= 0:
        raise ValueError("Invalid wavelength range. Wavelengths must be positive.")

    try:
        wavelengths = generate_wavelengths(request.start_nm, request.end_nm, request.step_nm)
    except ValueError as exc:
        message = str(exc)
        if "positive for increasing" in message or "negative for decreasing" in message:
            raise ValueError(
                "Invalid step direction. Use a positive step when start < end and a negative step when start > end."
            ) from exc
        raise ValueError(f"Invalid wavelength range: {message}") from exc

    if not wavelengths:
        raise ValueError("Invalid wavelength range. No wavelengths would be collected.")
    return wavelengths


def estimated_runtime_minutes(number_of_wavelengths: int, settle_time_s: float, live: bool) -> float:
    """Estimate minimum scan time including small per-wavelength overhead."""
    overhead_s = 0.25 if live else 0.02
    return number_of_wavelengths * (settle_time_s + overhead_s) / 60.0


def expected_output_files(request: ScanRequest) -> List[str]:
    scan_type = request.scan_type.lower()
    out = Path(request.output_dir)
    files = [
        out / f"{request.base_name}_{scan_type}_X.csv",
        out / f"{request.base_name}_{scan_type}_Y.csv",
        out / f"{request.base_name}_{scan_type}_AVG.csv",
        out / f"{request.base_name}_{scan_type}_notes",
    ]
    if request.record_intensity:
        files.extend(
            [
                out / f"{request.base_name}_{scan_type}_I_sample.csv",
                out / f"{request.base_name}_{scan_type}_I_avg.csv",
            ]
        )
    return [str(path) for path in files]


def build_instruments(config: Dict[str, Any], request: ScanRequest) -> tuple[ZurichMFLI, CM110, PEM100]:
    zurich_config = config.get("zurich", {})
    cm110_config = config.get("cm110", {})
    pem_config = config.get("pem", {})

    zurich = ZurichMFLI(
        device_id=str(zurich_config.get("device_id", "DEV4388")),
        connection=str(zurich_config.get("connection", "ethernet")),
        demod_index=int(zurich_config.get("demod_index", 0)),
        sample_node=str(zurich_config.get("sample_node", "/dev4388/demods/0/sample")),
        reference_aux_channel=str(zurich_config.get("reference_aux_channel", "auxin0")),
        poll_timeout_ms=int(zurich_config.get("poll_timeout_ms", 5)),
        settings=dict(zurich_config.get("settings", {})),
        dry_run=not request.live,
        host=str(zurich_config.get("host", "localhost")),
        port=int(zurich_config.get("port", 8004)),
        api_level=int(zurich_config.get("api_level", 6)),
        interface=str(zurich_config.get("interface", "1GbE")),
    )
    cm110 = CM110(
        port=str(cm110_config.get("port", "COM4")),
        baudrate=int(cm110_config.get("baudrate", 9600)),
        bytesize=int(cm110_config.get("bytesize", 8)),
        parity=str(cm110_config.get("parity", "N")),
        stopbits=int(cm110_config.get("stopbits", 1)),
        timeout_s=float(cm110_config.get("timeout_s", 10.0)),
        dry_run=not request.live,
    )
    pem = PEM100(
        port=str(pem_config.get("port", "COM1")),
        baudrate=int(pem_config.get("baudrate", 2400)),
        bytesize=int(pem_config.get("bytesize", 8)),
        parity=str(pem_config.get("parity", "N")),
        stopbits=int(pem_config.get("stopbits", 1)),
        timeout_s=float(pem_config.get("timeout_s", 2.0)),
        terminator=str(pem_config.get("terminator", "\r")),
        dry_run=not request.live,
    )
    return zurich, cm110, pem


def print_scan_summary(config: Dict[str, Any], request: ScanRequest, wavelengths: List[float]) -> None:
    zurich = config.get("zurich", {})
    cm110 = config.get("cm110", {})
    pem = config.get("pem", {})
    absorption = config.get("absorption", {})
    minimum_minutes = estimated_runtime_minutes(len(wavelengths), request.settle_time_s, request.live)

    print("\n=== MCD scan summary ===")
    print(f"Live scan: {'yes' if request.live else 'no'}")
    print(f"Mode: {'LIVE HARDWARE' if request.live else 'dry-run'}")
    print(f"Sample/base name: {request.base_name}")
    print(f"Scan type: {request.scan_type.lower()}")
    print(f"Wavelength range: {request.start_nm} to {request.end_nm} nm")
    print(f"Step size: {request.step_nm} nm")
    print(f"Number of wavelengths: {len(wavelengths)}")
    print(f"Points per wavelength: {request.points}")
    print(f"SR830 sensitivity: {request.sr830_sensitivity_v} V")
    print(f"Record raw intensity: {'yes' if request.record_intensity else 'no'}")
    print(f"Settle time: {request.settle_time_s} s")
    print(f"Selected output folder: {request.output_dir}")
    print(f"Estimated minimum runtime: {minimum_minutes:.2f} minutes")
    print("")
    print(f"Device ID: {zurich.get('device_id', 'DEV4388')}")
    print(f"Zurich node: {zurich.get('sample_node', '/dev4388/demods/0/sample')}")
    print(f"CM110 port: {cm110.get('port', 'COM4')}")
    print(f"PEM port: {pem.get('port', 'COM1')}")
    print(f"Reference aux channel: {zurich.get('reference_aux_channel', 'auxin0')}")
    if request.record_intensity:
        print(f"Raw intensity channel: {absorption.get('intensity_channel', 'auxin0')}")
    print(f"PEM retardation: {pem.get('retardation_waves', 0.250)} waves")
    print("Output files:")
    for path in expected_output_files(request):
        print(f"  {path}")
    print("========================\n")


def _extract_intensity(samples: List[Dict[str, float]], intensity_channel: str) -> List[float]:
    values = []
    for index, sample in enumerate(samples):
        if intensity_channel not in sample:
            raise ValueError(
                f"Sample {index} missing intensity channel {intensity_channel}. "
                f"Available keys: {list(sample.keys())}"
            )
        values.append(float(sample[intensity_channel]))
    return values


def _i_avg_row(i_samples: List[float], ddof: int) -> List[float]:
    mean_i = statistics.mean(i_samples)
    if len(i_samples) == 1:
        std_i = 0.0
    else:
        std_i = statistics.stdev(i_samples) if ddof == 1 else statistics.pstdev(i_samples)
    return [float(mean_i), float(std_i)]


def _warn_for_nan(values: List[float], label: str, warnings: List[str], wavelength: float) -> None:
    if any(math.isnan(value) for value in values):
        warnings.append(f"NaN detected in {label} at {wavelength} nm.")


def run_scan(config: Dict[str, Any], request: ScanRequest) -> ScanResult:
    """Run a live or dry-run MCD scan and write output files."""
    wavelengths = validate_scan_request(request)
    scan_type = request.scan_type.lower()
    zurich_config = config.get("zurich", {})
    pem_config = config.get("pem", {})
    absorption_config = config.get("absorption", {})
    scan_defaults = config.get("scan_defaults", {})
    reference_channel = validate_reference_channel(str(zurich_config.get("reference_aux_channel", "auxin0")))
    intensity_channel = str(absorption_config.get("intensity_channel", "auxin0"))
    ddof = int(scan_defaults.get("standard_deviation_ddof", 1))
    retardation = float(pem_config.get("retardation_waves", 0.250))
    warnings = list(request.warnings)

    zurich, cm110, pem = build_instruments(config, request)
    x_rows: List[List[float]] = []
    y_rows: List[List[float]] = []
    avg_rows: List[List[float]] = []
    i_rows: List[List[float]] = []
    i_avg_rows: List[List[float]] = []

    try:
        zurich.open()
        cm110.open()
        pem.open()
        pem.set_retardation(retardation)

        for wavelength in wavelengths:
            cm110.set_wavelength(int(round(wavelength)))
            pem.set_wavelength(wavelength)
            if request.live and request.settle_time_s > 0:
                time.sleep(request.settle_time_s)

            samples = zurich.poll_samples(request.points)
            min_ref = min(abs(float(sample[reference_channel])) for sample in samples)
            min_scaled_ref = min_ref * request.sr830_sensitivity_v / 10.0
            if min_scaled_ref < REFERENCE_CHANNEL_ZERO_THRESHOLD * 100:
                warnings.append(
                    f"Near-zero reference warning at {wavelength} nm: minimum scaled reference {min_scaled_ref:g}."
                )

            x_samples, y_samples = normalize_x_y_by_sr830_scaled_reference(
                samples,
                reference_channel=reference_channel,
                sr830_sensitivity=request.sr830_sensitivity_v,
                verbose=False,
            )
            _warn_for_nan(x_samples, "X_ratio", warnings, wavelength)
            _warn_for_nan(y_samples, "Y_ratio", warnings, wavelength)
            x_rows.append(x_samples)
            y_rows.append(y_samples)
            avg_rows.append(calculate_avg_row(wavelength, x_samples, y_samples, ddof=ddof))

            if request.record_intensity:
                intensity_samples = _extract_intensity(samples, intensity_channel)
                _warn_for_nan(intensity_samples, "I_sample", warnings, wavelength)
                i_rows.append(intensity_samples)
                i_avg_rows.append(_i_avg_row(intensity_samples, ddof=ddof))
    finally:
        try:
            pem.close()
        finally:
            try:
                cm110.close()
            finally:
                zurich.close()

    x_path, y_path, avg_path = write_labview_style_csvs(
        base_name=request.base_name,
        scan_type=scan_type,
        output_dir=request.output_dir,
        wavelengths=wavelengths,
        x_rows=x_rows,
        y_rows=y_rows,
        avg_rows=avg_rows,
    )
    files_created = [x_path, y_path, avg_path]

    if request.record_intensity:
        files_created.append(
            write_I_sample_csv(request.base_name, scan_type, request.output_dir, wavelengths, i_rows)
        )
        files_created.append(
            write_I_avg_csv(request.base_name, scan_type, request.output_dir, wavelengths, i_avg_rows)
        )

    metadata = {
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "operator": request.operator or "not provided",
        "base_name": request.base_name,
        "scan_type": scan_type,
        "wavelength_range_nm": f"{request.start_nm} to {request.end_nm}",
        "step_size_nm": request.step_nm,
        "points_per_wavelength": request.points,
        "settle_time_s": request.settle_time_s,
        "pem_retardation_waves": retardation,
        "sr830_sensitivity_v": request.sr830_sensitivity_v,
        "reference_aux_channel": reference_channel,
        "intensity_recording_enabled": request.record_intensity,
        "intensity_channel": intensity_channel if request.record_intensity else "not recorded",
        "zurich_node": zurich_config.get("sample_node", "/dev4388/demods/0/sample"),
        "zurich_device_id": zurich_config.get("device_id", "DEV4388"),
        "cm110_port": config.get("cm110", {}).get("port", "COM4"),
        "pem_port": config.get("pem", {}).get("port", "COM1"),
        "labone_server": f"{zurich_config.get('host', 'localhost')}:{zurich_config.get('port', 8004)}",
        "labone_api_level": zurich_config.get("api_level", 6),
        "software_version": SOFTWARE_VERSION,
        "git_commit": get_git_commit(),
        "warnings": "; ".join(warnings) if warnings else "none",
    }
    notes_path = write_notes_file(
        base_name=request.base_name,
        scan_type=scan_type,
        output_dir=request.output_dir,
        b_field_t=float(scan_defaults.get("b_field_t", 1.0)),
        path_length_m=float(scan_defaults.get("path_length_m", 0.002)),
        experiment_note=str(scan_defaults.get("experiment_note", "")),
        metadata=metadata,
    )
    files_created.append(notes_path)

    return ScanResult(
        wavelengths=wavelengths,
        files_created=files_created,
        avg_rows=avg_rows,
        i_avg_rows=i_avg_rows,
        warnings=warnings,
    )


def print_post_scan_summary(result: ScanResult, intensity_enabled: bool) -> None:
    mean_x = [row[1] for row in result.avg_rows]
    mean_y = [row[2] for row in result.avg_rows]
    print("\n=== Post-scan validation ===")
    print(f"Wavelengths collected: {len(result.wavelengths)}")
    print("Files created:")
    for path in result.files_created:
        print(f"  {path}")
    print(f"Mean X_ratio range: {min(mean_x):.8g} to {max(mean_x):.8g}")
    print(f"Mean Y_ratio range: {min(mean_y):.8g} to {max(mean_y):.8g}")
    if intensity_enabled and result.i_avg_rows:
        mean_i = [row[0] for row in result.i_avg_rows]
        print(f"Mean I_sample range: {min(mean_i):.8g} to {max(mean_i):.8g}")
    nan_warnings = [warning for warning in result.warnings if "NaN" in warning]
    ref_warnings = [warning for warning in result.warnings if "reference" in warning.lower()]
    print(f"NaN warnings: {len(nan_warnings)}")
    print(f"Near-zero reference warnings: {len(ref_warnings)}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  {warning}")
    print("MCD scan complete.")
