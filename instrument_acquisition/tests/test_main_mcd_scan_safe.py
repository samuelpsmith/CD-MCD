"""Safe tests for the user-facing MCD acquisition path."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import main_mcd_scan
from acquisition.config import load_config
from acquisition.scan_runner import ScanRequest, expected_output_files, validate_scan_request
from instruments.cm110 import CM110
from instruments.pem100 import PEM100


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_config_defaults_are_confirmed_values():
    config = load_config(ROOT / "config.yaml")
    assert config["zurich"]["device_id"] == "DEV4388"
    assert config["zurich"]["sample_node"] == "/dev4388/demods/0/sample"
    assert config["zurich"]["reference_aux_channel"] == "auxin0"
    assert config["absorption"]["intensity_channel"] == "auxin0"
    assert config["cm110"]["port"] == "COM4"
    assert config["pem"]["port"] == "COM1"
    assert float(config["scan_defaults"]["settle_time_s"]) == 2.0
    assert int(config["scan_defaults"]["standard_deviation_ddof"]) == 1


def test_cm110_goto_byte_generation():
    assert CM110().build_goto_command(400) == bytes([16, 1, 144])


def test_pem_command_formatting():
    pem = PEM100(terminator="\r")
    assert pem.build_wavelength_command(400.0) == "W:004000\r"
    assert pem.build_retardation_command(0.250) == "R:0250\r"


def test_expected_output_files_with_intensity(tmp_path):
    request = ScanRequest(
        live=False,
        base_name="sample",
        scan_type="pos",
        start_nm=400,
        end_nm=402,
        step_nm=1,
        points=3,
        sr830_sensitivity_v=0.1,
        settle_time_s=2,
        output_dir=str(tmp_path),
        record_intensity=True,
    )
    names = [Path(path).name for path in expected_output_files(request)]
    assert names == [
        "sample_pos_X.csv",
        "sample_pos_Y.csv",
        "sample_pos_AVG.csv",
        "sample_pos_notes",
        "sample_pos_I_sample.csv",
        "sample_pos_I_avg.csv",
    ]


def test_expected_output_files_accept_ref_scan_type(tmp_path):
    request = ScanRequest(
        live=False,
        base_name="blank",
        scan_type="ref",
        start_nm=400,
        end_nm=402,
        step_nm=1,
        points=3,
        sr830_sensitivity_v=0.5,
        settle_time_s=2,
        output_dir=str(tmp_path),
        record_intensity=True,
    )
    names = [Path(path).name for path in expected_output_files(request)]
    assert names == [
        "blank_ref_X.csv",
        "blank_ref_Y.csv",
        "blank_ref_AVG.csv",
        "blank_ref_notes",
        "blank_ref_I_sample.csv",
        "blank_ref_I_avg.csv",
    ]


def test_invalid_step_direction_message():
    request = ScanRequest(
        live=False,
        base_name="sample",
        scan_type="pos",
        start_nm=400,
        end_nm=450,
        step_nm=-1,
        points=3,
        sr830_sensitivity_v=0.1,
        settle_time_s=2,
        output_dir="data",
    )
    try:
        validate_scan_request(request)
    except ValueError as exc:
        assert "Invalid step direction" in str(exc)
    else:
        raise AssertionError("Expected invalid step direction to raise")


def test_gui_values_default_to_live_and_build_scan_request(tmp_path):
    config = load_config(ROOT / "config.yaml")
    request = main_mcd_scan.gui_values_to_request(
        {
            "base_name": "GuiSample",
            "scan_type": "ref",
            "start_nm": "350",
            "end_nm": "750",
            "step_nm": "1",
            "points": "100",
            "sr830_sensitivity_v": "0.1",
            "record_intensity": True,
            "settle_time_s": "2",
            "output_dir": str(tmp_path),
        },
        config,
        "config.yaml",
    )

    assert request.live is True
    assert request.operator == ""
    assert request.base_name == "GuiSample"
    assert request.scan_type == "ref"
    assert request.start_nm == 350
    assert request.end_nm == 750
    assert request.step_nm == 1
    assert request.points == 100
    assert request.sr830_sensitivity_v == 0.1
    assert request.record_intensity is True
    assert request.settle_time_s == 2
    assert request.output_dir == str(tmp_path)


def test_gui_blank_values_use_config_defaults(tmp_path):
    config = load_config(ROOT / "config.yaml")
    request = main_mcd_scan.gui_values_to_request(
        {
            "base_name": "",
            "scan_type": "",
            "start_nm": "",
            "end_nm": "",
            "step_nm": "",
            "points": "",
            "sr830_sensitivity_v": "",
            "record_intensity": False,
            "settle_time_s": "",
            "output_dir": str(tmp_path),
        },
        config,
        "config.yaml",
    )

    assert request.live is True
    assert request.base_name == "test"
    assert request.scan_type == "pos"
    assert request.start_nm == 400
    assert request.end_nm == 450
    assert request.points == 100
    assert request.sr830_sensitivity_v == 0.1
    assert request.record_intensity is False
    assert request.output_dir == str(tmp_path)


def test_gui_blank_values_default_sr830_and_settle_are_accepted(tmp_path):
    config = load_config(ROOT / "config.yaml")
    request = main_mcd_scan.gui_values_to_request(
        {
            "base_name": "defaults",
            "scan_type": "ref",
            "start_nm": "400",
            "end_nm": "402",
            "step_nm": "2",
            "points": "5",
            "sr830_sensitivity_v": "0.5",
            "record_intensity": False,
            "settle_time_s": "2",
            "output_dir": str(tmp_path),
        },
        config,
        "config.yaml",
    )
    assert request.sr830_sensitivity_v == 0.5
    assert request.settle_time_s == 2
    assert validate_scan_request(request) == [400.0, 402.0]


def test_main_mcd_scan_dry_run_outputs_and_cli_overrides(tmp_path):
    cmd = [
        sys.executable,
        str(ROOT / "main_mcd_scan.py"),
        "--start",
        "400",
        "--end",
        "402",
        "--step",
        "1",
        "--points",
        "3",
        "--base-name",
        "safe_dryrun",
        "--scan-type",
        "pos",
        "--sr830-sensitivity",
        "0.1",
        "--record-intensity",
        "--output-dir",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout

    x_path = tmp_path / "safe_dryrun_pos_X.csv"
    y_path = tmp_path / "safe_dryrun_pos_Y.csv"
    avg_path = tmp_path / "safe_dryrun_pos_AVG.csv"
    notes_path = tmp_path / "safe_dryrun_pos_notes"
    i_sample_path = tmp_path / "safe_dryrun_pos_I_sample.csv"
    i_avg_path = tmp_path / "safe_dryrun_pos_I_avg.csv"

    for path in [x_path, y_path, avg_path, notes_path, i_sample_path, i_avg_path]:
        assert path.exists(), f"Missing expected file {path}"

    x_rows = read_rows(x_path)
    y_rows = read_rows(y_path)
    avg_rows = read_rows(avg_path)
    i_sample_rows = read_rows(i_sample_path)
    i_avg_rows = read_rows(i_avg_path)

    assert len(x_rows) == 3
    assert len(y_rows) == 3
    assert len(avg_rows) == 3
    assert len(i_sample_rows) == 3
    assert len(i_avg_rows) == 3
    assert len(x_rows[0]) == 4
    assert len(y_rows[0]) == 4
    assert len(i_sample_rows[0]) == 4
    assert len(avg_rows[0]) == 8
    assert len(i_avg_rows[0]) == 3
    assert "sr830_sensitivity_v: 0.1" in notes_path.read_text(encoding="utf-8")
    assert "operator: not provided" in notes_path.read_text(encoding="utf-8")
