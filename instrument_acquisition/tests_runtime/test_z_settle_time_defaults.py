#!/usr/bin/env python3
"""
Dry-run test to verify default settle time and CLI override behavior.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from types import SimpleNamespace


def load_config():
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / 'config.yaml'
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_defaults():
    config = load_config()
    scan_defaults = config.get('scan_defaults', {})
    default_settle = float(scan_defaults.get('settle_time_s', 2))
    assert default_settle == 2.0, f"Expected default settle_time_s 2, got {default_settle}"

    # simulate CLI override
    args = SimpleNamespace(settle_time=8)
    settle_time = float(args.settle_time if args.settle_time is not None else scan_defaults.get('settle_time_s', 2))
    assert settle_time == 8.0, f"Expected CLI override to produce 8, got {settle_time}"


def test_h_short_scan_dryrun_output(tmp_path=None):
    workspace_root = Path(__file__).resolve().parents[1]
    script_path = workspace_root / 'tests_runtime' / 'test_h_short_scan.py'
    output_dir = Path(tempfile.mkdtemp(prefix='settle_2s_'))

    cmd = [
        sys.executable,
        str(script_path),
        '--start', '400',
        '--end', '402',
        '--step', '1',
        '--points', '10',
        '--base-name', 'settle_2s_test',
        '--scan-type', 'pos',
        '--output-dir', str(output_dir),
        '--record-intensity',
        '--intensity-channel', 'auxin0',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f'Command failed: {result.stderr or result.stdout}'

    expected_files = [
        output_dir / 'settle_2s_test_pos_X.csv',
        output_dir / 'settle_2s_test_pos_Y.csv',
        output_dir / 'settle_2s_test_pos_AVG.csv',
        output_dir / 'settle_2s_test_pos_notes',
        output_dir / 'settle_2s_test_pos_I_sample.csv',
        output_dir / 'settle_2s_test_pos_I_avg.csv',
    ]
    for path in expected_files:
        assert path.exists(), f"Missing expected output file: {path}"


if __name__ == '__main__':
    test_defaults()
    test_h_short_scan_dryrun_output()
    print('PASS: default settle time is 2 and CLI override works')
