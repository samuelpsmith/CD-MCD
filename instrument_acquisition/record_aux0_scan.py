#!/usr/bin/env python3

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.config import get_scan_defaults, load_config, resolve_parameter
from acquisition.wavelength_sequence import generate_wavelengths
from instruments.cm110 import CM110
from instruments.pem100 import PEM100
from instruments.zurich_mfli import ZurichMFLI
from output.csv_writer import write_aux_channel_csv

CONFIRMATION = 'I UNDERSTAND THIS WILL RECORD AUX0 VALUES'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Record Zurich auxiliary channel values over a wavelength scan.')
    parser.add_argument('--live', action='store_true', help='Run in live hardware mode.')
    parser.add_argument('--start', type=float, default=450.0, help='Start wavelength in nm.')
    parser.add_argument('--end', type=float, default=454.0, help='End wavelength in nm.')
    parser.add_argument('--step', type=float, default=1.0, help='Wavelength step in nm.')
    parser.add_argument('--points', type=int, default=100, help='Number of Zurich samples to collect per wavelength.')
    parser.add_argument('--base-name', default='aux0_scan', help='Output base file name.')
    parser.add_argument('--scan-type', default='pos', help='Scan type for output file naming: pos or neg.')
    parser.add_argument('--output-dir', default='data', help='Directory for the output CSV file.')
    parser.add_argument('--aux-channel', default='auxin0', choices=['auxin0', 'auxin1'], help='Auxiliary channel to record.')
    parser.add_argument('--reference-aux-channel', default=None, help='Zurich reference aux channel for normalization (auxin0 or auxin1).')
    parser.add_argument('--sr830-sensitivity', type=float, default=None, help='SR830 sensitivity in volts used by the main acquisition path.')
    parser.add_argument('--settle-time', type=float, default=None, help='Seconds to wait after moving the monochromator before recording.')
    parser.add_argument('--node', default='/dev4388/demods/0/sample', help='Zurich sample node.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    scan_defaults = get_scan_defaults(config)

    settle_time = resolve_parameter(args.settle_time, scan_defaults, 'settle_time_s', 2.0, float, 'settle_time')
    reference_aux_channel = resolve_parameter(args.reference_aux_channel, scan_defaults, 'reference_aux_channel', 'auxin0', str, 'reference_aux_channel')
    sr830_sensitivity = resolve_parameter(args.sr830_sensitivity, scan_defaults, 'sr830_sensitivity_v', 0.5, float, 'sr830_sensitivity')

    if args.live:
        answer = input(f'Type exactly: {CONFIRMATION}\n').strip()
        if answer != CONFIRMATION:
            print('Confirmation does not match. Aborting.')
            return 1

    dry_run = not args.live
    cm110 = CM110(dry_run=dry_run)
    pem = PEM100(dry_run=dry_run)
    zurich = ZurichMFLI(
        sample_node=args.node,
        reference_aux_channel=reference_aux_channel,
        poll_timeout_ms=int(config.get('zurich', {}).get('poll_timeout_ms', 5)),
        dry_run=dry_run,
    )

    try:
        if args.live:
            cm110.open()
            pem.open()
            zurich.open()

        wavelengths = generate_wavelengths(args.start, args.end, args.step)
        aux_rows: List[List[float]] = []

        for wavelength in wavelengths:
            cm110.set_wavelength(int(round(wavelength)))
            pem.set_wavelength(wavelength)

            if args.live:
                print(f'Waiting {settle_time} seconds at wavelength {wavelength} nm...')
                time.sleep(settle_time)

            samples = zurich.poll_samples(args.points)
            aux_values = [float(sample[args.aux_channel]) for sample in samples]
            aux_rows.append(aux_values)

            mean_aux = sum(aux_values) / len(aux_values)
            print(f'Wavelength {wavelength} nm: {args.aux_channel} mean={mean_aux:.6f}')

        aux_path = write_aux_channel_csv(
            base_name=args.base_name,
            scan_type=args.scan_type,
            output_dir=args.output_dir,
            wavelengths=wavelengths,
            aux_rows=aux_rows,
            aux_channel=args.aux_channel,
        )

        print('Recorded auxiliary data to:')
        print(f'  {aux_path}')
        print('PASS: Aux channel scan completed.')
        return 0
    except Exception as exc:
        print('FAIL:', exc)
        return 1
    finally:
        if args.live:
            cm110.close()
            pem.close()
            zurich.close()


if __name__ == '__main__':
    raise SystemExit(main())
