#!/usr/bin/env python3

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from acquisition.config import load_config, get_scan_defaults, resolve_parameter, print_scan_config
from acquisition.wavelength_sequence import generate_wavelengths
from instruments.cm110 import CM110
from instruments.pem100 import PEM100
from instruments.zurich_mfli import ZurichMFLI
from output.csv_writer import write_aux_channel_csv, write_labview_style_csvs, write_I_sample_csv, write_I_avg_csv
from output.notes_writer import write_notes_file
from processing.labview_output_math import (
    calculate_avg_row,
    normalize_x_y_by_sr830_scaled_reference,
    validate_reference_channel,
)

CONFIRMATION = 'I UNDERSTAND THIS WILL RUN A SHORT HARDWARE SCAN'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Short scan acquisition test.')
    parser.add_argument('--live', action='store_true', help='Run the short scan in live hardware mode.')
    parser.add_argument('--start', type=float, default=450.0, help='Start wavelength in nm.')
    parser.add_argument('--end', type=float, default=454.0, help='End wavelength in nm.')
    parser.add_argument('--step', type=float, default=1.0, help='Wavelength step in nm.')
    parser.add_argument('--points', type=int, default=100, help='Number of Zurich samples per wavelength.')
    parser.add_argument('--base-name', default='short_scan_test', help='Output base file name.')
    parser.add_argument('--scan-type', default='pos', help='Scan type: pos or neg.')
    parser.add_argument('--output-dir', default='data', help='Directory for output files.')
    parser.add_argument('--retardation', type=float, default=0.250, help='PEM retardation in waves.')
    parser.add_argument('--reference-aux-channel', default=None, help='Reference auxiliary channel (auxin0 or auxin1).')
    parser.add_argument('--sr830-sensitivity', type=float, default=None, help='SR830 sensitivity in volts.')
    parser.add_argument('--aux-channel', default='auxin0', help='Auxiliary channel to save (auxin0 or auxin1).')
    parser.add_argument('--save-aux', action='store_true', help='Save auxiliary channel sample data during the scan.')
    parser.add_argument('--record-intensity', action='store_true', help='Record raw detector intensity (I_sample) during the scan.')
    parser.add_argument('--intensity-channel', default=None, choices=['auxin0', 'auxin1'], help='Raw intensity channel to record (auxin0 or auxin1).')
    parser.add_argument('--settle-time', type=float, default=None, help='Settling time in seconds.')
    parser.add_argument('--node', default='/dev4388/demods/0/sample', help='Zurich sample node.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    scan_defaults = get_scan_defaults(config)
    
    ddof = int(scan_defaults.get('standard_deviation_ddof', 1))
    settle_time = resolve_parameter(args.settle_time, scan_defaults, 'settle_time_s', 2, float, 'settle_time')
    sr830_sensitivity = resolve_parameter(args.sr830_sensitivity, scan_defaults, 'sr830_sensitivity_v', 0.5, float, 'sr830_sensitivity')
    reference_aux_channel = resolve_parameter(args.reference_aux_channel, scan_defaults, 'reference_aux_channel', 'auxin0', str, 'reference_aux_channel')
    aux_channel = validate_reference_channel(args.aux_channel)
    save_aux = args.save_aux
    aux_rows = []

    record_intensity = args.record_intensity
    intensity_channel = args.intensity_channel if args.intensity_channel is not None else aux_channel
    intensity_channel = validate_reference_channel(intensity_channel)
    intensity_rows = []
    i_avg_rows = []

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

        if args.live:
            print_scan_config(reference_aux_channel, sr830_sensitivity)

        pem.set_retardation(args.retardation)

        wavelengths = generate_wavelengths(args.start, args.end, args.step)
        x_rows = []
        y_rows = []
        avg_rows = []

        for wavelength in wavelengths:
            cm110.set_wavelength(int(round(wavelength)))
            pem.set_wavelength(wavelength)

            if args.live:
                print(f'Waiting {settle_time} seconds at wavelength {wavelength} nm...')
                time.sleep(settle_time)

            samples = zurich.poll_samples(args.points)

            # optional saved aux channel data (same as before)
            aux_values = [float(sample[aux_channel]) for sample in samples]
            if save_aux:
                aux_rows.append(aux_values)

            # raw intensity collection (I_sample) - raw voltages, no SR830 scaling
            i_values = [float(sample[intensity_channel]) for sample in samples]
            if record_intensity:
                intensity_rows.append(i_values)

            # MCD normalization (unchanged)
            x_ratio_samples, y_ratio_samples = normalize_x_y_by_sr830_scaled_reference(
                samples,
                reference_channel=reference_aux_channel,
                sr830_sensitivity=sr830_sensitivity,
                verbose=False,
            )
            x_rows.append(x_ratio_samples)
            y_rows.append(y_ratio_samples)

            mean_I_sample = float('nan')
            std_I_sample = float('nan')
            if record_intensity:
                mean_I_sample = sum(i_values) / len(i_values)
                if len(i_values) > 1:
                    std_I_sample = statistics.stdev(i_values)
                else:
                    std_I_sample = 0.0

            avg_rows.append(calculate_avg_row(wavelength, x_ratio_samples, y_ratio_samples, ddof=ddof))
            if record_intensity:
                i_avg_rows.append([mean_I_sample, std_I_sample])

            mean_x_ratio = sum(x_ratio_samples) / len(x_ratio_samples)
            mean_y_ratio = sum(y_ratio_samples) / len(y_ratio_samples)
            print(
                f'Wavelength {wavelength} nm: ref_mean={sum([float(s[reference_aux_channel]) for s in samples])/len(samples):.6f}, '
                f'I_sample mean={mean_I_sample:.6f}, '
                f'Xratio mean={mean_x_ratio:.6f}, Yratio mean={mean_y_ratio:.6f}'
            )

        x_path, y_path, avg_path = write_labview_style_csvs(
            base_name=args.base_name,
            scan_type=args.scan_type,
            output_dir=args.output_dir,
            wavelengths=wavelengths,
            x_rows=x_rows,
            y_rows=y_rows,
            avg_rows=avg_rows,
        )

        aux_path = None
        if save_aux:
            aux_path = write_aux_channel_csv(
                base_name=args.base_name,
                scan_type=args.scan_type,
                output_dir=args.output_dir,
                wavelengths=wavelengths,
                aux_rows=aux_rows,
                aux_channel=aux_channel,
            )

        i_path = None
        i_avg_path = None
        if record_intensity:
            i_path = write_I_sample_csv(
                base_name=args.base_name,
                scan_type=args.scan_type,
                output_dir=args.output_dir,
                wavelengths=wavelengths,
                i_rows=intensity_rows,
            )
            i_avg_path = write_I_avg_csv(
                base_name=args.base_name,
                scan_type=args.scan_type,
                output_dir=args.output_dir,
                wavelengths=wavelengths,
                i_avg_rows=i_avg_rows,
            )

        notes_path = write_notes_file(
            base_name=args.base_name,
            scan_type=args.scan_type,
            output_dir=args.output_dir,
            b_field_t=float(scan_defaults.get('b_field_t', 1.0)),
            path_length_m=float(scan_defaults.get('path_length_m', 0.002)),
            experiment_note='Short scan acquisition test',
        )

        print('Created files:')
        print(f'  {x_path}')
        print(f'  {y_path}')
        print(f'  {avg_path}')
        if aux_path is not None:
            print(f'  {aux_path}')
        if i_path is not None:
            print(f'  {i_path}')
        if i_avg_path is not None:
            print(f'  {i_avg_path}')
        print(f'  {notes_path}')
        print('PASS: Short scan test completed.')
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
