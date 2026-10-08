#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instruments.pem100 import PEM100

CONFIRMATION = 'I UNDERSTAND THIS WILL COMMAND THE PEM'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test PEM wavelength and retardation commands.')
    parser.add_argument('--live', action='store_true', help='Send commands to the PEM100 in live mode.')
    parser.add_argument('--wavelength', type=float, default=400.0, help='Wavelength in nm to command.')
    parser.add_argument('--retardation', type=float, default=0.250, help='Retardation in waves to command.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = not args.live
    if args.live:
        answer = input(f'Type exactly: {CONFIRMATION}\n').strip()
        if answer != CONFIRMATION:
            print('Confirmation does not match. Aborting.')
            return 1

    pem = PEM100(dry_run=dry_run)
    try:
        if args.live:
            pem.open()

        print('Wavelength command:', pem.build_wavelength_command(args.wavelength).strip())
        print('Retardation command:', pem.build_retardation_command(args.retardation).strip())
        pem.set_wavelength(args.wavelength)
        pem.set_retardation(args.retardation)
        print('PASS: PEM command test completed.')
        return 0
    except Exception as exc:
        print('FAIL:', exc)
        return 1
    finally:
        if args.live:
            pem.close()


if __name__ == '__main__':
    raise SystemExit(main())
