#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instruments.cm110 import CM110

CONFIRMATION = 'I UNDERSTAND THIS WILL MOVE THE MONOCHROMATOR'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test CM110 GOTO command.')
    parser.add_argument('--live', action='store_true', help='Send a CM110 GOTO command in live mode.')
    parser.add_argument('--wavelength', type=int, help='Wavelength in nm to command.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.live and args.wavelength is None:
        print('Live mode requires --wavelength to be provided.')
        return 1

    wavelength = args.wavelength if args.wavelength is not None else 450
    dry_run = not args.live

    if args.live:
        print('WARNING: Do not use a wavelength far from current monochromator position for first live test.')
        answer = input(f'Type exactly: {CONFIRMATION}\n').strip()
        if answer != CONFIRMATION:
            print('Confirmation does not match. Aborting.')
            return 1

    cm110 = CM110(dry_run=dry_run)
    try:
        if args.live:
            cm110.open()

        print('CM110 GOTO command bytes:', cm110.build_goto_command(wavelength))
        cm110.set_wavelength(wavelength)
        print('PASS: CM110 command test completed.')
        return 0
    except Exception as exc:
        print('FAIL:', exc)
        return 1
    finally:
        if args.live:
            cm110.close()


if __name__ == '__main__':
    raise SystemExit(main())
