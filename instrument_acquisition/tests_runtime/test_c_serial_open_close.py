#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from instruments.cm110 import CM110
from instruments.pem100 import PEM100

CONFIRMATION = 'I UNDERSTAND THIS WILL ACCESS SERIAL PORTS'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test serial open/close for CM110 and PEM100.')
    parser.add_argument('--live', action='store_true', help='Run actual serial open/close on COM4 and COM1.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = not args.live
    print('Dry-run mode:' if dry_run else 'Live mode:')
    if args.live:
        answer = input(f'Type exactly: {CONFIRMATION}\n').strip()
        if answer != CONFIRMATION:
            print('Confirmation does not match. Aborting.')
            return 1

    cm110 = CM110(dry_run=dry_run)
    pem = PEM100(dry_run=dry_run)

    try:
        cm110.open()
        cm110.close()
        pem.open()
        pem.close()
        print('PASS: Serial open/close test completed.')
        return 0
    except Exception as exc:
        print('FAIL: ', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
