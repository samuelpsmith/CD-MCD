#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from instruments.zurich_mfli import ZurichMFLI

CONFIRMATION = 'I UNDERSTAND THIS WILL ACCESS THE ZURICH MFLI'


def load_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    with config_path.open('r', encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Test Zurich-only readout.')
    parser.add_argument('--live', action='store_true', help='Run a live Zurich readout using zhinst.')
    parser.add_argument('--host', default='localhost', help='LabOne Data Server host.')
    parser.add_argument('--port', type=int, default=8004, help='LabOne Data Server port.')
    parser.add_argument('--api-level', type=int, default=6, help='LabOne API level.')
    parser.add_argument('--device', default='DEV4388', help='Zurich device ID to use.')
    parser.add_argument('--interface', default='1GbE', help='Zurich interface to use.')
    parser.add_argument('--node', default=None, help='Zurich sample node to poll. Defaults to /<device>/demods/0/sample.')
    parser.add_argument('--reference-aux-channel', default='auxin1', help='Reference auxiliary channel name.')
    parser.add_argument('--assume-already-connected', action='store_true', help='Continue when the device is already in use.')
    parser.add_argument('--disconnect-first', action='store_true', help='Disconnect the device before reconnecting.')
    parser.add_argument('--no-connect-device', action='store_true', help='Skip connectDevice and try direct polling.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = not args.live
    config = load_config()
    zurich_config = config.get('zurich', {})

    if args.live:
        answer = input(f'Type exactly: {CONFIRMATION}\n').strip()
        if answer != CONFIRMATION:
            print('Confirmation does not match. Aborting.')
            return 1

    normalized_device = str(args.device).strip().upper()
    sample_node = args.node if args.node is not None else f'/{normalized_device.lower()}/demods/0/sample'

    zurich = ZurichMFLI(
        device_id=normalized_device,
        connection=str(args.interface),
        demod_index=int(zurich_config.get('demod_index', 0)),
        sample_node=sample_node,
        reference_aux_channel=str(args.reference_aux_channel),
        poll_timeout_ms=int(zurich_config.get('poll_timeout_ms', 5)),
        settings=dict(zurich_config.get('settings', {})),
        dry_run=dry_run,
        host=str(args.host),
        port=int(args.port),
        api_level=int(args.api_level),
        interface=str(args.interface),
        assume_already_connected=args.assume_already_connected,
        disconnect_first=args.disconnect_first,
        no_connect_device=args.no_connect_device,
    )

    try:
        if args.live:
            zurich.open()
            print('Data Server connection success')
            print(f'Device ID being used: {zurich.device_id}')
            print(f'Interface being used: {zurich.connected_interface or zurich.interface}')
            print('Root nodes found:')
            for node in zurich.root_nodes:
                print(node)
            print(f'Found /{zurich.device_id.lower()}: {zurich.device_found}')
            print(f'Sample node being polled: {zurich.sample_node}')

        sample = zurich.poll_one_sample()
        available_keys = sorted(sample.keys())

        def print_field(label: str, key: str) -> None:
            if key not in sample:
                print(f'{label}: <missing> (available keys: {", ".join(available_keys)})')
            else:
                print(f'{label}: {sample.get(key)}')

        print_field('X', 'x')
        print_field('Y', 'y')
        print_field('Aux In 0', 'auxin0')
        print_field('Aux In 1', 'auxin1')

        phase_value = sample.get('phase')
        if phase_value is None and 'phase_deg' in sample:
            phase_value = sample.get('phase_deg')
        if phase_value is None:
            print(f'Phase: <missing> (available keys: {", ".join(available_keys)})')
        else:
            print(f'Phase: {phase_value}')

        print('PASS: Zurich readout completed.')
        return 0
    except Exception as exc:
        print('FAIL:', exc)
        print('Troubleshooting:')
        print(' - Close the LabOne GUI or click Disconnect Instrument in LabOne to release DEV4388.')
        print(' - Try: python tests_runtime/test_d_zurich_readout.py --live --host localhost --port 8004 --device DEV4388 --interface 1GbE --disconnect-first')
        print(' - Try: python tests_runtime/test_d_zurich_readout.py --live --host localhost --port 8004 --device DEV4388 --interface 1GbE --no-connect-device')
        print(' - Try: python tests_runtime/test_d_zurich_readout.py --live --host localhost --port 8004 --device DEV4388 --interface 1GbE --node /dev4388/demods/1/sample')
        return 1
    finally:
        if args.live:
            zurich.close()


if __name__ == '__main__':
    raise SystemExit(main())
