#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import zhinst.core
    from zhinst.toolkit import Session
except Exception as exc:
    print(f'Failed to import Zurich modules: {exc}')
    raise SystemExit(1)


DEVICE_ID = 'DEV4388'


def print_discovery_details(label: str, data) -> None:
    print(f'{label}:')
    if isinstance(data, dict):
        for key in [
            'deviceid',
            'devicetype',
            'serveraddress',
            'serverport',
            'apilevel',
            'interfaces',
            'connected',
            'available',
            'owner',
            'status',
            'discoverable',
            'statusflags',
            'firmwarerev',
            'serverversion',
        ]:
            if key in data:
                print(f'  {key}: {data[key]}')
        reachable = data.get('discoverable', False)
        in_use = str(data.get('status', '')).lower().strip() == 'in use' or not bool(data.get('available', True))
        print(f'  reachable: {reachable}')
        print(f'  in use: {in_use}')
    else:
        print(f'  raw value: {data}')


def main() -> int:
    print('Zurich discovery diagnostic')
    print('Device ID:', DEVICE_ID)

    session = Session('localhost')
    print('session:', session)

    try:
        visible = session.devices.visible()
        print('session.devices.visible():', visible)
    except Exception as exc:
        print('session.devices.visible() failed:', exc)

    try:
        connected = session.devices.connected()
        print('session.devices.connected():', connected)
    except Exception as exc:
        print('session.devices.connected() failed:', exc)

    print('\nApproach A: zhinst.core ziDAQServer')
    try:
        daq = zhinst.core.ziDAQServer('localhost', 8004, 6)
        print('daq:', daq)
        try:
            nodes = daq.listNodes('/', 0)
            print('daq.listNodes("/", 0):', nodes)
        except Exception as exc:
            print('daq.listNodes failed:', exc)

        for method_name, query in [
            ('discoveryFind', DEVICE_ID),
            ('discoveryFind', DEVICE_ID.lower()),
            ('discoveryGet', DEVICE_ID),
            ('discoveryGet', DEVICE_ID.lower()),
        ]:
            try:
                method = getattr(daq, method_name)
                result = method(query)
                print(f'{method_name}({query!r}):', result)
            except Exception as exc:
                print(f'{method_name}({query!r}) failed:', exc)
    except Exception as exc:
        print('ziDAQServer setup failed:', exc)

    print('\nApproach B: zhinst.core.ziDiscovery')
    try:
        discovery = zhinst.core.ziDiscovery()
        print('discovery:', discovery)
        for query in [DEVICE_ID, DEVICE_ID.lower()]:
            try:
                found = discovery.find(query)
                print(f'discovery.find({query!r}):', found)
            except Exception as exc:
                print(f'discovery.find({query!r}) failed:', exc)

            try:
                info = discovery.get(query)
                print_discovery_details(f'discovery.get({query!r})', info)
            except Exception as exc:
                print(f'discovery.get({query!r}) failed:', exc)
    except Exception as exc:
        print('ziDiscovery setup failed:', exc)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
