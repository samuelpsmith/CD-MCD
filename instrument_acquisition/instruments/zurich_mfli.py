"""Zurich MFLI interface for dry-run simulation and live LabOne polling."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np


class ZurichMFLI:
    """Driver for Zurich MFLI data acquisition, with dry-run fallback."""

    def __init__(
        self,
        device_id: str = 'DEV4388',
        connection: str = 'ethernet',
        demod_index: int = 0,
        sample_node: Optional[str] = None,
        reference_aux_channel: str = 'auxin0',
        poll_timeout_ms: int = 5,
        settings: Optional[Dict[str, Any]] = None,
        dry_run: bool = True,
        host: str = 'localhost',
        port: int = 8004,
        api_level: int = 6,
        interface: str = '1GbE',
        assume_already_connected: bool = False,
        disconnect_first: bool = False,
        no_connect_device: bool = False,
    ) -> None:
        self.device_id = str(device_id)
        self.connection = connection
        self.demod_index = demod_index
        self.sample_node = sample_node if sample_node is not None else f'/{self.device_id.lower()}/demods/0/sample'
        self.reference_aux_channel = reference_aux_channel
        self.poll_timeout_ms = poll_timeout_ms
        self.settings = settings or {}
        self.dry_run = dry_run
        self.host = host
        self.port = port
        self.api_level = api_level
        self.interface = interface or connection
        self.assume_already_connected = assume_already_connected
        self.disconnect_first = disconnect_first
        self.no_connect_device = no_connect_device
        self._api: Optional[Any] = None
        self.root_nodes_before: List[str] = []
        self.root_nodes_after: List[str] = []
        self.root_nodes: List[str] = []
        self.device_found = False
        self.connected_interface: Optional[str] = None

    @staticmethod
    def _normalize_node_name(node: Any) -> str:
        return str(node).strip().strip('/').lower()

    def _device_path(self) -> str:
        return f'/{self.device_id.lower()}'

    def _found_node(self, nodes: List[str], node_path: str) -> bool:
        normalized_path = self._normalize_node_name(node_path)
        return any(self._normalize_node_name(item) == normalized_path for item in nodes)

    def _record_nodes(self, nodes: Any) -> List[str]:
        if nodes is None:
            return []
        return [str(node) for node in nodes]

    @staticmethod
    def _to_scalar_float(value: Any, field_name: str = 'value') -> float:
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return float(value)
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(f'Could not convert {field_name} to scalar float: {exc}') from exc
        array = array.ravel()
        if array.size == 0:
            raise ValueError(f'{field_name} is empty and cannot be converted to a scalar float')
        try:
            return float(array[0])
        except Exception as exc:
            raise ValueError(f'Could not convert {field_name} to scalar float: {exc}') from exc

    def open(self) -> None:
        """Open the Zurich connection or simulate it."""
        if self.dry_run:
            print(f'ZurichMFLI dry run open: {self.device_id} {self.sample_node}')
            return

        try:
            import zhinst.core as zhinst_core
        except ImportError as exc:
            raise RuntimeError('zhinst.core is required for Zurich live operation') from exc

        self._api = zhinst_core.ziDAQServer(self.host, self.port, self.api_level)
        self._api.connect()
        print('Data Server connection success')
        print(f'LabOne Data Server connected to {self.host}:{self.port} with API level {self.api_level}')

        print('Root nodes before connectDevice:')
        self.root_nodes_before = self._record_nodes(self._api.listNodes('/', 0))
        for node in self.root_nodes_before:
            print(node)

        self.device_found = self._found_node(self.root_nodes_before, self._device_path())
        print(f'{self._device_path()} found: {self.device_found}')

        if self.disconnect_first:
            try:
                self._api.disconnectDevice(self.device_id)
                print(f'Disconnected {self.device_id} before reconnect attempt.')
            except Exception as exc:
                print(f'Failed to disconnect {self.device_id} before reconnect attempt: {exc}')

        if self.no_connect_device:
            print(f'Skipping connectDevice for {self.device_id} because --no-connect-device was requested.')
            self.root_nodes_after = list(self.root_nodes_before)
        else:
            attempt_interfaces = [self.interface]
            if self.interface.lower() == '1gbe':
                attempt_interfaces = ['1GbE', 'USB']
            elif self.interface.lower() == 'usb':
                attempt_interfaces = ['USB']

            for interface in attempt_interfaces:
                try:
                    print(f'Connecting device {self.device_id} on interface {interface}')
                    self._api.connectDevice(self.device_id, interface)
                    self.connected_interface = interface
                    print(f'Device {self.device_id} connected through {interface}')
                    break
                except Exception as exc:
                    message = str(exc)
                    if 'already in use' in message.lower() or 'in use' in message.lower():
                        print(
                            f'{self.device_id} appears to already be in use, probably by the LabOne GUI or another API session. '
                            'Close/disconnect the instrument in LabOne, or rerun this test with --assume-already-connected '
                            'if the node is already available.'
                        )
                        if self.assume_already_connected:
                            print('Continuing because --assume-already-connected was set.')
                            break
                        raise RuntimeError(
                            f'{self.device_id} appears to already be in use, probably by the LabOne GUI or another API session. '
                            'Close/disconnect the instrument in LabOne, or rerun this test with --assume-already-connected '
                            'if the node is already available.'
                        ) from exc
                    print(f'Failed to connect device {self.device_id} on {interface}: {exc}')

            self.root_nodes_after = self._record_nodes(self._api.listNodes('/', 0))

        print('Root nodes after connectDevice attempt:')
        self.root_nodes = list(self.root_nodes_after)
        for node in self.root_nodes:
            print(node)

        self.device_found = self._found_node(self.root_nodes, self._device_path())
        print(f'{self._device_path()} found: {self.device_found}')

        if self.device_found:
            if self.connected_interface is None:
                self.connected_interface = self.interface
            return

        if self.assume_already_connected or self.no_connect_device:
            print(
                f'Continuing to direct node access because {self.device_id} was not visible in the API root nodes. '
                'Polling may still work if the node is already available.'
            )
            return

        print(
            f'LabOne Data Server is reachable, but {self.device_id} is not attached through the API. '
            'LabOne GUI may show the device, but Python still needs connectDevice(). '
            'Check interface type, device ID, and Data Server.'
        )
        raise RuntimeError(
            f'LabOne Data Server is reachable, but {self.device_id} is not attached through the API.'
        )

    def close(self) -> None:
        """Close the Zurich connection or simulate it."""
        if self.dry_run:
            print('ZurichMFLI dry run close')
            return
        if self._api is not None:
            self._api.disconnect()
            self._api = None

    def __enter__(self) -> 'ZurichMFLI':
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _fake_sample(self) -> Dict[str, Any]:
        phase_deg = random.uniform(-180.0, 180.0)
        x = math.cos(math.radians(phase_deg)) + random.gauss(0.0, 0.02)
        y = math.sin(math.radians(phase_deg)) + random.gauss(0.0, 0.02)
        return {
            'x': x,
            'y': y,
            'auxin0': 1.0 + random.gauss(0.0, 0.01),
            'auxin1': 0.5 + random.gauss(0.0, 0.01),
            'phase_deg': phase_deg,
            'timestamp': time.time(),
        }

    def poll_one_sample(self) -> Dict[str, Any]:
        """Poll one demod sample from Zurich or simulate it."""
        if self.dry_run:
            return self._fake_sample()
        if self._api is None:
            raise RuntimeError('ZurichMFLI is not connected')

        try:
            sample = self._api.getSample(self.sample_node) # NOTE how does this work? Just X and Y values? Divide by the intensity? 
        except Exception as exc:
            raise RuntimeError(
                f'Polling node {self.sample_node} failed. If the device is already attached in LabOne, '
                'use --assume-already-connected or --no-connect-device, or close/disconnect the instrument '
                'in LabOne and retry.'
            ) from exc

        if not isinstance(sample, dict):
            raise RuntimeError(f'Polling node {self.sample_node} returned unsupported sample: {type(sample).__name__}')

        phase = sample.get('phase', sample.get('phase_deg'))
        return {
            'x': self._to_scalar_float(sample.get('x'), 'x'),
            'y': self._to_scalar_float(sample.get('y'), 'y'),
            'auxin0': self._to_scalar_float(sample.get('auxin0'), 'auxin0'),
            'auxin1': self._to_scalar_float(sample.get('auxin1'), 'auxin1'),
            'phase_deg': self._to_scalar_float(phase, 'phase_deg'),
            'timestamp': self._to_scalar_float(sample.get('timestamp', time.time()), 'timestamp'),
        }

    def poll_samples(self, n: int) -> List[Dict[str, Any]]:
        """Poll multiple samples from Zurich or simulate them."""
        if n <= 0:
            raise ValueError('n must be positive')
        return [self.poll_one_sample() for _ in range(n)]

    def get_fake_sample(self, wavelength_nm: float) -> Dict[str, Any]:
        """Generate a fake MFLI sample set for a wavelength in dry-run mode."""
        sample = self._fake_sample()
        sample['wavelength_nm'] = float(wavelength_nm)
        return sample
