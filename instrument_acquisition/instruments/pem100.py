"""Dry-run PEM100 polarizer driver."""

from __future__ import annotations

from typing import Optional


class PEM100:
    """Dry-run driver for the Hinds PEM100."""

    def __init__(
        self,
        port: str = 'COM1',
        baudrate: int = 2400,
        bytesize: int = 8,
        parity: str = 'N',
        stopbits: int = 1,
        timeout_s: float = 2.0,
        terminator: str = "\r",
        dry_run: bool = True,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout_s = timeout_s
        self.terminator = terminator
        self.dry_run = dry_run
        self._serial: Optional[object] = None

    def build_wavelength_command(self, wavelength_nm: float) -> str:
        """Build the PEM100 wavelength command string."""
        command_value = int(round(wavelength_nm * 10))
        return f'W:{command_value:06d}{self.terminator}'

    def build_retardation_command(self, retardation_waves: float) -> str:
        """Build the PEM retardation command string."""
        command_value = int(round(retardation_waves * 1000))
        return f'R:{command_value:04d}{self.terminator}'

    def open(self) -> None:
        """Open the serial connection or simulate it."""
        if self._serial is not None:
            return
        if self.dry_run:
            print(f'PEM100 dry run open: {self.port} {self.baudrate}')
            return
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError('pyserial is required for PEM100 live operation') from exc

        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.timeout_s,
        )

    def close(self) -> None:
        """Close the serial connection or simulate it."""
        if self.dry_run:
            print('PEM100 dry run close')
            return
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> 'PEM100':
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _write_and_read(self, command: str) -> str:
        if self.dry_run:
            print(f'PEM100 dry run command: {command!r}')
            return ''
        if self._serial is None:
            self.open()
        self._serial.write(command.encode('ascii'))
        self._serial.flush()
        response = self._serial.readline().decode('ascii', errors='ignore')
        return response

    def set_wavelength(self, wavelength_nm: float) -> None:
        """Send the wavelength command or simulate it."""
        command = self.build_wavelength_command(wavelength_nm)
        response = self._write_and_read(command)
        if response:
            print(f'PEM100 response: {response}')

    def set_retardation(self, retardation_waves: float) -> None:
        """Send the retardation command or simulate it."""
        command = self.build_retardation_command(retardation_waves)
        response = self._write_and_read(command)
        if response:
            print(f'PEM100 response: {response}')
