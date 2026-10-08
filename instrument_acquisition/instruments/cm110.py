"""Dry-run CM110 monochromator interface."""

from __future__ import annotations

from typing import Optional


class CM110:
    """Dry-run driver for the CM110 monochromator."""

    def __init__(
        self,
        port: str = 'COM4',
        baudrate: int = 9600,
        bytesize: int = 8,
        parity: str = 'N',
        stopbits: int = 1,
        timeout_s: float = 10.0,
        dry_run: bool = True,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout_s = timeout_s
        self.dry_run = dry_run
        self._serial: Optional[object] = None

    def build_goto_command(self, wavelength_nm: int) -> bytes:
        """Build the raw CM110 wavelength command."""
        if not isinstance(wavelength_nm, int):
            raise TypeError('wavelength_nm must be an integer')
        if not 0 <= wavelength_nm <= 65535:
            raise ValueError('wavelength_nm must be between 0 and 65535')
        return bytes([16, wavelength_nm // 256, wavelength_nm % 256])

    def open(self) -> None:
        """Open the serial connection or simulate it."""
        if self._serial is not None:
            return
        if self.dry_run:
            print(f'CM110 dry run open: {self.port} {self.baudrate} {self.bytesize}{self.parity}{self.stopbits}')
            return
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError('pyserial is required for CM110 live operation') from exc

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
            print('CM110 dry run close')
            return
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> 'CM110':
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def set_wavelength(self, wavelength_nm: int) -> None:
        """Move the monochromator to the requested wavelength or simulate the command."""
        command = self.build_goto_command(wavelength_nm)
        if self.dry_run:
            print(f'CM110 dry run command: {command}')
            return
        if self._serial is None:
            self.open()
        self._serial.write(command)
        self._serial.flush()
