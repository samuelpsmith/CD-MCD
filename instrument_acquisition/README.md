# MCD Python Acquisition

This project is the Python replacement for the working LabVIEW CD/MCD acquisition workflow. It controls the CD/MCD instrument, reads the Zurich MFLI through LabOne Data Server, applies the confirmed LabVIEW-equivalent MCD normalization, and writes LabVIEW-style CSV outputs for research scans.

Normal research use opens a simple GUI for a live scan:

```powershell
python main_mcd_scan.py
```

The GUI has fields for the same scan settings that were previously typed into the terminal, plus a Browse button for the output folder. Click `Start Scan` to validate the settings and run the existing scan code. The scripts in `tests_runtime/` are diagnostics and debugging tools only; they are not the normal acquisition interface.

## What It Replaces

The Python code replaces the old LabVIEW scan routine for collecting MCD X/Y data, normalizing by the reference signal, and writing scan files. It preserves the confirmed math:

```text
reference_scaled = reference_aux_voltage * sr830_sensitivity / 10.0
X_ratio = X / reference_scaled
Y_ratio = Y / reference_scaled
```

LabOne physical Aux In 1 maps to Python/API `auxin0`. The default reference channel is therefore `auxin0`.

## Required Hardware

- Zurich MFLI `DEV4388`
- LabOne Data Server on `localhost:8004`
- CM110 monochromator on `COM4`, 9600 baud
- Hinds PEM100 on `COM1`, 2400 baud, `\r` terminator
- SR830 sensitivity entered in volts

Confirmed defaults are stored in `config.yaml`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Startup Checklist

1. Turn on the Zurich MFLI, CM110, PEM100, detector, and SR830.
2. Start LabOne Data Server.
3. Confirm `DEV4388` is visible.
4. Confirm no LabVIEW instance is holding `COM4`.
5. Confirm the PEM is on `COM1`.
6. Confirm SR830 sensitivity and enter that value in volts.
7. Run a dry-run scan before live acquisition.

## LabOne Data Server

Start LabOne Data Server from the Zurich Instruments LabOne installation. The Python code expects:

```text
host: localhost
port: 8004
api_level: 6
device_id: DEV4388
sample_node: /dev4388/demods/0/sample
```

If `DEV4388` is already in use, close or disconnect the device from LabOne GUI or any other Python session, then retry.

## GUI Live Scan

Run:

```powershell
python main_mcd_scan.py
```

The window includes sample/base name, scan type, wavelength range, step size, points per wavelength, SR830 sensitivity, raw intensity recording, settle time, and output folder. Use Browse to choose the save folder, then click `Start Scan`.

## Dry-Run Scan

Dry-run is available through command-line options for setup and developer testing. Dry-run does not move hardware.

```powershell
python main_mcd_scan.py --start 400 --end 450 --step 2 --points 100 --base-name dryrun_test --scan-type pos --record-intensity
```

## Real MCD Scans

Live command-line scans print a full hardware and output summary before acquisition. The GUI is the normal launch path for live scans.

Live positive scan:

```powershell
python main_mcd_scan.py --live --start 400 --end 450 --step 2 --points 100 --base-name ZnTPP_Soret --scan-type pos --sr830-sensitivity 0.1 --record-intensity
```

Live negative scan:

```powershell
python main_mcd_scan.py --live --start 400 --end 450 --step 2 --points 100 --base-name ZnTPP_Soret --scan-type neg --sr830-sensitivity 0.1 --record-intensity
```

For normal GUI use:

```powershell
python main_mcd_scan.py
```

## Output Files

Without intensity recording:

```text
<base_name>_<scan_type>_X.csv
<base_name>_<scan_type>_Y.csv
<base_name>_<scan_type>_AVG.csv
<base_name>_<scan_type>_notes
```

With `--record-intensity`:

```text
<base_name>_<scan_type>_X.csv
<base_name>_<scan_type>_Y.csv
<base_name>_<scan_type>_AVG.csv
<base_name>_<scan_type>_notes
<base_name>_<scan_type>_I_sample.csv
<base_name>_<scan_type>_I_avg.csv
```

`AVG.csv` is MCD-only and does not include intensity columns. Raw detector intensity is written separately to `I_sample.csv`; average intensity is written separately to `I_avg.csv`.

## Absorbance Later

This program records sample intensity when requested, but it does not automatically calculate absorbance yet. Calculate absorbance later from blank and sample intensities:

```text
A = -log10(I_sample / I_blank)
```

Use matching wavelength grids and compare `I_avg.csv` files for blank and sample scans.

## Safe Tests

Run safe tests without hardware:

```powershell
python -m pytest
```

These cover config loading, wavelength generation, CM110 command bytes, PEM command formatting, SR830 scaling, `auxin0` defaults, intensity outputs, `AVG.csv` format, CLI overrides, and dry-run file generation.

Runtime hardware diagnostics are separate:

```text
tests_runtime/test_c_serial_open_close.py
tests_runtime/test_d_zurich_readout.py
tests_runtime/test_e_pem_command_test.py
tests_runtime/test_f_cm110_command_test.py
tests_runtime/test_g_one_wavelength_acquisition.py
tests_runtime/test_h_short_scan.py
```

Run those only when intentionally debugging hardware.

## Troubleshooting

LabOne Data Server not running: start LabOne Data Server and confirm `localhost:8004`.

`DEV4388` already in use: disconnect it from LabOne GUI or stop the other Python session, then retry.

`COM4` access denied: close LabVIEW or any serial terminal holding the CM110 monochromator.

`COM1` fails: check PEM power, cable, Windows Device Manager port assignment, and whether another program has the port open.

Invalid wavelength range or step direction: use a positive step for increasing scans and a negative step for decreasing scans.

Invalid SR830 sensitivity: enter the sensitivity in volts, greater than zero, for example `0.1`.

Near-zero reference aux voltage: confirm detector/reference wiring and remember LabOne Aux In 1 is Python `auxin0`.

Missing or invalid `config.yaml`: restore the project `config.yaml` and verify it has `zurich`, `cm110`, `pem`, `scan_defaults`, and `absorption` sections.

## Archive and Backup

From the project folder:

```powershell
Compress-Archive -Path .* -DestinationPath ..\mcd_python_acquisition_BACKUP_$(Get-Date -Format "yyyyMMdd_HHmm").zip -Force
```

Also back up completed data folders separately according to lab policy.

## Later Phases

Not implemented yet: magnet automation, automatic absorption correction, automatic positive/negative merging, database export, and plotting dashboard.
