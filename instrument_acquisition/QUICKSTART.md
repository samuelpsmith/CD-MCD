# MCD Python Quickstart

Activate the environment:

```powershell
.\.venv\Scripts\activate
```

GUI live scan:

```powershell
python main_mcd_scan.py
```

The window has fields for scan settings and a Browse button for the output folder. Click `Start Scan` to run the scan.

Dry-run:

```powershell
python main_mcd_scan.py --start 400 --end 450 --step 2 --points 100 --base-name dryrun_test --scan-type pos --record-intensity
```

Dry-run is for setup/testing and does not move hardware.

Live positive scan:

```powershell
python main_mcd_scan.py --live --start 400 --end 450 --step 2 --points 100 --base-name ZnTPP_Soret --scan-type pos --sr830-sensitivity 0.1 --record-intensity
```

Live negative scan:

```powershell
python main_mcd_scan.py --live --start 400 --end 450 --step 2 --points 100 --base-name ZnTPP_Soret --scan-type neg --sr830-sensitivity 0.1 --record-intensity
```

Open data folder:

```powershell
explorer .\data
```

Run safe tests:

```powershell
python -m pytest
```

Archive code:

```powershell
Compress-Archive -Path .* -DestinationPath ..\mcd_python_acquisition_BACKUP_$(Get-Date -Format "yyyyMMdd_HHmm").zip -Force
```
