# MCD Python Verification Checklist

Use this checklist when finalizing or revalidating the acquisition computer.

- [ ] Dry-run output verified
- [ ] Zurich readout test passed
- [ ] Serial open/close passed
- [ ] PEM command test passed
- [ ] CM110 command test passed
- [ ] One-wavelength test passed
- [ ] Short scan test passed
- [ ] Positive ZnTPP scan looks reasonable
- [ ] Negative ZnTPP scan looks reasonable
- [ ] Intensity output verified
- [ ] `AVG.csv` confirmed MCD-only
- [ ] Backup archive created

Safe checks only:

```powershell
python -m pytest
python main_mcd_scan.py --start 400 --end 404 --step 2 --points 5 --base-name final_dryrun_verify --scan-type pos --sr830-sensitivity 0.1 --record-intensity
```

Suggested final live verification:

```powershell
python main_mcd_scan.py --live --start 400 --end 450 --step 2 --points 100 --base-name ZnTPP_final_verify_Soret --scan-type pos --sr830-sensitivity 0.1 --record-intensity
```
