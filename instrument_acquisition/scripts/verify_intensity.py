import csv
import math

base = 'data/ZnTPP_intensity_dryrun_pos'

i_file = base + '_I_sample.csv'
avg_file = base + '_AVG.csv'

# read I_sample
with open(i_file, newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

# header + data
header = rows[0]
data_rows = rows[1:]

print('I_sample header columns:', len(header))

# check columns: first is wavelength, rest are samples
num_samples = len(header) - 1
print('Detected sample columns per row:', num_samples)

# read I_avg and build a mapping wavelength -> mean_I_sample
avg_map = {}
with open(avg_file.replace('_AVG.csv', '_I_avg.csv'), newline='') as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for r in reader:
        if not r:
            continue
        try:
            wl = float(r[0])
        except Exception:
            wl = r[0]
        mean_i = float(r[1]) if len(r) > 1 and r[1] != '' else float('nan')
        avg_map[wl] = mean_i

print('I_avg parsed, entries:', len(avg_map))

# verify each data row
all_ok = True
for row in data_rows:
    wl = float(row[0])
    samples = [float(x) for x in row[1:]]
    if len(samples) != num_samples:
        print(f'Row for {wl} has {len(samples)} samples, expected {num_samples}')
        all_ok = False
        continue
    mean_calc = sum(samples) / len(samples) if samples else float('nan')
    mean_reported = avg_map.get(wl)
    # allow small numerical discrepancy
    if math.isnan(mean_reported) and math.isnan(mean_calc):
        ok = True
    else:
        ok = abs(mean_calc - mean_reported) < 1e-9
    if not ok:
        print(f'Wavelength {wl}: mean_calc={mean_calc}, mean_reported={mean_reported} -> MISMATCH')
        all_ok = False

print('All rows match mean_I_sample in AVG:', all_ok)

# basic X/Y/AVG existence check
import os
for suffix in ['_X.csv','_Y.csv','_AVG.csv','_I_sample.csv','_I_avg.csv','_notes']:
    path = base + suffix
    exists = os.path.exists(path)
    print(path, 'exists:', exists)

if not all_ok:
    raise SystemExit(2)
else:
    print('Verification passed')
