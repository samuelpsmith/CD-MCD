# bruteFit

bruteFit processes paired absorption (ABS) and magnetic circular dichroism (MCD)
spectra, generates transition-aware peak guesses, searches combinations of A-term,
B-term, and D-term components, and reports transition-level A/D or B/D values.

This repository is currently research software intended for collaborator use. Review
the fitted curves, residuals, processing metadata, and scientific conventions before
using exported values in a publication.

## Installation

Python 3.12 is the currently tested development version.

```bash
cd /path/to/bruteFit
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, use the same environment for PySide6, NumPy, pandas, matplotlib, SciPy,
and lmfit. Mixing packages from multiple virtual environments has caused Qt import
errors and segmentation faults.

## GUI Workflow

Start bruteFit from the repository root:

```bash
python main.py
```

1. Choose **Load and Process** for raw positive-MCD, negative-MCD, and ABS files, or
   **Load Processed** for a previously generated processed CSV.
2. Review concentration, pathlength, field, and the MCD harmonic-correction choice.
3. Choose **Save and Continue**. The selected folder becomes the default location for
   result images and saved fit sessions.
4. Review automatic guesses in FitConfig. Left-click a marker to edit it; right-click
   a marker to remove it. Hover over markers and controls for details.
5. Accept the current peaks to run the candidate fits.
6. Rank results by MCD redchi/BIC or combined RMS/combo metrics and inspect both model
   components and residuals.

The processing window intentionally accepts only one dataset. Cancelled or failed
loads unlock the controls; a successful load disables them to prevent accidental
double loading.

## Harmonic Correction

The processing screen requires one of two explicit MCD modes:

- **Apply x2.5464**: use for raw 0-to-V square-wave measurements that have not been
  corrected upstream.
- **No correction**: use when the imported MCD values were already corrected.

The selected mode is applied consistently to MCD signal and propagated MCD noise. It
is written to processed CSVs, FitConfig, batch manifests, saved sessions, console
output, and result plots. Legacy processed files without this field use the
compatibility default of applying x2.5464; review the processing-screen selection to
avoid double correction.

## Peak Editing And Pairing

ABS guesses use Gaussian components. MCD guesses use a modular local A+B
decomposition, testing derivative-Gaussian A terms and Gaussian B terms around ABS
anchors. ABS and MCD peaks are paired one-to-one by nearest center within `MERGE_DX`.
Unpaired peaks remain available to describe the individual spectra, but only paired
transitions produce A/D or B/D values.

FitConfig supports three peak sources:

- Automatic peaks only
- Manual peaks only
- Automatic and manual peaks merged

Manual CSV input requires `source`, `pc`, `ps`, and exactly one of `height` or
`amplitude`. Visual height is converted to the normalized model coefficient used by
lmfit.

## Pair Constraints

The default `PAIR_CONSTRAINT_MODE` is `independent`, preserving separate ABS and MCD
optimizations. `bounded_offsets` performs a joint optimization in which paired MCD
parameters follow:

```text
MCD center = ABS center + bounded center offset
MCD sigma  = ABS sigma * bounded width ratio
```

Defaults permit a center offset of 20 cm^-1 and an MCD/ABS sigma ratio from 0.90 to
1.10. Setting both tolerances to zero shares center and sigma exactly. Unpaired
components remain independent.

Per-spectrum redchi and BIC displayed for joint fits are approximate diagnostics
because parameters are shared across both spectra. Treat curve/residual inspection as
primary until a dedicated joint model-selection statistic is implemented.

## Saving A Particular Fit

The results window can contain many ranked candidates. Set **Fit rank to save**, then
choose **Save Selected Fit Session**. A `.brutefit.json` session contains:

- The processed dataframe and an integrity hash
- FitConfig and harmonic-correction mode
- Transition assignments and A/B labels
- The selected model and final fitted parameters
- Ranking metric, filters, and displayed statistics

Load the session from either the processing or results window. It is displayed without
rerunning optimization. **Refit / Edit Parameters** uses the saved final parameters as
new starting guesses.

Sessions are versioned but reconstruct curves with the installed bruteFit model code.
Keep the Git commit identifier with publication records when long-term bit-for-bit
reproducibility is required.

## Batch Processing

Batch input is expected as compound/run folders, with run names such as `Q0`, `Q1`,
`Q0_Q1`, or `B`. Each run folder must contain the required raw CSV files, and metadata
must provide `lims_ID`, `concentration_MOL_L`, `pathlength_cm`, and `field_B`.

Preview discovered runs:

```bash
python scripts/batch_fit.py /path/to/dataset --dry-run
```

Run the tree and export the top three fits:

```bash
python scripts/batch_fit.py /path/to/dataset \
  --top-n 3 \
  --metric redchi \
  --harmonic-correction apply
```

For upstream-corrected MCD data:

```bash
python scripts/batch_fit.py /path/to/dataset \
  --harmonic-correction already-corrected
```

Add `--pair-constraints` to enable bounded joint fitting. Each run writes a
`brutefit_batch/` folder containing processed data, initial guesses, SVG figures, a
transition summary CSV, and a JSON manifest.

## Scientific Conventions

Model normalization, units, conversion constants, sign reporting, and known fitting
limitations are documented in [docs/SCIENTIFIC_CONVENTIONS.md](docs/SCIENTIFIC_CONVENTIONS.md).

## Important Source Files

- `bruteFit/utils.py`: processing GUI and dataset loading
- `processrecord/`: raw-file parsing and processed dataframe construction
- `bruteFit/plotwindow.py`: FitConfig preview, peak editing, and result GUI
- `bruteFit/mcd_guessing.py`: modular MCD A+B initial-guess strategy
- `bruteFit/transition_matching.py`: paired/ABS-only/MCD-only transition objects
- `bruteFit/dataFitting.py`: model generation, fitting, ranking, and plotting
- `bruteFit/fit_session.py`: versioned selected-fit persistence
- `bruteFit/batch.py`: headless orchestration and exports

## Development Checks

Before sharing changes, run at minimum:

```bash
python -m compileall bruteFit processrecord scripts main.py
git diff --check
python scripts/batch_fit.py --help
```

Generated datasets, virtual environments, Qt caches, and batch output folders should
not be committed.
