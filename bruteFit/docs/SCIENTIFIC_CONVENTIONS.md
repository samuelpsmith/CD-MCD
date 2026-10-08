# Scientific Conventions

This document records the numerical and reporting conventions currently implemented
by bruteFit. It is intended to make scientific behavior reviewable independently of
the GUI labels.

## Coordinate And Energy Units

Fitting is performed against `wavenumber_out` in cm^-1. Peak centers, sigmas,
`MERGE_DX`, `DELTA_CTR`, and paired center offsets therefore use wavenumber units, not
array-point counts or wavelength units.

The fitting arrays are prepared in `bruteFit.dataFitting.prepare_fit_arrays`:

```text
D fitting data = ABS extinction / (wavenumber * 326.6)
MCD fitting data = MCD delta-extinction * harmonic_factor
                   / (wavenumber * 152.5)
```

The effective-field correction is currently `1.0`. The constants `326.6` and `152.5`
are retained from the established physical conversion equations and should be cited
or derived in publication-facing documentation.

## MCD Harmonic Correction

`HARMONIC_CORRECTION_MODE` has two supported values:

| Mode | Factor | Intended input |
| --- | ---: | --- |
| `apply_2.5464` | 2.5464 | Raw 0-to-V square-wave MCD data |
| `already_corrected` | 1.0 | MCD data corrected upstream |

The 2.5464 value is implemented as `2 * 1.2732`, where 1.2732 approximates `4/pi`.
The additional factor of two reflects a square wave ranging from zero to positive
voltage rather than being symmetric around zero. Signal and propagated noise receive
the same factor.

The mode is explicit because automatically inferring whether an arbitrary historical
CSV was corrected is unsafe. Processed data should retain the
`harmonic_correction_mode` column.

## Lineshapes And Amplitudes

ABS D terms and MCD B terms use normalized Gaussian functions. MCD A terms use the
first derivative of the same normalized Gaussian basis. The parameter named
`amplitude` is the model coefficient, not visual peak height:

- For a Gaussian, amplitude is the integrated area coefficient.
- For a derivative Gaussian, amplitude is the derivative-basis coefficient and
  carries one additional x dimension relative to a Gaussian area.

Manual GUI input uses visual peak height because it is easier to estimate. bruteFit
converts height to the appropriate model coefficient using the selected center,
sigma, and A/B/D lineshape. Hover text exposes both values.

## A-Term Sign Convention

The optimizer and plotted curve use the derivative coefficient required in ascending
wavenumber space. Converting the horizontal coordinate orientation used for typical
wavelength-space interpretation reverses the A-term sign.

Therefore:

```text
reported literature A amplitude = - fitted wavenumber A coefficient
```

The sign reversal occurs only at the reporting boundary. It does not alter the fitted
curve, residual, optimizer state, or saved raw coefficient. Batch summaries use:

- `mcd_value`: literature-sign A amplitude
- `mcd_fit_value_wavenumber`: raw fitted coefficient

Reported A/D uses the literature-sign A value. Historical B/D behavior remains an
absolute magnitude.

## Peak Pairing And Ratios

ABS and MCD guesses are paired by nearest center within `MERGE_DX`, using a one-to-one
greedy assignment. Transitions may be paired, ABS-only, or MCD-only. All retained
components may contribute to fitting, while ratios are reported only for paired
transitions.

```text
A/D = reported A derivative coefficient / fitted D Gaussian area coefficient
B/D = |fitted B Gaussian area coefficient / fitted D Gaussian area coefficient|
```

## Sigma And Center Constraints

In independent mode, ABS and MCD parameters are optimized separately around their own
initial guesses:

```text
center in initial center +/- DELTA_CTR
sigma  in initial sigma  +/- DELTA_SIGMA
```

Equal initial guesses and small deltas provide pseudo-constrained behavior but do not
guarantee final cross-spectrum separation. Bounded pair mode instead uses an explicit
MCD center offset and MCD/ABS sigma ratio.

## ABS Height Cap

ABS area amplitude is expressed as a bounded fraction of a local observed peak height,
with sigma included in the area/height conversion. This prevents any single Gaussian
component from becoming taller than its local measured ABS maximum as sigma changes.

This is deliberately a **component-level** cap. It does not guarantee that the sum of
multiple overlapping ABS components stays below the measured trace. The summed model
and residual must be inspected when components overlap.

## Smoothing And Guessing

Savitzky-Golay smoothing is used for initial peak detection and visual preview. Final
fitting operates on the prepared fit arrays rather than replacing the measured data
with smoothed curves. The justification for Savitzky-Golay and comparison with
alternative smoothers remain publication-documentation tasks.

ABS guesses use local peak measurements and Gaussian core fitting. MCD guesses use a
modular local A+B decomposition seeded from ABS anchors. Current MCD search slop and
sigma-scale factors are implementation defaults in `bruteFit/mcd_guessing.py` and may
be tuned as additional representative spectra are validated.

## Fit Statistics

Independent fits use lmfit's per-spectrum statistics. In bounded pair mode, ABS and
MCD are optimized jointly after normalizing each residual vector by its observed signal
scale. Displayed per-spectrum redchi and BIC values are reconstructed diagnostics and
do not constitute a rigorous joint model-selection statistic. For linked fits, inspect
both residual panels and treat rankings as provisional.

## Reproducibility Records

For a result intended for comparison or publication, retain:

- Raw source files and processing metadata
- Processed CSV with harmonic-correction mode
- Saved `.brutefit.json` selected-fit session
- Batch manifest and transition summary, when applicable
- SVG result plot
- Git commit identifier and Python environment/package versions
