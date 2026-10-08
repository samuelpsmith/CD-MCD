import time
import sys
from dataclasses import dataclass
from dateutil import rrule, tz  # noqa: F401
from PySide6.QtWidgets import QDialog
from . import gaussianModels
import itertools
from lmfit import minimize
from lmfit.model import ModelResult, Model
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from .fitConfig import FitConfig
from .gaussianModels import stable_gaussian_sigma
from .plotwindow import guessWindow, MainResultWindow, auto_guess_transitions
from .transition_matching import TransitionModelSpec

HARMONIC_CORRECTION_FACTOR = 2 * 1.2732


def harmonic_correction_details(fc: FitConfig) -> tuple[float, str]:
    mode = str(fc.HARMONIC_CORRECTION_MODE)
    if mode == "apply_2.5464":
        return HARMONIC_CORRECTION_FACTOR, "MCD harmonic correction: applied x2.5464"
    if mode == "already_corrected":
        return 1.0, "MCD harmonic correction: not applied (already corrected upstream)"
    raise ValueError(f"Unknown HARMONIC_CORRECTION_MODE: {mode!r}")


def reported_mcd_amplitude(prefix: str | None, amplitude: float) -> float:
    """Convert fitted wavenumber-space A coefficients to the literature sign convention."""
    value = float(amplitude)
    return -value if prefix and str(prefix).startswith("A") else value


def reported_transition_ratio(prefix: str | None, mcd_amplitude: float, abs_amplitude: float) -> float:
    """Report signed A/D in the literature convention; preserve historical |B/D| reporting."""
    if prefix and str(prefix).startswith("A"):
        return reported_mcd_amplitude(prefix, mcd_amplitude) / float(abs_amplitude)
    return abs(float(mcd_amplitude) / float(abs_amplitude))

def _resolve_process_count(processes):
    """
    macOS has been prone to segfaults when Qt, matplotlib, scipy/lmfit, and joblib all
    coexist across worker processes. Until that is proven stable, force serial fitting
    on Darwin even if callers request more workers.
    """
    requested = max(1, int(processes or 1))
    if sys.platform == "darwin" and requested != 1:
        print("macOS detected; forcing single-process fitting to avoid Qt/joblib segfaults.")
        return 1
    return requested


@dataclass
class FitCandidate:
    mcd_model: Model | None
    mcd_params: object
    abs_model: Model | None
    abs_params: object
    transition_specs: list[TransitionModelSpec]


@dataclass
class FitBundle:
    mcd_result: ModelResult
    abs_result: ModelResult
    transition_specs: list[TransitionModelSpec]


@dataclass
class LinkedModelResult:
    """Small ModelResult-compatible view of one side of a joint ABS/MCD fit."""
    model: Model
    params: object
    data: np.ndarray
    best_fit: np.ndarray
    residual: np.ndarray
    redchi: float
    bic: float

    def eval_components(self, x):
        return self.model.eval_components(params=self.params, x=x)

    def eval(self, x):
        return self.model.eval(params=self.params, x=x)


#TODO: Docs and use pyside6 for parameters

#TODO: add BIC back and maybe other metrics
class BfResult:
    def __init__(self, datax, datay, dataz, redchi_threshold=None, residual_rms_threshold=None, bic_threshold=None, processing_note=None):
        """
        Initialize the bfResult container with optional filter thresholds.

        Parameters:
            redchi_threshold (float): Max allowed reduced chi-square value. None = no filter.
            residual_rms_threshold (float): Max allowed residual RMS. None = no filter.
        """
        self.dataX = datax
        self.dataY = datay
        self.dataZ = dataz
        self.fit_bundles: list[FitBundle] = []
        self.redchi_threshold = redchi_threshold
        self.residual_rms_threshold = residual_rms_threshold
        self.bic_threshold = bic_threshold
        self.processing_note = str(processing_note or "")

        #default weights for combo metric TODO expose this so we can tune. 
        self.bic_w = 0.25
        self.redchi_w = 0.5
        self.rms_w = 0.25

    def add_result(self, result):
        """Add a new lmfit ModelResult to the list."""
        self.fit_bundles.append(result)

    def _residual_rms(self, result: ModelResult):
        """Compute the RMS of residuals for a fit."""
        return np.sqrt(np.mean(result.residual ** 2))

    def _filter_results(self, gc_start=None, gc_end=None, min_sigma = None, max_sigma = None, min_amplitude = None, max_amplitude = None):
        """
        Filter results based on thresholds set at initialization and number of components.
        Returns a list of ModelResult objects that pass all thresholds.

        Args:
            gc_start (int, optional): Minimum number of components allowed.
            gc_end (int, optional): Maximum number of components allowed.
        """
        filtered = list(self.fit_bundles)

        # --- Threshold filters ---
        if self.redchi_threshold is not None:
            filtered = [bundle for bundle in filtered if bundle.mcd_result.redchi < self.redchi_threshold]

        if self.residual_rms_threshold is not None:
            filtered = [
                bundle
                for bundle in filtered
                if self._residual_rms(bundle.mcd_result) < self.residual_rms_threshold
            ]

        if self.bic_threshold is not None:
            filtered = [bundle for bundle in filtered if bundle.mcd_result.bic < self.bic_threshold]

        # --- Component count filter ---
        start = gc_start if gc_start is not None else 0
        end = gc_end if gc_end is not None else float('inf')

        filtered = [
            bundle
            for bundle in filtered
            if start <= len(bundle.transition_specs) <= end
        ]

        #filter by min and max sigma/amp
        def get_sig(res):
            params_dict = self.eval_result(res)
            return [v for k, v in params_dict.items() if k.endswith("_sigma")]

        def get_amp(res):
            params_dict = self.eval_result(res)
            return [v for k, v in params_dict.items() if k.endswith("_amplitude")]

        # Ensure bounds are not None
        min_sigma = 0 if min_sigma is None else min_sigma
        max_sigma = float("inf") if max_sigma is None else max_sigma
        min_amplitude = 0 if min_amplitude is None else min_amplitude
        max_amplitude = float("inf") if max_amplitude is None else max_amplitude

        filtered_results = []
        for bundle in filtered:
            sigs = get_sig(bundle.mcd_result) + get_sig(bundle.abs_result)
            amps = get_amp(bundle.mcd_result) + get_amp(bundle.abs_result)

            # Skip if no components
            if not sigs or not amps:
                continue

            sig_in_range = all(min_sigma <= s <= max_sigma for s in sigs)
            amp_in_range = all(min_amplitude <= abs(a) <= max_amplitude for a in amps)

            if sig_in_range and amp_in_range:
                filtered_results.append(bundle)

        return filtered_results

    #TODO: add optional gaussian count g_n parameter
    def n_best_results(self, n=3, metric='redchi', gc_start=None, gc_end=None,min_sigma=None, max_sigma=None, min_amplitude=None, max_amplitude=None):
        # Filter results
        filtered = self._filter_results(gc_start, gc_end, min_sigma, max_sigma, min_amplitude, max_amplitude)

        valid_items = [bundle for bundle in filtered if bundle.mcd_result is not None]

        # Sort based on metric # inputs are pairs of model results
        if metric == 'residual_rms':
            return sorted(
                valid_items,
                key=lambda bundle: self._residual_rms(bundle.mcd_result) + self._residual_rms(bundle.abs_result)
            )[:n]
        elif metric == 'combo':
            return sorted(
                valid_items,
                key=lambda bundle: self._combo_metric(bundle.mcd_result) + self._combo_metric(bundle.abs_result)
            )[:n]
        else:
            #sort by any metric part of lmfit modelresult by default
            return sorted(valid_items, key=lambda bundle: getattr(bundle.mcd_result, metric))[:n]

    def _combo_metric(self, result):
        return self.bic_w * result.bic + self.redchi_w * result.redchi + self.rms_w * self._residual_rms(result)

    @staticmethod
    def _format_plot_param(name, value):
        if name.endswith("center"):
            return f"{value:.0f}"
        return f"{value:.3g}"

    @staticmethod
    def _reported_parameter_value(name, value):
        if name.endswith("amplitude"):
            prefix = name.rsplit("amplitude", 1)[0]
            return reported_mcd_amplitude(prefix, value)
        return value

    # TODO: add optional gaussian count g_n parameter
    def get_plot_figs(self, n=3, metric='redchi', gc_start=None, gc_end=None,min_sigma=None, max_sigma=None, min_amplitude=None, max_amplitude=None):
        """
        Plot the top N best fits over the data from both mcd and abs results.
        """
        fig_list = []

        results = self.n_best_results(n=n, metric=metric, gc_start=gc_start, gc_end=gc_end,min_sigma=min_sigma, max_sigma=max_sigma,min_amplitude=min_amplitude, max_amplitude=max_amplitude)

        for i, bundle in enumerate(results, 1):
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # 2 rows, 1 column subplots
            if self.processing_note:
                fig.suptitle(self.processing_note, fontsize=9, color="0.3")

            # Plot on the first axis
            self._plot_components_visible(bundle.mcd_result, self.dataX, self.dataZ, i, axs[0])

            # Plot on the second axis
            self._plot_components_visible(bundle.abs_result, self.dataX, self.dataY, i, axs[1])

            self.print_A_over_D(bundle, self.dataX)

            # Reserve a right-hand parameter column. Parameter text used to sit left
            # of the axes where it could collide with the MCD "Intensity" label.
            fig.tight_layout(rect=(0.0, 0.0, 0.76, 0.97 if self.processing_note else 1.0))
            fig_list.append(fig)
            plt.close(fig)  # Close to free memory if plotting many

        return fig_list
    def _plot_components_visible(self, result, x, z, i, ax):
        """Plot the original z-data, model components, metrics, and parameters on given axis."""
        #TODO: Talk with sam about plotting
        #No new figure creation here — use the provided ax

        # Plot measured data
        ax.plot(x, z, '-', lw=1, alpha=0.7, label='Measured Data')

        # Plot the combined fit so the overall model shape is visible alongside the
        # individual components.
        combined_fit = np.asarray(result.best_fit if hasattr(result, "best_fit") else result.eval(x=x), dtype=float)
        ax.plot(
            x,
            combined_fit,
            color="0.25",
            linestyle="--",
            lw=1.2,
            alpha=0.45,
            label="Combined Fit",
            zorder=3,
        )

        # Evaluate and plot each component
        components = result.eval_components(x=x)
        for name, comp in components.items():
            ax.plot(x, comp, lw=2, label=name)

        self._plot_residual_band(ax, result, x, z)

        # Axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Intensity")

        # Metrics in top-left (axes coords)
        metrics = {
            "Residual RMS": self._residual_rms(result),
            "BIC": getattr(result, 'bic'),
            "redchi": result.redchi,
            "Combo": self._combo_metric(result)
        }
        metrics_text = "\n".join(f"{name}: {val:.3g}" for name, val in metrics.items())
        ax.text(0.02, 0.95, metrics_text,transform=ax.transAxes, fontsize=8,va='top', ha='left')

        # Index number in top-right (axes coords)
        ax.text(0.98, 0.95, f"#{i}",transform=ax.transAxes, fontsize=12,va='top', ha='right')

        # Parameters from eval_result (grouped, with separation) – placed in data coords
        params_dict = self.eval_result(result)  # your single-result eval function

        grouped = defaultdict(list)
        for name, val in params_dict.items():
            prefix = name.split('_', 1)[0]  # e.g., "A0", "B1"
            reported_value = self._reported_parameter_value(name, val)
            grouped[prefix].append(f"{name}: {self._format_plot_param(name, reported_value)}")

        # Build lines with a blank line between groups
        params_lines = []
        for prefix in sorted(grouped.keys()):
            params_lines.extend(grouped[prefix])
            params_lines.append("")  # blank line between groups
        params_text = "\n".join(params_lines).strip()

        # Keep parameters in a reserved sidebar instead of competing with axis labels
        # or spectral components inside the data area.
        ax.text(
            1.02,
            0.95,
            params_text,
            fontsize=7,
            va="top",
            ha="left",
            transform=ax.transAxes,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.85", alpha=0.82),
        )

        ax.legend()

        # print results too
        self.print_eval_result(result, i)

    def _plot_residual_band(self, ax, result, x, z):
        residual = np.asarray(result.residual, dtype=float)
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        if residual.size == 0 or x.size != residual.size or z.size == 0:
            return

        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        z_span = max(z_max - z_min, 1e-12)
        residual_scale = float(np.nanmax(np.abs(residual)))
        if not np.isfinite(residual_scale) or residual_scale <= 0:
            return

        band_height = 0.12 * z_span
        band_center = z_min - 0.18 * z_span
        scaled_residual = band_center + (residual / residual_scale) * (0.45 * band_height)

        ax.axhspan(
            band_center - 0.55 * band_height,
            band_center + 0.55 * band_height,
            color="0.5",
            alpha=0.06,
            zorder=0,
        )
        ax.axhline(
            band_center,
            color="0.5",
            lw=0.8,
            alpha=0.25,
            ls="--",
            zorder=1,
        )
        ax.plot(
            x,
            scaled_residual,
            color="0.25",
            lw=1.0,
            alpha=0.65,
            zorder=2,
        )
        ax.text(
            0.98,
            0.06,
            "Residual (scaled)",
            transform=ax.transAxes,
            fontsize=7,
            ha="right",
            va="bottom",
            color="0.35",
            bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.45),
        )
    def eval_result(self, result):
        if not hasattr(result, "params"):
            raise ValueError("Provided object has no 'params' attribute")

        # Hide helper parameters that only exist to enforce fit constraints.
        visible_names = set(getattr(getattr(result, "model", None), "param_names", ()))
        return {
            name: p.value
            for name, p in result.params.items()
            if not name.endswith("_height_scale")
            and (not visible_names or name in visible_names)
        }

    def print_eval_result(self, result, i):
        """Pretty-print parameters from a single lmfit ModelResult."""
        params_dict = self.eval_result(result)
        print(f"=== Fit Parameters === Index: {i}")
        for name, value in params_dict.items():
            reported_value = self._reported_parameter_value(name, value)
            formatted = f"{reported_value:.0f}" if name.endswith("center") else f"{reported_value:.6g}"
            print(f"{name:20} {formatted}")
        print(f"=== Parameters === END: {i}")

    def print_A_over_D(self, bundle: FitBundle, x):
        del x
        print("=== Transition Ratios ===")
        for spec in bundle.transition_specs:
            if spec.status == "paired" and spec.mcd_prefix and spec.abs_prefix:
                mcd_amp = bundle.mcd_result.params[f"{spec.mcd_prefix}amplitude"].value
                abs_amp = bundle.abs_result.params[f"{spec.abs_prefix}amplitude"].value
                ratio = reported_transition_ratio(spec.mcd_prefix, mcd_amp, abs_amp)
                label = spec.ratio_label or "A/D or B/D"
                print(
                    f"{label} for {spec.transition_id}: {ratio:.6g} "
                    f"(mcd={spec.mcd_prefix} abs={spec.abs_prefix})"
                )
            elif spec.status == "abs_only":
                center = None if spec.abs_peak is None else spec.abs_peak.center
                print(f"{spec.transition_id}: ABS-only transition at center {center:.6g}")
            elif spec.status == "mcd_only":
                center = None if spec.mcd_peak is None else spec.mcd_peak.center
                print(f"{spec.transition_id}: MCD-only transition at center {center:.6g}")
        print("=== Transition Ratios END ===")

def build_fit_candidates_from_transitions(transitions, fc):
    """
    Build every model candidate implied by the preview transitions.

    Important behavior:
    - ABS peaks always become D-like Gaussian components in the absorption model.
    - MCD peaks are tried as both A-like derivative Gaussians and B-like Gaussians.
    - Unpaired peaks are kept as fit components, but only paired transitions later
      produce A/D or B/D ratios.
    """
    model_list = [gaussianModels.model_stable_gaussian_sigma, gaussianModels.model_stable_gaussian_deriv_sigma]
    if not transitions:
        raise RuntimeError("No peak transitions were accepted for fitting.")

    max_subset_size = min(len(transitions), int(fc.MAX_GC))
    all_subsets = []
    for r in range(max(1, fc.MIN_GC), max_subset_size + 1):
        all_subsets.extend(itertools.combinations(transitions, r))

    all_subsets = [list(subset) for subset in all_subsets]
    print(f"total subset length: {len(all_subsets)}")
    all_models: list[FitCandidate] = []

    total_subsets = len(all_subsets)
    for idx, subset in enumerate(all_subsets, start=1):
        print(f"{idx}/{total_subsets} - subset size {len(subset)}")
        mcd_transition_count = sum(1 for transition in subset if transition.mcd_peak is not None)
        if mcd_transition_count == 0 or not any(transition.abs_peak is not None for transition in subset):
            continue

        # Each MCD-bearing transition creates a binary A-vs-B branch. This keeps the
        # preview logic simple while letting final ranking decide whether an A or B
        # lineshape better describes the MCD data.
        for model_choices in itertools.product(model_list, repeat=mcd_transition_count):
            composite_mcd_model = None
            params_mcd = None
            composite_abs_model = None
            params_abs = None
            transition_specs: list[TransitionModelSpec] = []
            mcd_choice_index = 0

            for i, transition in enumerate(subset, start=1):
                abs_prefix = None
                mcd_prefix = None
                ratio_label = None

                if transition.mcd_peak is not None:
                    base_model = model_choices[mcd_choice_index]
                    mcd_choice_index += 1
                    prefix_letter = "A" if base_model == gaussianModels.model_stable_gaussian_deriv_sigma else "B"
                    mcd_prefix = f"{prefix_letter}{i}_"
                    m_mcd = Model(base_model.func, prefix=mcd_prefix)
                    peak = transition.mcd_peak
                    if composite_mcd_model is None:
                        composite_mcd_model = m_mcd
                        params_mcd = m_mcd.make_params(
                            amplitude=peak.amplitude,
                            center=peak.center,
                            sigma=peak.sigma,
                        )
                    else:
                        composite_mcd_model += m_mcd
                        params_mcd.update(
                            m_mcd.make_params(
                                amplitude=peak.amplitude,
                                center=peak.center,
                                sigma=peak.sigma,
                            )
                        )
                    if transition.abs_peak is not None:
                        ratio_label = f"{prefix_letter}/D"

                if transition.abs_peak is not None:
                    abs_prefix = f"D{i}_"
                    m_abs = Model(stable_gaussian_sigma, prefix=abs_prefix)
                    peak = transition.abs_peak
                    if composite_abs_model is None:
                        composite_abs_model = m_abs
                        params_abs = m_abs.make_params(
                            amplitude=peak.amplitude,
                            center=peak.center,
                            sigma=peak.sigma,
                        )
                    else:
                        composite_abs_model += m_abs
                        params_abs.update(
                            m_abs.make_params(
                                amplitude=peak.amplitude,
                                center=peak.center,
                                sigma=peak.sigma,
                            )
                        )

                transition_specs.append(
                    TransitionModelSpec(
                        transition_id=transition.transition_id,
                        status=transition.status,
                        abs_peak=transition.abs_peak,
                        mcd_peak=transition.mcd_peak,
                        match_distance=transition.match_distance,
                        abs_prefix=abs_prefix,
                        mcd_prefix=mcd_prefix,
                        ratio_label=ratio_label,
                    )
                )

            if composite_mcd_model is None or composite_abs_model is None:
                continue

            all_models.append(
                FitCandidate(
                    mcd_model=composite_mcd_model,
                    mcd_params=params_mcd,
                    abs_model=composite_abs_model,
                    abs_params=params_abs,
                    transition_specs=transition_specs,
                )
            )

    if not all_models:
        raise RuntimeError(
            "No valid fit candidates were generated. At least one ABS transition and one MCD transition "
            "must be present in the accepted preview."
        )
    return all_models, fc


def brute_force_models(x, y_abs, y_mcd, fc = FitConfig(), abs_noise=None, mcd_noise=None):
    """Open the GUI preview dialog, then convert the accepted preview to candidates."""
    dlg = guessWindow(x, y_abs, y_mcd, fc, abs_noise=abs_noise, mcd_noise=mcd_noise)

    if dlg.exec() == QDialog.DialogCode.Accepted:  # blocks until user clicks
        transitions = dlg.get_guess()
        fc = dlg.get_fc()
    else:
        raise RuntimeError("User cancelled peak guesses.")

    return build_fit_candidates_from_transitions(transitions, fc)


def prepare_fit_arrays(mcd_df, fc=None):
    """
    Convert processed-data columns into the x/ABS/MCD arrays used by all fitting paths.

    Keeping this conversion in one place is important because the GUI and batch runner
    must rank fits from exactly the same normalized data. MCD noise is propagated when
    ProcessRecord provides real std-dev columns from the pos/neg input files.
    """
    fc = fc or FitConfig()
    effective_field_corr = 1.0 # 1.32 for toluene, set to 1 to negate
    harmonic_correction, correction_note = harmonic_correction_details(fc)
    print(correction_note)
    x = mcd_df["wavenumber_out"]
    y_abs = mcd_df["uvvis_extinction_abs_molar-1cm-1_out"] / (mcd_df["wavenumber_out"] * effective_field_corr * 326.6)
    z_mcd = (mcd_df["deltaextinctionpertesla_mcdavg_molar-1cm-1T-1_out"] * harmonic_correction) / (mcd_df["wavenumber_out"] * effective_field_corr * 152.5)
    mcd_noise = None
    if "extinction_stddev_mcdavg_molar-1cm-1_out" in mcd_df.columns:
        field_values = np.asarray(mcd_df.get("field_B", 1.0), dtype=float)
        safe_field = np.where(np.abs(field_values) > 0, field_values, np.nan)
        mcd_noise = (
            np.asarray(mcd_df["extinction_stddev_mcdavg_molar-1cm-1_out"], dtype=float)
            * harmonic_correction
            / (safe_field * np.asarray(mcd_df["wavenumber_out"], dtype=float) * 152.5)
        )
    return x, y_abs, z_mcd, mcd_noise


def _split_into_n(seq, n):
    """Evenly split a sequence into n chunks."""
    n = max(1, min(n, len(seq)))
    k, m = divmod(len(seq), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        if size == 0:
            break
        chunks.append(seq[start:start + size])
        start += size
    return chunks


def run_fit_candidates(x, y_abs, z_mcd, fc, model_param_pairs, processes=4):
    """Fit all generated candidates and collect successful lmfit result bundles."""
    processes = _resolve_process_count(processes)
    _, correction_note = harmonic_correction_details(fc)
    results = BfResult(x, y_abs, z_mcd, processing_note=correction_note)
    chunks = _split_into_n(model_param_pairs, processes)

    start = time.perf_counter()
    if processes == 1:
        results_lists = [
            fit_worker("Worker-1", x, z_mcd, y_abs, fc, model_param_pairs)
        ]
    else:
        results_lists = Parallel(n_jobs=processes, backend="loky")(
            delayed(fit_worker)(f"Worker-{i + 1}", x, z_mcd, y_abs, fc, chunk)
            for i, chunk in enumerate(chunks)
        )

    for sublist in results_lists:
        for r in sublist:
            results.add_result(r)

    elapsed = time.perf_counter() - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\n####################################### -> Start of time print")
    print(f"Elapsed time: {minutes} min {seconds:.2f} sec for {processes} workers and {len(model_param_pairs)} fits")
    print("####################################### -> End of time print")
    return results


def fit_models_headless(mcd_df, fc=None, transitions=None, processes=1):
    """
    Batch-safe fitting entry point.

    This path intentionally bypasses Qt dialogs but reuses the same automatic guessing,
    candidate generation, and final fitting code as the GUI path.
    """
    fc = fc or FitConfig()
    x, y_abs, z_mcd, mcd_noise = prepare_fit_arrays(mcd_df, fc)
    if transitions is None:
        transitions = auto_guess_transitions(x, y_abs, z_mcd, fc, use_cache=False)
    model_param_pairs, fc = build_fit_candidates_from_transitions(transitions, fc)
    return run_fit_candidates(x, y_abs, z_mcd, fc, model_param_pairs, processes=processes), fc, transitions


def fit_models(mcd_df, fc = None, processes = 4):
    """
    Fit all brute-force-generated models to the data in df.

    percentage_range: +/- percent range around initial value for sigma, center, amplitude
    """
    fc = fc or FitConfig()
    x, y_abs, z_mcd, mcd_noise = prepare_fit_arrays(mcd_df, fc)
    model_param_pairs, fc = brute_force_models(x, y_abs, z_mcd, fc, mcd_noise=mcd_noise)
    results = run_fit_candidates(x, y_abs, z_mcd, fc, model_param_pairs, processes=processes)

    # --- Show results window before returning ---
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    win = MainResultWindow(bfResult=results, df_fc=(mcd_df, fc))
    win.show()

    app.exec()

    return results

def fit_worker(name, x, z_mcd, y_abs, fc, model_param_pairs):
    count = 0
    total = len(model_param_pairs)

    out = []

    def _clamp_initial_values(params):
        """Keep optimizer starting values inside their declared finite bounds."""
        params.update_constraints()
        for parameter in params.values():
            if parameter.expr is not None or not parameter.vary:
                continue
            value = float(parameter.value)
            if np.isfinite(parameter.min):
                value = max(value, float(parameter.min))
            if np.isfinite(parameter.max):
                value = min(value, float(parameter.max))
            parameter.set(value=value)

    def _result_view(model, params, data, best_fit):
        data_array = np.asarray(data, dtype=float)
        best_fit_array = np.asarray(best_fit, dtype=float)
        residual = data_array - best_fit_array
        chisqr = max(float(np.sum(residual ** 2)), gaussianModels.TINY)
        ndata = max(1, residual.size)
        # These per-spectrum metrics remain comparable to historical output. Joint
        # fitting shares variables across panels, so the variable count is an
        # approximation used for display/ranking rather than a joint-fit statistic.
        nvarys = sum(1 for parameter in params.values() if parameter.vary and parameter.expr is None)
        dof = max(1, ndata - nvarys)
        redchi = chisqr / dof
        bic = ndata * np.log(chisqr / ndata) + nvarys * np.log(ndata)
        return LinkedModelResult(
            model=model,
            params=params.copy(),
            data=data_array,
            best_fit=best_fit_array,
            residual=residual,
            redchi=float(redchi),
            bic=float(bic),
        )

    def _fit_linked(candidate, model_mcd, params_mcd, model_abs, params_abs):
        """Jointly fit ABS/MCD with bounded cross-spectrum constraints for paired peaks."""
        joint_params = params_abs.copy()
        joint_params.update(params_mcd)

        center_limit = float(fc.PAIR_CENTER_OFFSET)
        ratio_delta = float(fc.PAIR_SIGMA_RATIO_DELTA)
        ratio_min = max(gaussianModels.TINY, 1.0 - ratio_delta)
        ratio_max = 1.0 + ratio_delta

        for spec in candidate.transition_specs:
            if spec.status != "paired" or not spec.abs_prefix or not spec.mcd_prefix:
                continue

            abs_center_name = f"{spec.abs_prefix}center"
            abs_sigma_name = f"{spec.abs_prefix}sigma"
            mcd_center_name = f"{spec.mcd_prefix}center"
            mcd_sigma_name = f"{spec.mcd_prefix}sigma"
            center_offset_name = f"{spec.mcd_prefix}_center_offset"
            sigma_ratio_name = f"{spec.mcd_prefix}_sigma_ratio"

            initial_offset = joint_params[mcd_center_name].value - joint_params[abs_center_name].value
            initial_offset = float(np.clip(initial_offset, -center_limit, center_limit))
            initial_ratio = joint_params[mcd_sigma_name].value / max(
                joint_params[abs_sigma_name].value,
                gaussianModels.TINY,
            )
            initial_ratio = float(np.clip(initial_ratio, ratio_min, ratio_max))

            if center_limit > 0:
                joint_params.add(
                    center_offset_name,
                    value=initial_offset,
                    min=-center_limit,
                    max=center_limit,
                )
            else:
                joint_params.add(center_offset_name, value=0.0, vary=False)
            if ratio_delta > 0:
                joint_params.add(
                    sigma_ratio_name,
                    value=initial_ratio,
                    min=ratio_min,
                    max=ratio_max,
                )
            else:
                joint_params.add(sigma_ratio_name, value=1.0, vary=False)
            # The offset/ratio parameters now own the cross-spectrum bounds. Clear
            # the old independent MCD bounds so the expression-derived starting
            # value cannot be rejected as outside a stale center/sigma interval.
            joint_params[mcd_center_name].set(
                min=-np.inf,
                max=np.inf,
                expr=f"{abs_center_name} + {center_offset_name}",
            )
            joint_params[mcd_sigma_name].set(
                min=-np.inf,
                max=np.inf,
                expr=f"{abs_sigma_name} * {sigma_ratio_name}",
            )

        x_array = np.asarray(x, dtype=float)
        mcd_data = np.asarray(z_mcd, dtype=float)
        abs_data = np.asarray(y_abs, dtype=float)
        mcd_scale = max(float(np.nanmax(np.abs(mcd_data))), gaussianModels.TINY)
        abs_scale = max(float(np.nanmax(np.abs(abs_data))), gaussianModels.TINY)

        def objective(params):
            mcd_fit = np.asarray(model_mcd.eval(params=params, x=x_array), dtype=float)
            abs_fit = np.asarray(model_abs.eval(params=params, x=x_array), dtype=float)
            return np.concatenate(((mcd_data - mcd_fit) / mcd_scale, (abs_data - abs_fit) / abs_scale))

        _clamp_initial_values(joint_params)
        joint_result = minimize(objective, joint_params, method="least_squares")
        fitted_params = joint_result.params
        mcd_best_fit = model_mcd.eval(params=fitted_params, x=x_array)
        abs_best_fit = model_abs.eval(params=fitted_params, x=x_array)
        return (
            _result_view(model_mcd, fitted_params, mcd_data, mcd_best_fit),
            _result_view(model_abs, fitted_params, abs_data, abs_best_fit),
        )

    def _local_abs_peak_height_cap(center, sigma):
        """
        Estimate the observed ABS peak height near one initial component guess.

        The ABS amplitude parameter is an area coefficient, so constraining it directly
        does not constrain the visible peak height once sigma is allowed to vary.
        We therefore cap each component against the measured ABS intensity in a local
        window around its initial center/width.
        """
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y_abs, dtype=float)
        safe_sigma = max(abs(float(sigma)), gaussianModels.TINY)
        half_window = abs(float(fc.DELTA_CTR)) + safe_sigma
        in_window = np.abs(x_array - float(center)) <= half_window

        if not np.any(in_window):
            return max(float(np.nanmax(y_array)), gaussianModels.TINY)

        local_max = float(np.nanmax(y_array[in_window]))
        if not np.isfinite(local_max) or local_max <= 0.0:
            return max(float(np.nanmax(y_array)), gaussianModels.TINY)
        return max(local_max, gaussianModels.TINY)

    for candidate in model_param_pairs:
        model_mcd = candidate.mcd_model
        params_mcd = candidate.mcd_params
        model_abs = candidate.abs_model
        params_abs = candidate.abs_params

        if model_mcd is None or model_abs is None or params_mcd is None or params_abs is None:
            continue

        # Largest absolute value in the measured MCD/ABS data (used for scaling)
        max_data_value_mcd = float(np.nanmax(np.abs(z_mcd)) or 1e-12)
        max_data_value_abs = float(np.nanmax(np.abs(y_abs)) or 1e-12)

        # --- MCD Component Scaling ---
        # Amplitude bounds are expressed in model-amplitude units, not peak-height
        # units. For each component shape, evaluate a unit-amplitude curve so we can
        # translate "a few times the measured signal height" into a safe amplitude
        # bound for the optimizer.
        component_unit_max_mcd = {}
        for param_name in params_mcd:
            if param_name.endswith('amplitude'):
                prefix = param_name[:-len('amplitude')]
                temp_params = params_mcd.copy()
                temp_params[param_name].set(value=1.0)

                components = model_mcd.eval_components(x=x, params=temp_params)
                comp_data = components.get(prefix)

                component_unit_max_mcd[prefix] = float(np.nanmax(np.abs(comp_data))) or 1e-12

        abs_height_caps = {}
        for param_name in params_abs:
            if param_name.endswith('amplitude'):
                prefix = param_name[:-len('amplitude')]
                center_name = f"{prefix}center"
                sigma_name = f"{prefix}sigma"
                abs_height_caps[prefix] = _local_abs_peak_height_cap(
                    params_abs[center_name].value,
                    params_abs[sigma_name].value,
                )

        # --- Set bounds for MCD parameters ---
        for p_name, p in params_mcd.items():
            delta_sig = fc.DELTA_SIGMA
            delta_ctr = fc.DELTA_CTR

            if p_name.endswith('sigma'):
                p.set(min=p.value - delta_sig, max=p.value + delta_sig, vary=True)

            elif p_name.endswith('center'):
                p.set(min=p.value - delta_ctr, max=p.value + delta_ctr, vary=True)

            elif p_name.endswith('amplitude'):
                prefix = p_name[:-len('amplitude')]
                unit_max = component_unit_max_mcd.get(prefix, 1e-12)
                allowed_amp = fc.AMPLITUDE_SCALE_LIMIT * (max_data_value_mcd / unit_max)
                p.set(min=-allowed_amp, max=allowed_amp, vary=True)

        # --- Set bounds for ABS parameters ---
        for p_name, p in list(params_abs.items()):
            delta_sig = fc.DELTA_SIGMA
            delta_ctr = fc.DELTA_CTR

            if p_name.endswith('sigma'):
                p.set(min=max(gaussianModels.TINY, p.value - delta_sig), max=p.value + delta_sig, vary=True)

            elif p_name.endswith('center'):
                p.set(min=p.value - delta_ctr, max=p.value + delta_ctr, vary=True)

            elif p_name.endswith('amplitude'):
                prefix = p_name[:-len('amplitude')]
                sigma_name = f"{prefix}sigma"
                height_scale_name = f"{prefix}_height_scale"
                height_cap = abs_height_caps.get(prefix, max_data_value_abs)
                safe_sigma = max(float(params_abs[sigma_name].value), gaussianModels.TINY)
                initial_height = max(float(p.value) / (safe_sigma * gaussianModels.S2PI), 0.0)
                initial_scale = min(1.0, initial_height / max(height_cap, gaussianModels.TINY))

                # Drive ABS area-amplitude from a bounded peak-height fraction so sigma
                # changes cannot produce a Gaussian taller than the measured spectrum.
                if height_scale_name not in params_abs:
                    params_abs.add(height_scale_name, value=initial_scale, min=0.0, max=1.0, vary=True)
                else:
                    params_abs[height_scale_name].set(value=initial_scale, min=0.0, max=1.0, vary=True)
                p.set(
                    expr=(
                        f"{height_scale_name} * {height_cap:.17g} * "
                        f"{sigma_name} * {gaussianModels.S2PI:.17g}"
                    )
                )

        try:
            if fc.PAIR_CONSTRAINT_MODE == "bounded_offsets":
                res_mcd, res_abs = _fit_linked(
                    candidate,
                    model_mcd,
                    params_mcd,
                    model_abs,
                    params_abs,
                )
            else:
                _clamp_initial_values(params_mcd)
                _clamp_initial_values(params_abs)
                res_mcd = model_mcd.fit(np.float32(z_mcd), params_mcd, x=np.float32(x))
                res_abs = model_abs.fit(np.float32(y_abs), params_abs, x=np.float32(x))

            count += 1
            print(f"Process {name}: fit {count} of {total}")
            out.append(
                FitBundle(
                    mcd_result=res_mcd,
                    abs_result=res_abs,
                    transition_specs=list(candidate.transition_specs),
                )
            )

        except Exception as e:
            print(f"Exception raised while fitting: {e}\n")
            continue

    return out
