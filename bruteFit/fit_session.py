from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lmfit import Model

from . import gaussianModels
from .fitConfig import FitConfig
from .transition_matching import PeakGuess, TransitionGuess, TransitionModelSpec


SESSION_FORMAT = "brutefit-fit-session"
SESSION_VERSION = 1


@dataclass
class LoadedFitSession:
    dataframe: pd.DataFrame
    fit_config: FitConfig
    bf_result: object
    ranking: dict
    source_path: Path


def _peak_to_dict(peak: PeakGuess | None):
    return None if peak is None else asdict(peak)


def _peak_from_dict(payload):
    return None if payload is None else PeakGuess(**payload)


def _spec_to_dict(spec: TransitionModelSpec) -> dict:
    return {
        "transition_id": spec.transition_id,
        "status": spec.status,
        "abs_peak": _peak_to_dict(spec.abs_peak),
        "mcd_peak": _peak_to_dict(spec.mcd_peak),
        "match_distance": spec.match_distance,
        "abs_prefix": spec.abs_prefix,
        "mcd_prefix": spec.mcd_prefix,
        "ratio_label": spec.ratio_label,
    }


def _spec_from_dict(payload: dict) -> TransitionModelSpec:
    return TransitionModelSpec(
        transition_id=payload["transition_id"],
        status=payload["status"],
        abs_peak=_peak_from_dict(payload.get("abs_peak")),
        mcd_peak=_peak_from_dict(payload.get("mcd_peak")),
        match_distance=payload.get("match_distance"),
        abs_prefix=payload.get("abs_prefix"),
        mcd_prefix=payload.get("mcd_prefix"),
        ratio_label=payload.get("ratio_label"),
    )


def _fitconfig_to_dict(fc: FitConfig) -> dict:
    return {
        item.name: getattr(fc, item.name)
        for item in fields(fc)
        if not item.name.startswith("_")
    }


def _dataframe_payload(df: pd.DataFrame) -> dict:
    # pandas handles numpy scalars and converts NaN/inf values to portable JSON nulls.
    return json.loads(df.to_json(orient="split", double_precision=15))


def _dataframe_from_payload(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        data=payload["data"],
        columns=payload["columns"],
        index=payload.get("index"),
    )


def _dataframe_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _visible_parameter_values(result) -> dict[str, float]:
    model_names = set(getattr(result.model, "param_names", ()))
    return {
        name: float(parameter.value)
        for name, parameter in result.params.items()
        if name in model_names
    }


def _result_payload(result) -> dict:
    def finite_or_none(value):
        value = float(value)
        return value if np.isfinite(value) else None

    return {
        "parameters": _visible_parameter_values(result),
        "redchi": finite_or_none(result.redchi),
        "bic": finite_or_none(result.bic),
    }


def _array_to_json(values) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values, dtype=float)]


def save_fit_session(
    path: str | Path,
    dataframe: pd.DataFrame,
    fit_config: FitConfig,
    bf_result,
    bundle,
    ranking: dict,
) -> Path:
    """Save one explicitly selected fit, including enough state to display or refit it."""
    path = Path(path)
    dataframe_payload = _dataframe_payload(dataframe)
    payload = {
        "format": SESSION_FORMAT,
        "version": SESSION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataframe_sha256": _dataframe_sha256(dataframe_payload),
        "dataframe": dataframe_payload,
        "preferred_save_directory": str(dataframe.attrs.get("brutefit_save_dir", "")),
        "fit_config": _fitconfig_to_dict(fit_config),
        "ranking": dict(ranking),
        "transition_specs": [_spec_to_dict(spec) for spec in bundle.transition_specs],
        "mcd_result": _result_payload(bundle.mcd_result),
        "abs_result": _result_payload(bundle.abs_result),
        "plot_arrays": {
            "x": _array_to_json(bf_result.dataX),
            "abs": _array_to_json(bf_result.dataY),
            "mcd": _array_to_json(bf_result.dataZ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _build_model(specs: list[TransitionModelSpec], source: str):
    model = None
    for spec in specs:
        prefix = spec.abs_prefix if source == "abs" else spec.mcd_prefix
        if not prefix:
            continue
        if source == "abs" or prefix.startswith("B"):
            function = gaussianModels.stable_gaussian_sigma
        else:
            function = gaussianModels.stable_gaussian_derivative_sigma
        component = Model(function, prefix=prefix)
        model = component if model is None else model + component
    if model is None:
        raise ValueError(f"Saved session has no {source.upper()} model components.")
    return model


def _restore_result(model, payload: dict, x, data):
    from .dataFitting import LinkedModelResult

    params = model.make_params()
    for name in model.param_names:
        if name not in payload["parameters"]:
            raise ValueError(f"Saved session is missing fitted parameter '{name}'.")
        params[name].set(value=float(payload["parameters"][name]), vary=False)
    best_fit = np.asarray(model.eval(params=params, x=x), dtype=float)
    data = np.asarray(data, dtype=float)
    residual = data - best_fit
    finite_residual = residual[np.isfinite(residual)]
    chisqr = max(float(np.sum(finite_residual ** 2)), gaussianModels.TINY)
    ndata = max(1, finite_residual.size)
    redchi = payload.get("redchi")
    bic = payload.get("bic")
    return LinkedModelResult(
        model=model,
        params=params,
        data=data,
        best_fit=best_fit,
        residual=residual,
        redchi=float(redchi) if redchi is not None else chisqr / ndata,
        bic=float(bic) if bic is not None else ndata * np.log(chisqr / ndata),
    )


def _fitted_transitions(specs, mcd_result, abs_result) -> list[TransitionGuess]:
    """Convert final parameters into editable/refittable transition starting guesses."""
    transitions = []
    for spec in specs:
        abs_peak = spec.abs_peak
        mcd_peak = spec.mcd_peak
        if spec.abs_prefix:
            prefix = spec.abs_prefix
            amp = abs_result.params[f"{prefix}amplitude"].value
            ctr = abs_result.params[f"{prefix}center"].value
            sig = abs_result.params[f"{prefix}sigma"].value
            abs_peak = PeakGuess(
                "abs", ctr, amp, sig,
                height=gaussianModels.component_peak_height(amp, ctr, sig),
                origin="saved_fit",
            )
        if spec.mcd_prefix:
            prefix = spec.mcd_prefix
            label = prefix[0] if prefix[0] in {"A", "B"} else None
            amp = mcd_result.params[f"{prefix}amplitude"].value
            ctr = mcd_result.params[f"{prefix}center"].value
            sig = mcd_result.params[f"{prefix}sigma"].value
            mcd_peak = PeakGuess(
                "mcd", ctr, amp, sig,
                height=gaussianModels.component_peak_height(amp, ctr, sig, label=label),
                origin="saved_fit",
                label=label,
            )
        match_distance = (
            abs(abs_peak.center - mcd_peak.center)
            if abs_peak is not None and mcd_peak is not None
            else None
        )
        transitions.append(TransitionGuess(spec.transition_id, abs_peak, mcd_peak, match_distance))
    return transitions


def load_fit_session(path: str | Path) -> LoadedFitSession:
    """Restore a saved fit for immediate plotting and optional parameter-seeded refitting."""
    from .dataFitting import BfResult, FitBundle, harmonic_correction_details

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != SESSION_FORMAT:
        raise ValueError("This is not a bruteFit fit-session file.")
    if int(payload.get("version", -1)) != SESSION_VERSION:
        raise ValueError(f"Unsupported fit-session version: {payload.get('version')}")

    dataframe_payload = payload["dataframe"]
    if _dataframe_sha256(dataframe_payload) != payload.get("dataframe_sha256"):
        raise ValueError("The saved dataframe failed its integrity check.")
    dataframe = _dataframe_from_payload(dataframe_payload)
    preferred_save_directory = str(payload.get("preferred_save_directory", ""))
    if not preferred_save_directory or not Path(preferred_save_directory).is_dir():
        preferred_save_directory = str(path.parent.resolve())
    dataframe.attrs["brutefit_save_dir"] = preferred_save_directory
    config_values = payload["fit_config"]
    valid_config_names = {item.name for item in fields(FitConfig) if not item.name.startswith("_")}
    fit_config = FitConfig(**{k: v for k, v in config_values.items() if k in valid_config_names})

    specs = [_spec_from_dict(item) for item in payload["transition_specs"]]
    arrays = payload["plot_arrays"]
    x = np.asarray(arrays["x"], dtype=float)
    abs_data = np.asarray(arrays["abs"], dtype=float)
    mcd_data = np.asarray(arrays["mcd"], dtype=float)
    mcd_result = _restore_result(_build_model(specs, "mcd"), payload["mcd_result"], x, mcd_data)
    abs_result = _restore_result(_build_model(specs, "abs"), payload["abs_result"], x, abs_data)

    _, correction_note = harmonic_correction_details(fit_config)
    bf_result = BfResult(x, abs_data, mcd_data, processing_note=correction_note)
    bf_result.add_result(FitBundle(mcd_result, abs_result, specs))
    fitted_transitions = _fitted_transitions(specs, mcd_result, abs_result)
    fit_config.set_current_peaks(fitted_transitions, [], "auto")
    return LoadedFitSession(
        dataframe=dataframe,
        fit_config=fit_config,
        bf_result=bf_result,
        ranking=dict(payload.get("ranking", {})),
        source_path=path,
    )
