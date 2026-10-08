"""User-facing MCD acquisition script.

Run without arguments for a simple Tkinter GUI, or pass CLI options for a
repeatable dry-run or live scan.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from acquisition.config import load_config
from acquisition.scan_runner import (
    ScanRequest,
    print_post_scan_summary,
    print_scan_summary,
    run_scan,
    validate_scan_request,
)


def _config_error(exc: Exception) -> str:
    return (
        f"Missing or invalid config.yaml: {exc}\n"
        "Check that config.yaml exists in the project folder and contains YAML sections "
        "for zurich, cm110, pem, scan_defaults, and absorption."
    )


def validate_config(config: Dict[str, Any]) -> None:
    required_sections = ["zurich", "cm110", "pem", "scan_defaults", "absorption"]
    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config.yaml section '{section}' must be present and be a mapping")

    zurich = config["zurich"]
    if str(zurich.get("reference_aux_channel", "auxin0")).lower() != "auxin0":
        print(
            "Warning: reference_aux_channel is not auxin0. Confirmed LabOne Aux In 1 maps to Python auxin0."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCD Python acquisition")
    parser.add_argument("--live", action="store_true", help="Run live hardware acquisition. Default is dry-run.")
    parser.add_argument("--start", type=float, help="Start wavelength in nm.")
    parser.add_argument("--end", type=float, help="End wavelength in nm.")
    parser.add_argument("--step", type=float, help="Wavelength step in nm.")
    parser.add_argument("--points", type=int, help="Points per wavelength.")
    parser.add_argument("--base-name", help="Sample/base name for output files.")
    parser.add_argument("--scan-type", choices=["pos", "neg", "ref"], help="Scan type: pos, neg, or ref.")
    parser.add_argument("--sr830-sensitivity", type=float, help="SR830 sensitivity in volts.")
    parser.add_argument("--settle-time", type=float, help="Wavelength settle delay in seconds.")
    parser.add_argument("--output-dir", help="Output directory.")
    parser.add_argument("--record-intensity", action="store_true", help="Write I_sample.csv and I_avg.csv.")
    parser.add_argument("--operator", default="", help="Operator name for notes metadata.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    return parser


def resolve_output_folder(default_output_dir: str) -> str:
    """Resolve the configured output folder to an absolute path."""
    default_path = Path(default_output_dir).expanduser()
    if not default_path.is_absolute():
        default_path = Path.cwd() / default_path
    return str(default_path)


def gui_values_to_request(values: Dict[str, Any], config: Dict[str, Any], config_path: str) -> ScanRequest:
    """Convert GUI field values into the existing ScanRequest object."""
    defaults = config["scan_defaults"]

    def text_value(key: str, default: str) -> str:
        value = str(values.get(key, "")).strip()
        return value if value else default

    def float_value(key: str, default: float) -> float:
        return float(text_value(key, str(default)))

    def int_value(key: str, default: int) -> int:
        return int(text_value(key, str(default)))

    return ScanRequest(
        live=True,
        base_name=text_value("base_name", str(defaults.get("base_name", "test"))),
        scan_type=text_value("scan_type", str(defaults.get("scan_type", "pos"))).lower(),
        start_nm=float_value("start_nm", float(defaults.get("start_nm", 400))),
        end_nm=float_value("end_nm", float(defaults.get("end_nm", 450))),
        step_nm=float_value("step_nm", float(defaults.get("step_nm", 2))),
        points=int_value("points", int(defaults.get("points", defaults.get("number_of_points", 100)))),
        sr830_sensitivity_v=float_value("sr830_sensitivity_v", float(defaults.get("sr830_sensitivity_v", 0.1))),
        settle_time_s=float_value("settle_time_s", float(defaults.get("settle_time_s", 2))),
        output_dir=text_value("output_dir", str(defaults.get("output_dir", "data"))),
        record_intensity=bool(values.get("record_intensity", False)),
        operator="",
        config_path=config_path,
    )


def request_from_args(args: argparse.Namespace, config: Dict[str, Any]) -> ScanRequest:
    defaults = config["scan_defaults"]
    return ScanRequest(
        live=bool(args.live),
        base_name=args.base_name or str(defaults.get("base_name", "test")),
        scan_type=args.scan_type or str(defaults.get("scan_type", "pos")),
        start_nm=float(args.start if args.start is not None else defaults.get("start_nm", 400)),
        end_nm=float(args.end if args.end is not None else defaults.get("end_nm", 450)),
        step_nm=float(args.step if args.step is not None else defaults.get("step_nm", 2)),
        points=int(args.points if args.points is not None else defaults.get("points", defaults.get("number_of_points", 100))),
        sr830_sensitivity_v=float(
            args.sr830_sensitivity
            if args.sr830_sensitivity is not None
            else defaults.get("sr830_sensitivity_v", 0.1)
        ),
        settle_time_s=float(
            args.settle_time if args.settle_time is not None else defaults.get("settle_time_s", 2)
        ),
        output_dir=args.output_dir or str(defaults.get("output_dir", "data")),
        record_intensity=bool(args.record_intensity),
        operator=args.operator or "",
        config_path=args.config,
    )


def launch_gui(config: Dict[str, Any], config_path: str) -> int:
    """Open the simple user-facing scan form."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:
        print(f"Tkinter GUI could not be started: {exc}", file=sys.stderr)
        return 1

    defaults = config["scan_defaults"]
    absorption = config["absorption"]

    root = tk.Tk()
    root.title("CD/MCD Scan")
    root.resizable(False, False)

    fields: Dict[str, tk.StringVar] = {
        "base_name": tk.StringVar(value=""),
        "scan_type": tk.StringVar(value=""),
        "start_nm": tk.StringVar(value=""),
        "end_nm": tk.StringVar(value=""),
        "step_nm": tk.StringVar(value=""),
        "points": tk.StringVar(value=""),
        "sr830_sensitivity_v": tk.StringVar(value="0.5"),
        "settle_time_s": tk.StringVar(value="2"),
        "output_dir": tk.StringVar(value=""),
    }
    record_intensity_var = tk.BooleanVar(value=False)
    status_var = tk.StringVar(value="Ready.")

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")

    row = 0

    def add_entry(label: str, key: str) -> None:
        nonlocal row
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(frame, textvariable=fields[key], width=38).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

    add_entry("Sample/base name", "base_name")

    ttk.Label(frame, text="Scan type").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
    ttk.Combobox(frame, textvariable=fields["scan_type"], values=("pos", "neg", "ref"), width=35, state="readonly").grid(
        row=row, column=1, sticky="ew", pady=4
    )
    row += 1

    add_entry("Start wavelength nm", "start_nm")
    add_entry("End wavelength nm", "end_nm")
    add_entry("Step size nm", "step_nm")
    add_entry("Points per wavelength", "points")
    add_entry("SR830 sensitivity in volts", "sr830_sensitivity_v")

    ttk.Checkbutton(frame, text="Record raw intensity", variable=record_intensity_var).grid(
        row=row, column=1, sticky="w", pady=4
    )
    row += 1

    add_entry("Settle time seconds", "settle_time_s")

    ttk.Label(frame, text="Output folder").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
    output_frame = ttk.Frame(frame)
    output_frame.grid(row=row, column=1, sticky="ew", pady=4)
    ttk.Entry(output_frame, textvariable=fields["output_dir"], width=28).grid(row=0, column=0, sticky="ew")

    def browse_output_dir() -> None:
        selected = filedialog.askdirectory(
            title="Choose CD/MCD scan output folder",
            initialdir=fields["output_dir"].get() or resolve_output_folder(str(defaults.get("output_dir", "data"))),
            mustexist=False,
        )
        if selected:
            fields["output_dir"].set(selected)

    ttk.Button(output_frame, text="Browse", command=browse_output_dir).grid(row=0, column=1, padx=(6, 0))
    output_frame.columnconfigure(0, weight=1)
    row += 1

    ttk.Label(frame, textvariable=status_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))
    row += 1

    button_frame = ttk.Frame(frame)
    button_frame.grid(row=row, column=0, columnspan=2, sticky="e", pady=(8, 0))

    start_button = ttk.Button(button_frame, text="Start Scan")
    start_button.grid(row=0, column=0, padx=(0, 6))
    ttk.Button(button_frame, text="Close", command=root.destroy).grid(row=0, column=1)

    def values_from_form() -> Dict[str, Any]:
        return {
            "base_name": fields["base_name"].get(),
            "scan_type": fields["scan_type"].get(),
            "start_nm": fields["start_nm"].get(),
            "end_nm": fields["end_nm"].get(),
            "step_nm": fields["step_nm"].get(),
            "points": fields["points"].get(),
            "sr830_sensitivity_v": fields["sr830_sensitivity_v"].get(),
            "settle_time_s": fields["settle_time_s"].get(),
            "output_dir": fields["output_dir"].get(),
            "record_intensity": record_intensity_var.get(),
        }

    def scan_worker(request: ScanRequest) -> None:
        try:
            result = run_scan(config, request)
        except Exception as exc:
            error_message = format_common_error(exc)
            root.after(0, lambda: start_button.configure(state="normal"))
            root.after(0, lambda: status_var.set("Scan failed."))
            root.after(0, lambda message=error_message: messagebox.showerror("Scan failed", message))
            return

        def finish() -> None:
            print_post_scan_summary(result, request.record_intensity)
            status_var.set("Scan complete.")
            start_button.configure(state="normal")
            messagebox.showinfo("Scan complete", "CD/MCD scan complete.")

        root.after(0, finish)

    def start_scan() -> None:
        try:
            request = gui_values_to_request(values_from_form(), config, config_path)
            wavelengths = validate_scan_request(request)
        except Exception as exc:
            messagebox.showerror("Invalid scan settings", format_common_error(exc))
            return

        print_scan_summary(config, request, wavelengths)
        status_var.set("Scan running...")
        start_button.configure(state="disabled")
        threading.Thread(target=scan_worker, args=(request,), daemon=True).start()

    start_button.configure(command=start_scan)
    root.mainloop()
    return 0


def format_common_error(exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()
    if "connection refused" in lower or "ziapi" in lower or "data server" in lower:
        return (
            "LabOne Data Server connection failed. Start LabOne Data Server, confirm it is listening on "
            "localhost:8004, then retry.\n"
            f"Original error: {message}"
        )
    if "already in use" in lower or "in use" in lower:
        return (
            "DEV4388 appears to already be in use. Close/disconnect it in LabOne or another Python session, "
            "then retry.\n"
            f"Original error: {message}"
        )
    if "com4" in lower or "access is denied" in lower:
        return (
            "COM4 is unavailable. LabVIEW or another serial program may be holding the CM110 monochromator.\n"
            f"Original error: {message}"
        )
    if "com1" in lower:
        return (
            "COM1 is unavailable. Check PEM power/cable/port assignment and close other serial programs.\n"
            f"Original error: {message}"
        )
    if "too close to zero" in lower or "near-zero" in lower:
        return (
            "Near-zero reference aux voltage. Confirm the detector/reference signal is on LabOne Aux In 1 "
            "(Python auxin0), and verify SR830 sensitivity.\n"
            f"Original error: {message}"
        )
    return message


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    try:
        config = load_config(config_path)
        validate_config(config)
    except Exception as exc:
        print(_config_error(exc), file=sys.stderr)
        return 2

    try:
        if not argv:
            return launch_gui(config, str(config_path))

        request = request_from_args(args, config)
        wavelengths = validate_scan_request(request)
        print_scan_summary(config, request, wavelengths)
        result = run_scan(config, request)
        print_post_scan_summary(result, request.record_intensity)
    except KeyboardInterrupt:
        print("\nScan cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(format_common_error(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
