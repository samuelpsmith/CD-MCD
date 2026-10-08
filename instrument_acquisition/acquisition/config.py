"""Shared configuration utilities for MCD acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml in the project root.
    
    Returns:
        Dictionary containing parsed YAML configuration.
    
    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config file is not a valid YAML mapping at the top level.
    """
    if config_path is None:
        # Try to find config relative to this module
        config_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError('Configuration file must contain a YAML mapping at the top level')

    return config


def get_scan_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate scan_defaults section from config.
    
    Args:
        config: Parsed configuration dictionary.
    
    Returns:
        The scan_defaults dictionary.
    
    Raises:
        ValueError: If scan_defaults is not defined or is not a dict.
    """
    scan_defaults = config.get('scan_defaults')
    if not isinstance(scan_defaults, dict):
        raise ValueError('scan_defaults must be defined as a mapping in config.yaml')
    return scan_defaults


def resolve_parameter(
    cli_value: Optional[Any],
    config_dict: Dict[str, Any],
    config_key: str,
    default: Any,
    param_type: type = float,
    param_name: str = 'parameter',
) -> Any:
    """
    Resolve a parameter from CLI argument, config file, or default value.
    
    CLI argument takes precedence, then config, then default.
    
    Args:
        cli_value: Value from command-line argument (None if not provided).
        config_dict: Configuration dictionary (e.g., scan_defaults).
        config_key: Key to look up in config_dict.
        default: Default value if not in CLI or config.
        param_type: Type to convert the resolved value to.
        param_name: Name of parameter (for error messages).
    
    Returns:
        Resolved parameter value, converted to param_type.
    
    Raises:
        ValueError: If conversion fails.
    """
    if cli_value is not None:
        try:
            return param_type(cli_value)
        except (TypeError, ValueError) as e:
            raise ValueError(f'Failed to convert {param_name} CLI argument: {e}') from e
    
    config_value = config_dict.get(config_key)
    if config_value is not None:
        try:
            return param_type(config_value)
        except (TypeError, ValueError) as e:
            raise ValueError(f'Failed to convert {param_name} from config: {e}') from e
    
    return param_type(default)


def print_scan_config(
    reference_aux_channel: str,
    sr830_sensitivity: float,
) -> None:
    """
    Print scan configuration details to stdout.
    
    Used before live hardware scans to verify settings.
    
    Args:
        reference_aux_channel: Reference aux channel name (e.g., 'auxin0').
        sr830_sensitivity: SR830 sensitivity in volts.
    """
    print(f'\n=== Scan Configuration ===')
    print(f'Reference aux channel: {reference_aux_channel}')
    print(f'SR830 sensitivity: {sr830_sensitivity} V')
    print(f'Normalization formula: X_ratio = X / (ref_aux * {sr830_sensitivity} / 10)')
    print(f'                      Y_ratio = Y / (ref_aux * {sr830_sensitivity} / 10)')
    print(f'==========================\n')
