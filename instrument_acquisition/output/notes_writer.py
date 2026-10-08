"""Write human-readable acquisition notes."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def write_notes_file(
    base_name: str,
    scan_type: str,
    output_dir: str,
    b_field_t: float,
    path_length_m: float,
    experiment_note: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Write an acquisition notes file.

    The original arguments are kept for runtime diagnostic scripts. Newer scan
    code can pass metadata for a more complete research record.
    """
    normalized = scan_type.lower()
    if normalized not in {'pos', 'neg', 'ref'}:
        raise ValueError('scan_type must be "pos", "neg", or "ref"')

    os.makedirs(output_dir, exist_ok=True)
    notes_path = os.path.join(output_dir, f'{base_name}_{normalized}_notes')

    lines = [
        f'Field: {b_field_t} T',
        f'Path Length: {path_length_m} m',
        f'Notes: {experiment_note}',
        f'Scan type: {normalized}',
    ]
    if metadata:
        lines.append('')
        lines.append('Acquisition metadata:')
        for key, value in metadata.items():
            lines.append(f'{key}: {value}')
    content = '\n'.join(lines) + '\n'

    with open(notes_path, 'w', encoding='utf-8') as notes_file:
        notes_file.write(content)

    return notes_path
