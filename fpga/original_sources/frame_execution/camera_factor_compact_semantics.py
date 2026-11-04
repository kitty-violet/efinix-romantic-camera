"""Reuse frozen audits for the independent compact candidate."""
from pathlib import Path
COMPACT_SOURCE=Path(__file__).with_name('camera_factor_balanced_semantics.py')
code=COMPACT_SOURCE.read_text().replace('camera_factor_balanced','camera_factor_compact').replace('camera-factor-balanced','camera-factor-compact')
exec(compile(code,str(COMPACT_SOURCE)+':compact','exec'),globals())
