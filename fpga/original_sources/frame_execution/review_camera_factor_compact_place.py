"""Reuse frozen audits for the independent compact candidate."""
from pathlib import Path
COMPACT_SOURCE=Path(__file__).with_name('review_camera_factor_balanced_place.py')
code=COMPACT_SOURCE.read_text().replace('camera_factor_balanced','camera_factor_compact').replace('camera-factor-balanced','camera-factor-compact')
exec(compile(code,str(COMPACT_SOURCE)+':compact','exec'),globals())
