"""Compact-map binding; unchanged complete-tile and release requirements."""
from pathlib import Path
COMPACT_ISLAND_BASE=Path(__file__).with_name('camera_factor_island_flow.py')
code=COMPACT_ISLAND_BASE.read_text().replace('camera_factor_balanced','camera_factor_compact').replace('camera_factor_island','camera_factor_compact_island')
exec(compile(code,str(COMPACT_ISLAND_BASE)+':compact','exec'),globals())
