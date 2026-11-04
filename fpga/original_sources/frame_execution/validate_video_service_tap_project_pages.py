"""Preserve the full observation oracle and prove its current APB boundary."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_same_edge_pages.py');code=source.read_text()
assert code.count('from same_edge_observation_binding import bind_apb')==1
code=code.replace('from same_edge_observation_binding import bind_apb','from tap_project_observation_binding import bind_apb')
exec(compile(code,str(source)+':shared-frontier','exec'),globals())
