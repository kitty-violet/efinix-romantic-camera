"""Original full synthesis probe with the additional indexed serializer source."""
from pathlib import Path
source=Path(__file__).with_name('map_video_service_candidate.py')
code=source.read_text();old="'video_service_modes.sv','video_service_health.sv']"
assert code.count(old)==1
code=code.replace(old,"'video_service_modes.sv','video_service_health.sv','video_snapshot_serializer.sv']")
exec(compile(code,str(source)+':indexed-serializer','exec'),globals())
