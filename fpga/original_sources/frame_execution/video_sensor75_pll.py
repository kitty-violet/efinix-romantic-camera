"""Actual routed PLL proof with the sensor75 exact source-scope gate."""
from pathlib import Path
source=Path(__file__).with_name('video_service_pll.py')
code=source.read_text()
old='from place_video_service_epoch import audit_delta';assert code.count(old)==1
code=code.replace(old,'from place_video_sensor75 import audit_delta')
code=code.replace("board/'retained-map-audit.json'","board/'sensor75-map-audit.json'")
code=code.replace("HERE/'place_video_service_epoch.py'","HERE/'place_video_sensor75.py'")
exec(compile(code,str(source)+':sensor75-current-geometry','exec'),globals())
