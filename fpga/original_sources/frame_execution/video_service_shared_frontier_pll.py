"""Retain routed PLL pulse gates with the current exact mapping proof."""
from pathlib import Path
source=Path(__file__).with_name('video_service_pll.py');code=source.read_text()
assert code.count('from place_video_service_epoch import audit_delta')==1
code=code.replace('from place_video_service_epoch import audit_delta','from place_video_service_shared_frontier import audit_delta')
code=code.replace("HERE/'place_video_service_epoch.py',","HERE/'video_service_pll.py',HERE/'place_video_service_shared_frontier.py',HERE/'place_video_service_epoch.py',")
exec(compile(code,str(source)+':shared-frontier-bound','exec'),globals())
