"""Original routed PLL pulse checks bound to the explicit color/exposure delta."""
from pathlib import Path

source=Path(__file__).with_name('video_service_pll.py')
code=source.read_text()
old='from place_video_service_epoch import audit_delta'
assert code.count(old)==1
code=code.replace(old,'from place_video_service_color_exposure import audit_delta')
code=code.replace("board/'retained-map-audit.json'","board/'color-map-audit.json'")
code=code.replace("HERE/'place_video_service_epoch.py'","HERE/'place_video_service_color_exposure.py'")
exec(compile(code,str(source)+':color-exposure-current-geometry','exec'),globals())
