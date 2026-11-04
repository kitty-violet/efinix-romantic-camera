"""Original routed PLL proof bound to the current input ISP source/map."""
from pathlib import Path

source = Path(__file__).with_name('video_service_pll.py')
code = source.read_text()
old = 'from place_video_service_epoch import audit_delta'
assert code.count(old) == 1
code = code.replace(old, 'from place_video_input_isp import audit_delta')
code = code.replace("board/'retained-map-audit.json'", "board/'input-isp-map-audit.json'")
code = code.replace("HERE/'place_video_service_epoch.py'", "HERE/'place_video_input_isp.py'")
exec(compile(code, str(source) + ':input-isp-current-geometry', 'exec'), globals())
