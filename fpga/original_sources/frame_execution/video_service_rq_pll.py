"""Keep the routed PLL pulse checks, using the current RQ graph proof."""
from pathlib import Path
source=Path(__file__).with_name('video_service_pll.py')
code=source.read_text()
old='from place_video_service_epoch import audit_delta'
assert code.count(old)==1
code=code.replace(old,'from place_video_service_rq import audit_delta')
old="HERE/'place_video_service_epoch.py',"
assert code.count(old)==1
code=code.replace(old,"HERE/'video_service_pll.py',HERE/'place_video_service_rq.py',"+old)
exec(compile(code,str(source)+':rq-bound-graph','exec'),globals())
