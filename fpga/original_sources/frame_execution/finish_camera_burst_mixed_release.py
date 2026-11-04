"""Complete C1-place/C2-route release checks with exact wrapper provenance."""
from pathlib import Path
source=Path(__file__).with_name('finish_camera_burst_c1_release.py')
code=source.read_text(encoding='utf8').replace('camera_burst_c1_route','camera_burst_mixed_route').replace('finish_camera_burst_c1_release','finish_camera_burst_mixed_release')
exec(compile(code,str(source)+':mixed-route-provenance','exec'),globals())
