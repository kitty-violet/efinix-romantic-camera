"""Bounded capacity probe, preserving the RQ-specific retained-map proof."""
from pathlib import Path
source = Path(__file__).with_name('probe_video_service_packing.py')
code = source.read_text()
assert code.count('place_video_service_epoch.py') == 1
code = code.replace('place_video_service_epoch.py', 'place_video_service_rq.py')
exec(compile(code, str(source)+':rq-identity', 'exec'), globals())
