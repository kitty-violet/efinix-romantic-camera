"""Schema5 capture uses five fixed-address RAM banks; all CDC checks retained."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_compact_jtag.py')
code=source.read_text()
old='from video_service_sample_ram import storage_structure'
assert code.count(old)==1
code=code.replace(old,'from video_service_narrow_sample_ram import storage_structure')
exec(compile(code,str(source)+':schema5','exec'),globals())
