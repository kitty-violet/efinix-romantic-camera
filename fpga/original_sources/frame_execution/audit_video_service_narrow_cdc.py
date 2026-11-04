"""Same complete actual-map crossing inventory, checking the seven narrow RAMs."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_cdc.py');code=source.read_text()
old='from video_service_sample_ram import banks,storage_structure,bit_tokens'
assert code.count(old)==1
code=code.replace(old,'from video_service_narrow_sample_ram import banks,storage_structure,bit_tokens')
exec(compile(code,str(source)+':narrow-storage','exec'),globals())
