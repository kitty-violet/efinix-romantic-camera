"""Version5 independent word oracle against actual mapped RAM WDATA."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_compact_mapped_pages.py');code=source.read_text()
changes=[
 ('from video_service_sample_ram import capture_roots','from video_service_narrow_sample_ram import capture_roots'),
 ('range(4)','range(16)'),("'service_page_part',2","'service_page_part',4"),("'service_pixel_part',2","'service_pixel_part',4"),
 ('128','32'),('192-len(roots)','96-len(roots)'),('<<160','<<64'),('<<168','<<72'),('<<176','<<80'),('<<177','<<81'),('<<178','<<82'),
 ('(4<<72)','(5<<72)'),('All8 original legacy chunks','All32 original legacy words')]
for old,new in changes:
 assert old in code,old;code=code.replace(old,new)
exec(compile(code,str(source)+':narrow-word-oracle','exec'),globals())
