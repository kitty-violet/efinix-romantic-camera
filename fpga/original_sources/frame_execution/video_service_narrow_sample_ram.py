"""Actual-map RAM assertions for version5, no weakening of clock/mode checks."""
from pathlib import Path
source=Path(__file__).with_name('video_service_sample_ram.py');code=source.read_text()
for old,new in [('set(range(10))','set(range(5))'),('bit<192','bit<96'),('len(roots)==192','len(roots)==96'),
 ("('service_serializer',192,","('service_serializer',96,"),("('service_pixel_snapshot',128,","('service_pixel_snapshot',32,"),('len(rows)==17','len(rows)==7')]:
 assert old in code,old;code=code.replace(old,new)
exec(compile(code,str(source)+':narrow-geometry','exec'),globals())
