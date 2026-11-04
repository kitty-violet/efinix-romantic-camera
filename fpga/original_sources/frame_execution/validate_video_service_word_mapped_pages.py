"""Actual mapped schema3 packet and all16 process/pixel words."""
from pathlib import Path
original=Path(__file__).with_name('validate_video_service_mapped_pages.py')
code=original.read_text()
changes=[('[:183]','[:87]'),('len(positions)==183','len(positions)==87'),
 ('rng.getrandbits(128)','rng.getrandbits(32)'),("setword('service_pixel_cache',128,pixel)","setword('service_pixel_cache',32,pixel)"),
 ('range(4):','range(16):'),("setword('service_page_part',2,part)","setword('service_page_part',4,part)"),
 ("setword('service_pixel_part',2,part)","setword('service_pixel_part',4,part)"),
 ('(2<<40)','(3<<40)'),('packet>>128','packet>>32'),('((1<<128)-1)','((1<<32)-1)'),
 ('actual[fields[1]],1,*','actual[fields[1]],0x53520001,*'),('128*part','32*part'),
 ('full128-bit pixel cache','full32-bit pixel cache')]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
# The generic LUT evaluator must retain its four input pins.
code=code.replace('for i in range(16))&1','for i in range(4))&1')
exec(compile(code,str(original)+':schema3-word','exec'),globals())
