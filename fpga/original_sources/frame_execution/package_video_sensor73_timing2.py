"""Package the TIMING_2 candidate with every original sensor73 release gate."""
from pathlib import Path
source=Path(__file__).with_name('package_video_sensor73.py')
code=source.read_text(encoding='utf8')
for before,after in [("override['value']=='TIMING_1'","override['value']=='TIMING_2'"),('route-sensor73-timing1.py','route-sensor73-timing2.py')]:
 assert code.count(before)==1;code=code.replace(before,after)
needle="for f in (Path(__file__),HERE/'video_sensor73_physical.py'"
assert code.count(needle)==1
code=code.replace(needle,"for f in (Path(__file__),source,HERE/'video_sensor73_physical.py'")
exec(compile(code,str(source)+':verified-TIMING_2-route','exec'),globals())
