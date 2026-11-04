"""Complete original-constraint offline gates after reviewed84 refinement."""
from pathlib import Path
source=Path(__file__).with_name('finish_video_control84.py')
code=source.read_text(encoding='utf8')
for old,new in [("'video_service_narrow_gray.py'","'video_control84_margin_gray.py'"),("'package_video_control84.py'","'package_video_control84_refined.py'")]:
 assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':refined-original-constraints','exec'),globals())
