"""Native52 release profile: complete current evidence, unchanged release gates."""
from pathlib import Path
source=Path(__file__).with_name('package_video_service_fault_ready.py')
code=source.read_text(encoding='utf8')
for a,b in [
 ('video-service-native51-fault-ready','video-service-native53-parallel-group'),
 ('video-service-mapped-boot30-fault-ready','video-service-mapped-boot32-parallel-group'),
 ('video-service-owner-boundary18-fault-ready','video-service-owner-boundary20-parallel-group'),
 ('video-service-fault-ready','video-service-parallel-group'),
 ('mapped-control2','mapped-control1'),
 ('1048576','2097152'),
]:
 assert a in code,a;code=code.replace(a,b)
needle=" bind(HERE/'validate_video_service_fault_ready.py');bind(HERE/'validate_video_service_fault_ready_v2.py')"
assert needle in code
code=code.replace(needle,needle+";bind(HERE/'validate_video_service_terminal_ready.py');bind(HERE/'package_video_service_fault_ready.py')")
exec(compile(code,str(source)+':terminal-ready-current','exec'),globals())
