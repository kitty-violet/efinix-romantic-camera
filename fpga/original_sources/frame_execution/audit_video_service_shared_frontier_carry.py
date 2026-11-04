"""Same whole-cone/state proof with the operand-only carry plan."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_shared_frontier.py');code=source.read_text()
old="O=ROOT/'tmp/frame-execution/video-service63-batch-screen1'";assert code.count(old)==1
code=code.replace(old,"O=Path(r['plan_root'])")
exec(compile(code,str(source)+':dedicated-carry-plan','exec'),globals())
