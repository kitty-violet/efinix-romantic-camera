"""Full offline gates bound to the frozen sixteen-beat capture candidate."""
from pathlib import Path
source=Path(__file__).with_name('camera_payload_flow.py')
code=source.read_text(encoding='utf8').replace('camera_payload_board','camera_capture_prefetch_board').replace('camera_payload_implementation','camera_prefetch_implementation').replace('camera_payload_flow','camera_prefetch_flow')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_recovery_gray.py'")
exec(compile(code,str(source)+':capture-burst-bound','exec'),globals())
