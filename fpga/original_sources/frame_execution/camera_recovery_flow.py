"""Retain the full offline flow, scoped to the camera recovery map."""
from pathlib import Path
source=Path(__file__).with_name('camera_payload_flow.py')
code=source.read_text(encoding='utf8').replace('camera_payload_board','camera_recovery_board').replace('camera_payload_implementation','camera_recovery_implementation').replace('camera_payload_flow','camera_recovery_flow')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_recovery_gray.py'")
exec(compile(code,str(source)+':recovery-bound','exec'),globals())
