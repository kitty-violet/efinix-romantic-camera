"""Existing full video/A gates with a new explicitly bounded native delta."""
from pathlib import Path
source=Path(__file__).with_name('camera_payload_flow.py')
code=source.read_text(encoding='utf8').replace('camera_payload_board','camera_joint_board').replace('camera_payload_implementation','camera_joint_implementation').replace('camera_payload_flow','camera_joint_flow')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_joint_gray.py'")
exec(compile(code,str(source)+':joint-bound','exec'),globals())
