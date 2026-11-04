"""Full interface and physical gates for late-control factorization."""
from pathlib import Path
source=Path(__file__).with_name('camera_payload_flow.py')
code=source.read_text().replace('camera_payload_board','camera_factor_board').replace('camera_payload_implementation','camera_factor_implementation').replace('camera_payload_flow','camera_factor_flow')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_joint_gray.py'")
exec(compile(code,str(source)+':factor-bound','exec'),globals())
