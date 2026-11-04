"""Current-map short gates; no hardware access."""
from pathlib import Path
SOURCE=Path(__file__).with_name('camera_payload_flow.py')
code=SOURCE.read_text().replace('camera_payload','cnn_timing_closure')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_joint_gray.py'")
exec(compile(code,str(SOURCE)+':timing-closure','exec'),globals())
