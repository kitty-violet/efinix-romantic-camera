"""Same-map short, route and original timing gates for the data64 candidate."""
from pathlib import Path
SOURCE = Path(__file__).with_name('camera_payload_flow.py')
code = SOURCE.read_text(encoding='utf-8').replace('camera_payload', 'cnn_data64')
code = code.replace("HERE/'camera_checked_gray_v2.py'", "HERE/'camera_joint_gray.py'")
exec(compile(code, str(SOURCE) + ':cnn-data64', 'exec'), globals())
