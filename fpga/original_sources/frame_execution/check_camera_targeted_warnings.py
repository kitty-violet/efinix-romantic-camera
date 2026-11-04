"""Keep original complete warning checks on the targeted map."""
from pathlib import Path
source=Path(__file__).with_name('check_camera_joint_warnings.py')
code=source.read_text().replace('from camera_joint_board import','from camera_targeted_board import')
exec(compile(code,str(source)+':targeted','exec'),globals())
