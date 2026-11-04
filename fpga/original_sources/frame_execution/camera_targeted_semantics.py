"""Preserve full two-generation semantic probe binding for targeted map."""
from pathlib import Path
source=Path(__file__).with_name('camera_joint_semantics.py')
code=source.read_text().replace('from camera_joint_board import','from camera_targeted_board import')
exec(compile(code,str(source)+':targeted','exec'),globals())
