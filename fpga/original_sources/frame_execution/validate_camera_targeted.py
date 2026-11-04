"""Reuse all actual-map primitive checks for the measured-net candidate."""
from pathlib import Path
source=Path(__file__).with_name('validate_camera_joint.py')
code=source.read_text().replace('from camera_joint_board import','from camera_targeted_board import')
needle="runner_sha256=sha(__file__),files=";assert code.count(needle)==1
code=code.replace(needle,"runner_sha256=sha(__file__),base_runner_sha256=sha(source),files=")
exec(compile(code,str(source)+':targeted','exec'),globals())
