"""Complete upstream load transfer review for the smaller candidate."""
from pathlib import Path
source=Path(__file__).with_name('review_camera_joint_loads.py')
code=source.read_text().replace('from camera_joint_board import','from camera_targeted_board import')
exec(compile(code,str(source)+':targeted','exec'),globals())
