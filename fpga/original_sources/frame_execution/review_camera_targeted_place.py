"""Same complete-chip and consumer-distance comparison for targeted placement."""
from pathlib import Path
source=Path(__file__).with_name('review_camera_joint_place.py')
code=source.read_text().replace('from camera_joint_board import','from camera_targeted_board import').replace('camera-joint-loads2.json','camera-targeted-loads1.json').replace('camera-joint-release-preflight1.json','camera-targeted-release-preflight1.json').replace('camera-joint-evidence2','camera-targeted-evidence1')
exec(compile(code,str(source)+':targeted','exec'),globals())
