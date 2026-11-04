"""New independently frozen candidate; do not modify prior joint map evidence."""
from pathlib import Path
BASE_SCRIPT=Path(__file__).with_name('camera_joint_board.py')
code=BASE_SCRIPT.read_text().replace('connected-board-camera-joint3','connected-board-camera-targeted1').replace('camera-joint-native6','camera-targeted-native1')
needle='(Path(__file__),TRANSFORM/';assert code.count(needle)==1
code=code.replace(needle,'(Path(__file__),BASE_SCRIPT,TRANSFORM/')
exec(compile(code,str(BASE_SCRIPT)+':measured-critical-only','exec'),globals())
