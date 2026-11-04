"""Independent map of the physically grouped late-control candidate."""
from pathlib import Path
FACTOR_BUILDER=Path(__file__).with_name('camera_factor_board.py')
code=FACTOR_BUILDER.read_text().replace('connected-board-camera-factor1','connected-board-camera-factor-balanced1').replace('camera-factor-native2','camera-factor-balanced-native1')
code=code.replace("'(Path(__file__),BASE_SCRIPT,TRANSFORM/'","'(Path(__file__),BASE_SCRIPT,FACTOR_BUILDER,TRANSFORM/'")
exec(compile(code,str(FACTOR_BUILDER)+':balanced','exec'),globals())
