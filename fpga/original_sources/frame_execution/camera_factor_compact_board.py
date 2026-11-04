"""Independent map; unchanged balanced graph, primitive and capacity audits."""
from pathlib import Path
COMPACT_BASE=Path(__file__).with_name('camera_factor_balanced_board.py')
code=COMPACT_BASE.read_text()
needle="exec(compile(code,str(FACTOR_BUILDER)+':balanced','exec'),globals())"
assert code.count(needle)==1
code=code.replace(needle,"code=code.replace('connected-board-camera-factor-balanced1','connected-board-camera-factor-compact1').replace('camera-factor-balanced-native1','camera-factor-compact-native1')\ncode=code.replace('FACTOR_BUILDER,TRANSFORM/', 'FACTOR_BUILDER,COMPACT_BASE,TRANSFORM/')\n"+needle)
exec(compile(code,str(COMPACT_BASE)+':compact','exec'),globals())
