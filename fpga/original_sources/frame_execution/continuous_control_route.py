"""Selected C2 route, isolated full original timing/constraint checks and Gray."""
from pathlib import Path
SOURCE=Path(__file__).with_name('cnn_sparse_control_route.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('cnn_sparse_control_board','continuous_control_board_v2')
code=code.replace('cnn_sparse_control_implementation','continuous_control_implementation')
exec(compile(code,str(SOURCE)+':continuous-control','exec'),globals())
