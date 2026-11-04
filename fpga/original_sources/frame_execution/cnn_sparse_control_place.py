"""Placement only, retaining original clock targets and ordinary-cell freedom."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('cnn_observer_trim_place.py')
code=SOURCE.read_text(encoding='utf8').replace('cnn_observer_trim_board','cnn_sparse_control_board')
code=code.replace('cnn-observer-trim-place1','cnn-sparse-control-place1')
code=code.replace('cnn-observer-trim-native1','cnn-sparse-control-native1')
code=code.replace('OBSERVER_TRIM_PLACE_DRIVER_CHECKED','SPARSE_CONTROL_PLACE_DRIVER_CHECKED')
code=once(code,"wrapper_sha256=sha(__file__),","wrapper_sha256=sha(__file__),base_wrapper_sha256=sha(SOURCE),")
exec(compile(code,str(SOURCE)+':sparse-control-place','exec'),globals())
