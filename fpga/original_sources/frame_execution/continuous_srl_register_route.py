"""Selected seed3/C2 route and all original STA/CDC checks for the new map."""
from pathlib import Path
source=Path(__file__).with_name('continuous_control_route.py')
body=source.read_text(encoding='utf8')
body=body.replace(".replace('cnn_sparse_control_board','continuous_control_board_v2')", ".replace('cnn_sparse_control_board','continuous_srl_register_board')")
body=body.replace(".replace('cnn_sparse_control_implementation','continuous_control_implementation')", ".replace('cnn_sparse_control_implementation','continuous_srl_register_implementation')")
_entry_name=__name__;__name__='srl_route_generator_only'
try:exec(compile(body,str(source)+':srl-register-route','exec'),globals())
finally:__name__=_entry_name
_original_route_code=route_code
def route_code():
    source,code=_original_route_code()
    assert code.count('from windows_process_tree import stop_launched_tree')==1
    code=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')
    return source,code
if __name__=='__main__':main()
