"""Run existing full interface short gates against the new four-SRL map."""
from pathlib import Path
import windows_process_tree as cleanup
from windows_process_tree_safe import stop_launched_tree
cleanup.stop_launched_tree=stop_launched_tree
source=Path(__file__).with_name('continuous_control_flow.py')
code=source.read_text(encoding='utf8')
code=code.replace(".replace('camera_payload','continuous_control')", ".replace('camera_payload','continuous_srl_register')")
code=code.replace("code.replace('continuous_control_board','continuous_control_board_v2')", "code.replace('continuous_srl_register_board','continuous_srl_register_board')")
needle="        generated(source,code,'-payload-short-wrapper')"
# The inner wrapper must use the already-corrected two-sided serial quiet
# stimulus, not repeat the old one-sided testbench error.
extra="""        code=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')
        code=once(code,\"run('serial','continuous_srl_register_flow.py',['flow','serial',\",\"run('serial','validate_srl_register_serial.py',['flow','serial',\")
        code=code.replace('validate_continuous_adapter.py','validate_srl_register_adapter.py')
"""
# needle is a string literal in this adapter layer; extend its existing
# injection string before it is applied to the actual flow.
anchor="code=once(code,needle,injection+needle)"
assert code.count(anchor)==1
code=code.replace(anchor,"injection+="+repr(extra)+"\n"+anchor)
exec(compile(code,str(source)+':srl-register-shorts','exec'),globals())
