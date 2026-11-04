"""Use the corrected continuous serial stimulus on the current full map."""
from pathlib import Path
import windows_process_tree as cleanup
from windows_process_tree_safe import stop_launched_tree
cleanup.stop_launched_tree=stop_launched_tree
source=Path(__file__).with_name('validate_continuous_serial.py')
code=source.read_text(encoding='utf8')
assert code.count('import continuous_control_flow as flow')==1
code=code.replace('import continuous_control_flow as flow','import continuous_srl_register_flow as flow')
exec(compile(code,str(source)+':srl-register-serial','exec'),globals())
