"""Independent seed3/C2 placement of the verified four-SRL structural delta."""
from pathlib import Path
source=Path(__file__).with_name('continuous_control_place.py')
code=source.read_text(encoding='utf8')
for old,new in [('continuous_control_board_v2','continuous_srl_register_board'),('continuous-control-place1','continuous-srl-register-place1'),('continuous-control-native2','continuous-srl-register-native1'),('CONTINUOUS_CONTROL_PLACE_DRIVER_CHECKED','CONTINUOUS_SRL_REGISTER_PLACE_DRIVER_CHECKED')]:
 assert old in code,old;code=code.replace(old,new)
# Preserve frozen parent tools; inject the validated safe timeout cleanup only
# into this candidate's generated placement driver.
needle="code=once(code,\"wrapper_sha256=sha(__file__),\",\"wrapper_sha256=sha(__file__),base_wrapper_sha256=sha(SOURCE),\")"
assert code.count(needle)==1
code=code.replace(needle,needle+'''\ncode=once(code,"source,code=original('free-arena')", "source,code=original('free-arena')\\n    assert code.count('code=source.read_text()')==1\\n    code=code.replace('code=source.read_text()', \\\"code=source.read_text().replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')\\\")")''')
exec(compile(code,str(source)+':srl-register-place','exec'),globals())
