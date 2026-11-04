"""Current-map short gates and physical helpers, preserving original checks."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('camera_payload_flow.py')
code=SOURCE.read_text(encoding='utf8').replace('camera_payload','continuous_control')
code=code.replace('continuous_control_board','continuous_control_board_v2')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_joint_gray.py'")
needle="        generated(source,code,'-payload-short-wrapper')"
injection='''        old="[sys.executable,'-B',str(HERE/script),*map(str,args)]"
        new="[sys.executable,'-B',str(ROOT/'tmp/board-test/run-with-locked-evidence.py'),'--cache-log',str(root/(name+'-cache.json')),str(HERE/script),*map(str,args)]"
        code=once(code,old,new)
        code=once(code,"root/'map-audit'],100)","root/'map-audit'],360)")
        code=once(code,'validate_control_adapter_idle_mapped.py','validate_continuous_adapter.py')
'''
code=once(code,needle,injection+needle)
compile(code,str(SOURCE)+':continuous-control','exec')
exec(compile(code,str(SOURCE)+':continuous-control','exec'),globals())
