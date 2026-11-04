"""Exhaustive21-input queue/fault handshake cut using actual EFX primitives."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_fault_ready_v2.py')
code=source.read_text(encoding='utf8')
needle="exec(compile(code,str(source)+':mapped-boundary-binding','exec'),globals())"
assert code.count(needle)==1
adjust="""
for a,b in [('len(boundary)==20','len(boundary)==21'),('reg[19:0] stim','reg[20:0] stim'),('1048576','2097152'),('known_controls=20','known_controls=21'),('known_control_inputs=20','known_control_inputs=21')]:
 assert a in code,a
 code=code.replace(a,b)
code=code.replace('paths=[Path(__file__).resolve(),source,','paths=[Path(__file__).resolve(),source,Path(__file__).with_name("validate_video_service_fault_ready_v2.py"),')
"""
code=code.replace(needle,adjust+'\n'+needle)
exec(compile(code,str(source)+':terminal-ready21','exec'),globals())
