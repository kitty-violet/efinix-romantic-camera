"""One independent placement-only backup; no automatic route or board access."""
from pathlib import Path
parent=Path(__file__).with_name('continuous_control_place.py')
body=parent.read_text(encoding='utf8')
assert body.count('continuous-control-place1')==1
body=body.replace('continuous-control-place1','continuous-control-seed1-place1')
needle="exec(compile(code,str(SOURCE)+':continuous-control-place','exec'),globals())"
assert body.count(needle)==1
change="""
assert code.count("'--seed','3'")==1 and code.count('seed=3,host_threads=4')==1
code=code.replace("'--seed','3'","'--seed','1'").replace('seed=3,host_threads=4','seed=1,host_threads=4')
"""
body=body.replace(needle,change+'\n'+needle)
exec(compile(body,str(parent)+':seed1-backup-place','exec'),globals())
