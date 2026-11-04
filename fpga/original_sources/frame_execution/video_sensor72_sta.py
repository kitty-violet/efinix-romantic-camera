"""Original-constraint STA with isolated reporters and 2000 setup/hold endpoints."""
from pathlib import Path
source_text=Path(__file__).with_name('canonical_snapshot_sta_isolated.py').read_text()
needle="exec(compile(code,str(source)+':isolated-reporter-processes','exec'),globals())"
assert source_text.count(needle)==1
source_text=source_text.replace(needle,
 "assert code.count('-npaths 5 -nworst 1')==2\n"
 "code=code.replace('-npaths 5 -nworst 1','-npaths 2000 -nworst 1')\n"
 "code=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')\n"+needle)
exec(compile(source_text,__file__+':isolated-2000','exec'),globals())
