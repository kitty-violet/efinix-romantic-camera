"""Read-only expanded path inventory on frozen routed geometry; no release."""
from pathlib import Path
from run_units import sha
source=Path(__file__).with_name('canonical_snapshot_sta_isolated.py')
code=source.read_text(encoding='utf8')
needle="code=source.read_text(encoding='utf8')"
assert code.count(needle)==1
code=code.replace(needle,needle+"\ncode=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree').replace('-npaths 5 -nworst 1','-npaths 100 -nworst 3')")
code=code.replace('timeout=210','timeout=360')
adapter_sources={str(source):sha(source),str(Path(__file__)):sha(__file__)}
exec(compile(code,str(source)+':expanded-path-inventory','exec'),globals())
output.update(adapter_sources=adapter_sources,requested_paths=100,paths_per_endpoint=3,inventory_only=True,board_release=False)
(root/'result.json').write_text(json.dumps(output,indent=2)+'\n')
