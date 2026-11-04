"""Run the existing complete original STA with creation-time-safe timeout cleanup."""
from pathlib import Path
from run_units import sha

source = Path(__file__).with_name('canonical_snapshot_sta_isolated.py')
code = source.read_text(encoding='utf8')
needle = "code=source.read_text(encoding='utf8')"
assert code.count(needle) == 1
code = code.replace(needle, needle + "\ncode=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')")
adapter_sources = {str(source): sha(source), str(Path(__file__)): sha(__file__)}
exec(compile(code, str(source) + ':safe-cleanup', 'exec'), globals())
output['adapter_sources'] = adapter_sources
(root/'result.json').write_text(json.dumps(output, indent=2)+'\n')
