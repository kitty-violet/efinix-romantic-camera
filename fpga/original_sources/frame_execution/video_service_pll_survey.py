"""Keep the frozen historical survey intact; use safe cleanup in this run."""
import json,sys
from pathlib import Path
from run_units import sha
source=Path(__file__).with_name('survey_pll_shift_routed.py')
code=source.read_text()
old='from windows_process_tree import stop_launched_tree'
assert code.count(old)==1
scope=dict(__file__=__file__,__name__='service_pll_survey')
exec(compile(code.replace(old,'from windows_process_tree_safe import stop_launched_tree'),str(source)+':safe-cleanup','exec'),scope)
if __name__=='__main__':
 scope['main']()
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 result=json.loads((root/'result.json').read_text())
 result['adapter_sources']={str(f):sha(f) for f in (source,Path(__file__),source.with_name('windows_process_tree_safe.py'))}
 (root/'result.json').write_text(json.dumps(result,indent=2))
