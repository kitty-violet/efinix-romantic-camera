"""Place complete unchanged CNN data cones, then stop for global review."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('cnn_run_budget_engine_place.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('cnn_run_budget_engine_locations','cnn_run_budget_cone_locations')
code=code.replace('engine_island','cnn_cone').replace('9084','27572').replace('51297','32809')
needle=" root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();"
check=""" reference=Path(sys.argv[sys.argv.index('--cnn-ram-reference')+1]).resolve()
 expected=json.loads((ROOT/'tmp/frame-execution/cnn-run-budget-place2/cnn_ram_placement.json').read_text())['reference_canonical']
 assert reference==Path(expected).resolve() and (reference/'inputs.json').exists()
 rejected=json.loads((ROOT/'tmp/frame-execution/cnn-run-budget-engine-place3/result.json').read_text())
 assert all(s['exit_code']==0 for s in rejected['stages'])
 assert (ROOT/'tmp/board-test/cnn-budget-engine-layout-rejection1.json').exists()
"""
code=once(code,needle,check+needle)
exec(compile(code,str(SOURCE)+':connected-cone-placement','exec'),globals())
