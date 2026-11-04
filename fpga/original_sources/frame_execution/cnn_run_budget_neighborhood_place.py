"""Place with unchanged physical surroundings and free changed neighborhoods."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('cnn_run_budget_engine_place.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('cnn_run_budget_engine_locations','cnn_run_budget_neighborhood_locations')
code=code.replace('engine_island','neighborhood').replace('9084','44485').replace('51297','15896')
needle=" root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();"
check=""" reference=Path(sys.argv[sys.argv.index('--cnn-ram-reference')+1]).resolve()
 expected=json.loads((ROOT/'tmp/frame-execution/cnn-run-budget-place2/cnn_ram_placement.json').read_text())['reference_canonical']
 assert reference==Path(expected).resolve() and (reference/'inputs.json').exists()
 assert (ROOT/'tmp/board-test/cnn-budget-cone-layout-rejection1.json').exists()
"""
code=once(code,needle,check+needle)
exec(compile(code,str(SOURCE)+':physical-neighborhood-placement','exec'),globals())
