"""Freeze reviewed placement with correct wrapper lineage and original timing."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
from camera_early_ratio_route import prepare_code
source,code=prepare_code()
code=once(code,'from camera_checked_board import audit_delta','from cnn_run_budget_board import audit_delta')
code=once(code,"('-ratio-place-wrapper','-place-driver')","('-place-wrapper','-place-driver')")
if sys.argv[1:]==['--check-only']:
 compile(code,'cnn-budget-route-prepare','exec');print('CNN_BUDGET_ROUTE_PREPARE_COMPILES')
else:
 review=Path(sys.argv[sys.argv.index('--review')+1]).resolve();root=Path(json.loads(review.read_text())['physical'])
 driver=root.parent/(root.name+'-prepare-cnn-budget.py');assert not driver.exists()
 driver.write_text(code,encoding='utf8');driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
