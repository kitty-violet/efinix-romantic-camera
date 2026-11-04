"""Alternative after two slow route prefixes: preserve unchanged engine tiles.

Stops after pack/place, before routing, for a whole-chip congestion comparison.
"""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
from camera_factor_island_flow import driver_code as previous

def driver_code():
 source,code=previous()
 code=once(code,'from camera_factor_balanced_board import BOARD,audit_delta','from cnn_run_budget_board import BOARD,audit_delta')
 code=code.replace('camera_factor_island_locations','cnn_run_budget_locations')
 needle="'arena_island=retain_arena_island(root)\\n' +"
 code=once(code,needle,needle+"\n        'from cnn_run_budget_engine_locations import generate as retain_engine\\n' +\n        'engine_island=retain_engine(root)\\n' +")
 code=once(code,"    from fixed_cnn_ram_locations import verify_placed", "    from cnn_run_budget_engine_locations import verify_placed as verify_engine\n    (root/'engine_island_actual_locations.json').write_text(json.dumps(verify_engine(root),indent=2))\n    from fixed_cnn_ram_locations import verify_placed")
 code=once(code,'all remaining ordinary cells free.', 'unchanged 9084-cell arithmetic engine retained as complete tiles/chains, 51297 ordinary cells free.')
 compile(code,'cnn-budget-engine-place','exec');return source,code

def main():
 source,code=driver_code()
 if sys.argv[1:]==['--check-only']:
  from cnn_run_budget_engine_locations import plan
  r=plan(ROOT/'tmp/frame-execution/cnn-run-budget-place2')
  print(json.dumps({k:r[k] for k in ('status','ordinary_cells_retained','ordinary_cells_free')}));return
 assert sys.argv.pop(1)=='place'
 for name in ('cnn-run-budget-route1','cnn-run-budget-route2'):
  batch=ROOT/'tmp/frame-execution'/name
  assert (batch/'failure.json').exists(),f'{name}: finish existing route first'
  assert all(s['status']!='running' for s in json.loads((batch/'stages.json').read_text()))
  assert (batch/'route-timeout.json').exists(),f'{name}: verify process cleanup'
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists()
 driver.write_text(code,encoding='utf8');driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
