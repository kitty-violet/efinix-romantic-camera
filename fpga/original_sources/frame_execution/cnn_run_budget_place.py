"""Reviewed whole-board placement recipe, stop before routing."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_factor_island_flow import driver_code as previous
def main():
 source,code=previous()
 code=once(code,'from camera_factor_balanced_board import BOARD,audit_delta','from cnn_run_budget_board import BOARD,audit_delta')
 code=code.replace('camera_factor_island_locations','cnn_run_budget_locations')
 if sys.argv[1:]==['--check-only']:
  compile(code,'cnn-run-placement','exec');print('CNN_RUN_PLACE_DRIVER_CHECKED');return
 assert sys.argv.pop(1)=='place'
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists();driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
