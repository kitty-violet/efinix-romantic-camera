"""Same frozen balanced map, with only 36 justified ordinary cells retained."""
import json,sys,runpy
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once

def driver_code():
 from camera_ratio_before_place import driver_code as prior
 source,code=prior()
 code=once(code,'from camera_checked_v2 import BOARD\nimport camera_checked_board\ncamera_checked_board.BOARD=BOARD\naudit_delta=camera_checked_board.audit_delta',
  'from camera_factor_balanced_board import BOARD,audit_delta')
 needle="f'cnn_ram_layout=retain_cnn_ram(root,Path({str(reference)!r}))\\n'+"
 code=once(code,needle,needle+"\n        'from camera_factor_island_locations import generate as retain_arena_island\\n' +\n        'arena_island=retain_arena_island(root)\\n' +")
 code=once(code,'same DSP cascade and CNN RAM anchors, no general logic fixed.',
  'Same DSP/CNN RAM anchors plus 36 exact unchanged arena cells at prior legal complete tiles; all remaining ordinary cells free.')
 code=once(code,"from fixed_cnn_ram_locations import verify_placed",
  "from camera_factor_island_locations import verify_placed as verify_arena_island\n    (root/'arena_island_actual_locations.json').write_text(json.dumps(verify_arena_island(root),indent=2))\n    from fixed_cnn_ram_locations import verify_placed")
 compile(code,str(source)+':arena-island','exec')
 return source,code

def main():
 source,code=driver_code()
 if sys.argv[1:]==['--check-only']:
  print('ISLAND_PLACEMENT_DRIVER_COMPILES');return
 assert sys.argv.pop(1)=='place'
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert not root.exists()
 driver=root.parent/(root.name+'-ratio-place-wrapper.py');assert not driver.exists()
 driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
