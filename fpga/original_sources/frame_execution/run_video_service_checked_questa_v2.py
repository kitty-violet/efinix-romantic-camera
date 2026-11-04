"""Run existing mapped short checks with bounded Questa child processes."""
import json,os,runpy,sys
from pathlib import Path
import run_units
from map_video_service_safe import run as safe_run

if __name__=='__main__':
 name=sys.argv[1]
 assert name in ('validate_video_service_mapped_boot.py','validate_video_service_owner_boundary.py','validate_video_service_owner_boundary_v2.py')
 script=run_units.HERE/name
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 assert root.is_relative_to(run_units.ROOT/'tmp') and not root.exists()
 run_units.MS=Path('E:/quatra/questa_fse/win64');assert (run_units.MS/'vsim.exe').is_file()
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run_units.run=safe_run
 receipt=dict(wrapper_sha256=run_units.sha(__file__),test_source=str(script),test_sha256=run_units.sha(script),
              safe_runner_sha256=run_units.sha(run_units.HERE/'map_video_service_safe.py'),simulator=str(run_units.MS),hardware_access=False)
 sys.argv=[str(script)]+sys.argv[2:]
 try:runpy.run_path(str(script),run_name='__main__')
 finally:
  if root.exists():(root/'execution-wrapper.json').write_text(json.dumps(receipt,indent=2))

