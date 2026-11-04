"""Four-thread C1 placement comparison; same frozen hardware, stop before route."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
from cnn_timing_closure_place import driver_code as parent

def driver_code():
 source,code=parent()
 code=once(code,'value="8"','value="4"')
 code=once(code,'official_max_threads=8','official_max_threads=4')
 compile(code,'closure-c1-place','exec');return source,code

def main():
 source,code=driver_code()
 if sys.argv[1:]==['--check-only']:print('FOUR_THREAD_C1_PLACE_COMPILES');return
 assert sys.argv.pop(1)=='place'
 assert sys.argv[sys.argv.index('--optimization')+1]=='CONGESTION_1'
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists()
 driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
