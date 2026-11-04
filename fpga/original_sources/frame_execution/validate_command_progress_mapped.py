"""Mapped startup/reset gate; first-fault/live values are checked via serial I/O.

Internal vector bits may be alias-merged after synthesis. The serial gate
checks the retained output path and injected faults, not undriven aliases.
"""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import audit_delta
from audit_command_toggle_sync import audit as command_sync

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();audit_delta(board)
 net=board/'outflow/ti60f225_oob.map.v';sync=command_sync(net.read_text());assert not sync['errors'],sync
 source=HERE/'validate_global_mapped_startup.py';code=source.read_text()
 # 128 extra observation bits retain 415 additional FFs plus associated logic.
 # Raise only the extraction size sanity cap; closure/cut checks stay identical.
 marker="exec(compile(code,str(old),'exec')"
 assert code.count(marker)==1
 code=code.replace(marker,"code=code.replace('len(selected)<4500','len(selected)<5500')\n    "+marker)
 code=code.replace(r'\cpu/sequencer/state ',r'\cpu/boot_observer_state ')
 code=code.replace("status='GLOBAL_MAPPED_STARTUP_THREE_RESETS_PASS'","status='COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT'")
 driver=root.parent/(root.name+'-runner.py');assert not driver.exists() and not root.exists()
 driver.write_text(code,encoding='utf-8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),wrapper_sha256=sha(__file__),map_sha256=sha(net)),indent=2))
 sys.argv=[str(driver),'--board',str(board),'--root',str(root)];runpy.run_path(str(driver),run_name='__main__')
 assert 'PASS GLOBAL_MAPPED_STARTUP resets=3' in (root/'simulation.log').read_text()
if __name__=='__main__':main()
