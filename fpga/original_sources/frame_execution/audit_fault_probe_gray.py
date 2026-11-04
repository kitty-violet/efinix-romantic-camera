"""Use the established frozen-route audit with the six new CDC endpoints."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
def main():
 source=HERE/'audit_command_progress_gray.py';code=source.read_text()
 before="code=source.read_text()"
 after=before+"\n code=code.replace('from snapshot_clock_cdc import inventory','from fault_probe_clock_cdc import inventory')"
 assert code.count(before)==1;code=code.replace(before,after)
 # This outer adapter only changes endpoint classification, not any SDC.
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 driver=root.parent/(root.name+'-profile-driver.py');assert not driver.exists()
 driver.write_text(code);driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),wrapper_sha256=sha(__file__)),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
