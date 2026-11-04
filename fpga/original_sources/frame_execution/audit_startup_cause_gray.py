"""Keep original acceptance SDC and measure the new held-data CDC endpoints."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
def main():
 source=HERE/'audit_command_progress_gray.py';code=source.read_text()
 before='code=source.read_text()'
 after=before+"\n code=code.replace('from snapshot_clock_cdc import inventory','from startup_cause_clock_cdc import inventory')"
 after+="\n code=code.replace(\"here/'snapshot_clock_cdc.py'\",\"here/'startup_cause_clock_cdc.py'\")"
 assert code.count(before)==1;code=code.replace(before,after)
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 driver=root.parent/(root.name+'-profile-driver.py');assert not driver.exists()
 driver.write_text(code);driver.with_suffix('.json').write_text(json.dumps(dict(
  source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),
  wrapper_sha256=sha(__file__),cdc_auditor_sha256=sha(HERE/'startup_cause_clock_cdc.py')),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
