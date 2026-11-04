"""Keep vendor Gray and finite probe checks; audit payload via native RAM CDC."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
def main():
 source=HERE/'audit_command_progress_gray.py';code=source.read_text()
 before='code=source.read_text()'
 extra="\n code=code.replace('from snapshot_clock_cdc import inventory','from split_logic_outputs_cdc import inventory')"
 extra+="\n code=code.replace(\"here/'snapshot_clock_cdc.py'\",\"here/'split_logic_outputs_cdc.py'\")"
 extra+="\n code=code.replace('from audit_snapshot_jtag_paths import audit as audit_jtag','from snapshot_ram_cdc import audit as audit_jtag')"
 code=once(code,before,before+extra)
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 driver=root.parent/(root.name+'-profile-driver.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code)
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  driver_sha256=sha(driver),wrapper_sha256=sha(__file__),
  cdc_auditor_sha256=sha(HERE/'split_logic_outputs_cdc.py'),ram_cdc_sha256=sha(HERE/'snapshot_ram_cdc.py')),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
