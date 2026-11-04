"""Original strict route/STA sequence with RAM-specific JTAG CDC audit."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
def main():
 source=HERE/'registered_boundary_route.py';code=source.read_text()
 code=once(code,'from registered_boundary_board import audit_delta','from snapshot_ram_board import audit_delta')
 code=once(code,"HERE/'audit_split_logic_outputs_gray.py'","HERE/'audit_snapshot_ram_gray.py'")
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 driver=root.parent/(root.name+'-ram-wrapper.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code)
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  driver_sha256=sha(driver),wrapper_sha256=sha(__file__),acceptance_gates_unchanged=True,
  changed_payload_cdc_auditor=sha(HERE/'snapshot_ram_cdc.py')),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
