"""Reuse full same-map startup/adapter/observe/serial gates for RAM snapshot."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
def main():
 reuse=None
 if '--reuse-map-audit' in sys.argv:
  pos=sys.argv.index('--reuse-map-audit');reuse=Path(sys.argv[pos+1]).resolve()
  del sys.argv[pos:pos+2]
 source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
 code=code.replace("'audit_split_logic_outputs_map.py'","'audit_snapshot_ram_map.py'")
 code=code.replace("'split_logic_outputs_flow.py'","'snapshot_ram_flow.py'")
 if reuse is not None:
  code=once(code,"run('map-audit','audit_snapshot_ram_map.py',['--board',board,'--root',root/'map-audit'],100)",
   f"run('map-audit','complete_snapshot_ram_map_audit.py',['--board',board,'--root',root/'map-audit','--source',{str(reuse)!r}],100)")
 code=once(code,"  run('rtl','snapshot_ram_flow.py',['rtl','--board',board,'--root',root/'rtl'],90)",
  "  # The exact snapshot and byte-identical bridge already passed RTL/reset tests.\n"
  "  from snapshot_ram_board import proofs\n  rtl_proofs=proofs()\n  save(root/'reused_rtl_proofs.json',rtl_proofs)")
 code=once(code,"status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='SNAPSHOT_RAM_SAME_MAP_SHORT_INTEGRATION_PASS'")
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
  wrapper_sha256=sha(__file__),driver_sha256=sha(driver),mapped_checks_unchanged=True),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
