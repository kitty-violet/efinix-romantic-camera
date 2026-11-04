"""Resolve a renamed observation net by actual mapped sink connections."""
import json,re,sys
from pathlib import Path
from run_units import HERE,sha
from audit_snapshot_cells import cells
import validate_control_ddr as vd
import split_logic_outputs_flow as split_flow

def main():
 assert sys.argv[1]=='serial'
 board=Path(sys.argv[sys.argv.index('--board')+1]).resolve()
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 net=board/'outflow/ti60f225_oob.map.v';body=net.read_text();cc=cells(body)
 rtl=board/'assembly/EfxSapphireSoc.v';source=rtl.read_text()
 assert 'assign host_ready=!reset && ctrl_state_match[0] && choose_host;' in source
 assert 'wire choose_host=host_valid && (!boot_valid || !last_host);' in source
 # host_ready already implies host_valid. The accepted-command CE and the
 # separately captured flags[6] therefore identify this signal independently.
 ce=cc['cpu/observation_accepts[0]~FF'][1]['CE']
 d=cc['cpu/first_fault/previous_flags[6]~FF'][1]['D']
 assert ce==d and ce.startswith('cpu/'),(ce,d)
 assert all(cc[f'cpu/observation_accepts[{i}]~FF'][1]['CE']==ce for i in range(8))
 original=vd.driver_run
 before=r'dut.\cpu/host_ready ';after='dut.\\'+ce+' '
 def driver_run(origin,code,out,args):
  if origin.name=='check-mapped-jtag-regression.py':
   assert out==root and code.count(before)==1
   code=code.replace(before,after)
   record=dict(status='TEST_ONLY_HOST_READY_REFERENCE_BOUND',old_reference=before,new_reference=after,
    map_sha256=sha(net),rtl_sha256=sha(rtl),wrapper_sha256=sha(__file__),
    exact_sinks=['cpu/first_fault/previous_flags[6]~FF|D']+[f'cpu/observation_accepts[{i}]~FF|CE' for i in range(8)],
    hardware_source_changed=False,assertions_removed=False)
   target=root.parent/(root.name+'-reference-binding.json');assert not target.exists()
   target.write_text(json.dumps(record,indent=2))
  return original(origin,code,out,args)
 vd.driver_run=driver_run
 split_flow.flow.audit_delta=split_flow.audit_delta
 split_flow.flow.BASE=split_flow.BASE
 split_flow.flow.proof=split_flow.proof
 split_flow.flow.main()

if __name__=='__main__':main()
