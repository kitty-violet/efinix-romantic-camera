"""Existing OFP5 interface checks, rebound to the exact registered candidate."""
import json,sys
from pathlib import Path
from run_units import HERE,sha
from registered_boundary_board import BASE,audit_delta
from audit_snapshot_cells import cells
import validate_control_ddr as vd
import startup_cause_flow as flow

def main():
    if sys.argv[1]=='serial':
        board=Path(sys.argv[sys.argv.index('--board')+1]).resolve()
        root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
        net=board/'outflow/ti60f225_oob.map.v';cc=cells(net.read_text())
        rtl=board/'assembly/EfxSapphireSoc.v';body=rtl.read_text()
        assert 'assign host_ready=!reset && ctrl_state_match[0] && choose_host;' in body
        assert 'wire choose_host=host_valid && (!boot_valid || !last_host);' in body
        ce=cc['cpu/observation_accepts[0]~FF'][1]['CE']
        assert cc['cpu/first_fault/previous_flags[6]~FF'][1]['D']==ce
        assert all(cc[f'cpu/observation_accepts[{i}]~FF'][1]['CE']==ce for i in range(8))
        original=vd.driver_run
        def driver_run(origin,code,out,args):
            if origin.name=='check-mapped-jtag-regression.py':
                before=r'dut.\cpu/host_ready ';after='dut.\\'+ce+' '
                assert out==root and code.count(before)==1
                code=code.replace(before,after)
                record=dict(status='TEST_ONLY_HOST_READY_REFERENCE_BOUND',old_reference=before,
                    new_reference=after,map_sha256=sha(net),rtl_sha256=sha(rtl),
                    wrapper_sha256=sha(__file__),hardware_source_changed=False,assertions_removed=False)
                dest=root.parent/(root.name+'-reference-binding.json');assert not dest.exists()
                dest.write_text(json.dumps(record,indent=2),encoding='utf8')
            return original(origin,code,out,args)
        vd.driver_run=driver_run
    flow.audit_delta=audit_delta;flow.BASE=BASE
    flow.main()

if __name__=='__main__':main()
