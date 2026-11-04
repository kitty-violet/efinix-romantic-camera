"""Reuse unchanged complete-board gates with audited local-driver lineage."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from local_driver_board import BASE,audit_delta
from control_command_repair import once

def generated(source,code):
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=root.parent/(root.name+'-local-driver-wrapper.py');assert not driver.exists()
    driver.parent.mkdir(parents=True,exist_ok=True);compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

def main():
    action=sys.argv.pop(1)
    if action=='flow':
        import startup_cause_flow as flow
        flow.BASE=BASE;flow.audit_delta=audit_delta;flow.main()
    elif action=='audit':
        import audit_fault_probe_map as prior
        prior.audit_delta=audit_delta;prior.main()
        board=Path(sys.argv[sys.argv.index('--board')+1]).resolve();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
        from audit_native_split_map import extra_map
        from native_split_cdc import map_structure
        (root/'new_diagnostic_cells.json').write_text(json.dumps(extra_map(board),indent=2))
        (root/'held_data_structure.json').write_text(json.dumps(map_structure(board),indent=2))
    elif action=='place':
        source=HERE/'place_registered_boundary_board.py'
        code=once(source.read_text(),'from registered_boundary_board import audit_delta','from local_driver_board import audit_delta')
        generated(source,code)
    elif action=='shorts':
        source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
        code=once(code,"'audit_split_logic_outputs_map.py',['--board'","'local_driver_flow.py',['audit','--board'")
        code=code.replace("'split_logic_outputs_flow.py',['", "'local_driver_flow.py',['flow','")
        assert 'split_logic_outputs_flow.py' not in code
        code=once(code,"status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'")
        generated(source,code)
    else:raise ValueError(action)

if __name__=='__main__':main()
