"""Reuse existing complete offline gates with the new bounded native delta."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_payload_board import BOARD,audit_delta

def generated(source,code,suffix):
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=root.parent/(root.name+suffix+'.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

def main():
    action=sys.argv.pop(1)
    if action=='shorts':
        source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
        code=once(code,"'audit_split_logic_outputs_map.py',['--board'","'camera_payload_flow.py',['audit','--board'")
        code=code.replace("'split_logic_outputs_flow.py',['", "'camera_payload_flow.py',['flow','")
        code=once(code,"status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'")
        generated(source,code,'-payload-short-wrapper')
    elif action=='flow':
        import startup_cause_flow as flow
        from camera_checked_board import BASE
        flow.BASE=BASE;flow.audit_delta=audit_delta;flow.main()
    elif action=='audit':
        argv=list(sys.argv);board=Path(argv[argv.index('--board')+1]).resolve();root=Path(argv[argv.index('--root')+1]).resolve()
        import audit_fault_probe_map as prior
        prior.audit_delta=audit_delta
        try:prior.main()
        finally:sys.argv=argv
        from camera_checked_cdc import map_structure
        (root/'held_data_structure.json').write_text(json.dumps(map_structure(board),indent=2))
    elif action=='place':
        from camera_ratio_before_place import driver_code
        source,code=driver_code()
        code=once(code,'from camera_checked_v2 import BOARD\nimport camera_checked_board\ncamera_checked_board.BOARD=BOARD\naudit_delta=camera_checked_board.audit_delta',
                       'from camera_payload_board import BOARD,audit_delta')
        generated(source,code,'-ratio-place-wrapper')
    elif action=='prepare-route':
        from camera_early_ratio_route import prepare_code
        source,code=prepare_code()
        code=once(code,'from camera_checked_board import audit_delta','from camera_payload_board import audit_delta')
        review=Path(sys.argv[sys.argv.index('--review')+1]).resolve();r=json.loads(review.read_text());root=Path(r['physical'])
        driver=root.parent/(root.name+'-prepare-payload.py');assert not driver.exists()
        driver.write_text(code,encoding='utf8')
        driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
            generated_sha256=sha(driver),adapter_sha256=sha(__file__),hardware_access=False),indent=2))
        sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
    elif action=='route':
        from route_local_driver_board import driver_code
        source,code=driver_code();assert code.count('TIMING_1')==2
        code=code.replace('TIMING_1','CONGESTION_2')
        code=once(code,'from local_driver_board import audit_delta','from camera_payload_board import audit_delta')
        code=once(code,'import audit_control_state_implementation as impl','import camera_payload_implementation as impl')
        code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_checked_gray_v2.py'")
        generated(source,code,'-local-driver-route')
    else:raise ValueError(action)

if __name__=='__main__':main()
