"""One C1 route after explicit whole-placement review; no hardware access."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once

def prepare_code():
    from camera_early_ratio_route import prepare_code as parent
    source,code=parent()
    code=once(code,'from camera_checked_board import audit_delta','from camera_capture_burst_board import audit_delta')
    code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_2'}","{'before':'CONGESTION_1','after':'CONGESTION_1'}")
    compile(code,str(source)+':burst-c1','exec')
    return source,code

def route_code():
    from route_local_driver_board import driver_code
    source,code=driver_code()
    assert code.count('TIMING_1')==2
    code=code.replace('TIMING_1','CONGESTION_1')
    code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_1'}","{'before':'CONGESTION_1','after':'CONGESTION_1'}")
    code=once(code,'from local_driver_board import audit_delta','from camera_capture_burst_board import audit_delta')
    code=once(code,'import audit_control_state_implementation as impl','import camera_burst_implementation as impl')
    code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_recovery_gray.py'")
    compile(code,str(source)+':burst-c1','exec')
    return source,code

def main():
    action=sys.argv.pop(1)
    if action=='--check-only':
        prepare_code();route_code();print('C1_PREPARE_ROUTE_GENERATORS_COMPILE');return
    if action=='prepare':
        source,code=prepare_code()
        review=Path(sys.argv[sys.argv.index('--review')+1]).resolve()
        root=Path(json.loads(review.read_text())['physical'])
        driver=root.parent/(root.name+'-prepare-payload.py');assert not driver.exists()
        driver.write_text(code,encoding='utf8')
        driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__),hardware_access=False),indent=2))
        sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
    elif action=='route':
        source,code=route_code()
        root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
        driver=root.parent/(root.name+'-local-driver-route.py');assert not driver.exists()
        driver.write_text(code,encoding='utf8')
        driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
        sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
    else:raise ValueError(action)

if __name__=='__main__':main()
