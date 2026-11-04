"""Route only after reviewing the placement produced with the correct clocks."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
from camera_checked_v2 import BOARD
import camera_checked_board
camera_checked_board.BOARD=BOARD


def prepare_code():
    source=HERE/'prepare_local_driver_route.py'
    code=once(source.read_text(),'from local_driver_board import audit_delta',
              'from camera_checked_board import audit_delta')
    code=once(code,"'after':'TIMING_1'","'after':'CONGESTION_2'")
    code=once(code,"('-local-driver-wrapper','-place-wrapper','-place-driver')",
                   "('-ratio-place-wrapper','-place-driver')")
    compile(code,str(source)+':early-ratio-copy','exec')
    return source,code


def main():
    action=sys.argv.pop(1)
    if action=='prepare':
        source,code=prepare_code()
        review=Path(sys.argv[sys.argv.index('--review')+1]).resolve()
        r=json.loads(review.read_text());root=Path(r['physical'])
        assert r['placement'].endswith('camera-checked-place3')
        driver=root.parent/(root.name+'-prepare-ratio.py');assert not driver.exists()
        driver.write_text(code,encoding='utf8')
        driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
            generated_sha256=sha(driver),adapter_sha256=sha(__file__),hardware_access=False),indent=2))
        sys.argv=[str(driver),*sys.argv[1:]]
        runpy.run_path(str(driver),run_name='__main__')
        # The ratio was already applied before packing; no clock edit here.
        assert json.loads((root/'config_clock_ratio.json').read_text())['applied_before']=='interface_pack_place'
        print('EARLY_RATIO_PLACEMENT_REBOUND_READY_NO_CLOCK_REWRITE')
    elif action=='route':
        from route_local_driver_board import driver_code
        from camera_checked_flow import generated
        source,code=driver_code();assert code.count('TIMING_1')==2
        code=code.replace('TIMING_1','CONGESTION_2')
        code=once(code,'from local_driver_board import audit_delta','from camera_checked_board import audit_delta')
        code=once(code,'import audit_control_state_implementation as impl','import audit_camera_implementation as impl')
        code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_checked_gray_v2.py'")
        generated(source,code,'-local-driver-route')
    else:raise ValueError(action)


if __name__=='__main__':main()
