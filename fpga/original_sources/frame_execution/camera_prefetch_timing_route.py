"""Controlled timing-route comparison on the reviewed C2 placement."""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
import camera_burst_c1_route as original

def prepare_code():
    from camera_early_ratio_route import prepare_code as parent
    source,code=parent()
    code=once(code,'from camera_checked_board import audit_delta','from camera_capture_prefetch_board import audit_delta')
    code=once(code,"{'before':'CONGESTION_2','after':'CONGESTION_2'}","{'before':'CONGESTION_2','after':'TIMING_1'}")
    compile(code,str(source)+':prefetch-timing','exec');return source,code

def route_code():
    from route_local_driver_board import driver_code
    source,code=driver_code()
    assert code.count('TIMING_1')==2
    code=once(code,'from local_driver_board import audit_delta','from camera_capture_prefetch_board import audit_delta')
    code=once(code,'import audit_control_state_implementation as impl','import camera_prefetch_implementation as impl')
    code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_recovery_gray.py'")
    compile(code,str(source)+':prefetch-timing','exec');return source,code

# Keep the established bounded process tree cleanup and all physical gates.
exec(compile(inspect.getsource(original.main),__file__+':complete-runner','exec'),globals())
if __name__=='__main__':main()
