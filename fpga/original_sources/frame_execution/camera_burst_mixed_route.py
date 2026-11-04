"""C2 routing of an explicitly reviewed C1 placement, without RTL changes."""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
import camera_burst_c1_route as parent

def prepare_code():
    source,code=parent.prepare_code()
    return source,once(code,"{'before':'CONGESTION_1','after':'CONGESTION_1'}","{'before':'CONGESTION_1','after':'CONGESTION_2'}")

def route_code():
    source,code=parent.route_code()
    code=once(code,'optimization_level=CONGESTION_1','optimization_level=CONGESTION_2')
    code=once(code,"{'before':'CONGESTION_1','after':'CONGESTION_1'}","{'before':'CONGESTION_1','after':'CONGESTION_2'}")
    compile(code,'burst-mixed-route','exec')
    return source,code

# Preserve every preparation, source hash, host deadline and release gate.
exec(compile(inspect.getsource(parent.main),__file__+':same-complete-driver','exec'),globals())
if __name__=='__main__':main()
