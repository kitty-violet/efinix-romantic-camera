"""C2 route on an explicitly reviewed process-targeted placement; original SDC."""
from pathlib import Path
import sys
from control_command_repair import once
source=Path(__file__).with_name('cnn_observer_trim_t1_route.py')
body=source.read_text()
# This adapter restores C2 in both preparation and routing, while retaining
# the already exercised isolated complete STA and unchanged release checks.
body=body.replace("'after':'TIMING_1'","'after':'CONGESTION_2'")
body=body.replace("after='TIMING_1'","after='CONGESTION_2'")
body=once(body," code=once(code,'optimization_level=CONGESTION_2','optimization_level=TIMING_1')\n",'')
body=body.replace('OBSERVER_TIMING_ROUTE_SAME_MAP_PLACEMENT_SDC_GATES_COMPILE','OBSERVER_MARGIN_PLACEMENT_C2_ROUTE_GATES_COMPILE')
exec(compile(body,str(source)+':margin-placement-c2-route','exec'),globals())
