"""Narrow adaptations of existing audits; inherited files remain frozen."""
import runpy,sys
from run_units import HERE
mode=sys.argv.pop(1)
if mode=='startup':
 import validate_first_fault_mapped as target
 from ddr_cal_observer import audit_delta
 target.audit_delta=audit_delta;target.main()
elif mode=='gray':
 import snapshot_clock_cdc
 from audit_ddr_observer import inventory
 snapshot_clock_cdc.inventory=inventory
 runpy.run_path(str(HERE/'audit_snapshot_routed_gray.py'),run_name='__main__')
else:raise ValueError(mode)
