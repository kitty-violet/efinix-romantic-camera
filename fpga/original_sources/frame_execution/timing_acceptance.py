"""Nonnegative real timing is mandatory; optional extra margin is explicit."""
import math

CHECKS={'no_clock','unconstrained_internal_endpoints','no_input_delay',
        'no_output_delay','multiple_clock'}

def timing_passes(setup_ns,hold_ns,checks,min_setup_ns=0.0):
    assert math.isfinite(min_setup_ns) and min_setup_ns>=0
    assert set(checks)==CHECKS
    assert all(isinstance(v,int) and v>=0 for v in checks.values())
    return (math.isfinite(setup_ns) and math.isfinite(hold_ns)
            and setup_ns>=min_setup_ns and hold_ns>=0
            and all(v==0 for v in checks.values()))
