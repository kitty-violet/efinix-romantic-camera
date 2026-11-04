"""Use only with a deployment manifest declaring this diagnostic profile.

The transport remains OFP5. Retired historical fields are unavailable, not
zero-valued measurements; this module performs no hardware access.
"""
from read_cnn_run_progress import decode as original

PROFILE='cnn-observer-live-levels-only-v1'

def decode(raw,slots):
    result=original(raw,slots)
    result.pop('clock_probe',None)
    result.pop('startup_trace',None)
    result['diagnostic_profile']=PROFILE
    result['unavailable_diagnostics']=['relative_clock_counts_and_ticks','reset_age_and_edges','startup_trace']
    result['clock_levels']=None
    if result['valid']:
        c=(raw>>2)>>576
        result['clock_levels']=dict(divided_levels=c>>120&15,
            ddr_reset_n=bool(c>>124&1),cke=bool(c>>125&1),system_reset_n=bool(c>>126&1),
            scope='Sampled levels only; no clock frequency or reset age measurement')
    return result
