"""Retain actual native protected-instance names in CDC evidence.

Only vendor-prefix recognition changes; endpoint classification, widths,
nonempty70-bit Gray checks, reset review and held-data deadlines stay intact.
"""
import inspect
from run_units import sha
from native_split_cdc import map_structure
import startup_cause_clock_cdc as original

def inventory(board):
    code=inspect.getsource(original.inventory)
    marker=' scope=dict(old.__dict__,structure=structure);exec'
    assert code.count(marker)==1
    insertion=" source=source.replace(\"end.startswith('u_ddr3_top/controller/')\",\"re.match(r'^(?:protected_inst/)*u_ddr3_top/controller/',end)\")\n"
    insertion+=" source=source.replace(\"start.startswith('u_ddr3_top/controller/')\",\"re.match(r'^(?:protected_inst/)*u_ddr3_top/controller/',start)\")\n"
    code=code.replace(marker,insertion+marker)
    scope=dict(original.__dict__,map_structure=map_structure,__file__=__file__)
    exec(compile(code,__file__+':native-prefix','exec'),scope)
    r=scope['inventory'](board)
    r['native_boundary_auditor_sha256']=sha(__file__)
    r['all_endpoint_names_retained']=True
    return r
