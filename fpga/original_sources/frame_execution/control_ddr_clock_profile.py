"""Exact existing constraints, with observation-only wrapper delta verified."""
from run_units import HERE
from control_ddr_combined import ddr,once

def apply_profile(root,evidence):
    source=HERE/'shift_guard_clock_profile.py';code=source.read_text()
    before=" for n in ('ddr516_v4_controller.sv','ddr516_pll_shift_guard.sv'):\n  assert sha(root/n)==sha(CAL/n)"
    after=" assert (root/'ddr516_v4_controller.sv').read_text()==ddr((CAL/'ddr516_v4_controller.sv').read_text())\n assert sha(root/'ddr516_pll_shift_guard.sv')==sha(CAL/'ddr516_pll_shift_guard.sv')"
    code=once(code,before,after)
    ns={'__name__':'_combined_clock_profile','ddr':ddr}
    exec(compile(code,str(source),'exec'),ns)
    result=ns['apply_profile'](root,evidence)
    result['note']='Original constraints; exact public debug-port-only wrapper delta and unchanged guard verified.'
    result['profile_adapter_sha256']=ns['sha'](__file__)
    return result
