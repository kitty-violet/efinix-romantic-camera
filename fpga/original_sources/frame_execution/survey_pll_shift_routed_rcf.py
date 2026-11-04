"""Existing PLL arrival measurement, preserving RCF collateral in its copy."""
import inspect
import survey_pll_shift_routed as prior
from control_command_repair import once
code=once(inspect.getsource(prior.main),"('.xml','.sdc','.ini','.placeloc')","('.xml','.sdc','.ini','.placeloc','.rcf','.troutingtraces')")
env=dict(prior.__dict__,__file__=__file__)
exec(compile(code,__file__+':same-measurement-complete-copy','exec'),env)
if __name__=='__main__':env['main']()
