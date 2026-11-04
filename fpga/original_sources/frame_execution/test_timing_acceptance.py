"""Guard the user-authorized policy change against accepting real violations."""
import json
from timing_acceptance import timing_passes,CHECKS

zero=dict.fromkeys(CHECKS,0)
cases=[('current_original',0.154,0.026,zero,0,True),
       ('deferred_target',0.154,0.026,zero,0.3,False),
       ('zero_boundary',0,0,zero,0,True),
       ('negative_setup',-0.001,0.026,zero,0,False),
       ('negative_hold',0.154,-0.001,zero,0,False),
       ('old_failed_route',-3.298,0.019,zero,0,False),
       ('invalid_number',float('nan'),0.026,zero,0,False)]
for name in sorted(CHECKS):
    cases.append((name,0.154,0.026,dict(zero,**{name:1}),0,False))
for name,setup,hold,checks,margin,expected in cases:
    assert timing_passes(setup,hold,checks,margin)==expected,name
for args in [(0.154,0.026,{},0),(0.154,0.026,zero,-0.001)]:
    try:timing_passes(*args)
    except AssertionError:pass
    else:raise AssertionError(('Must reject invalid policy/incomplete checks',args))
print(json.dumps(dict(status='TIMING_ACCEPTANCE_POLICY_REGRESSION_PASS',cases=len(cases)+2)))
