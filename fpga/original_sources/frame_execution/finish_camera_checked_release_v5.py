"""Full release for eight-route preservation, including analysis-copy lineage."""
from run_units import HERE,sha
from control_command_repair import once
from pathlib import Path

def canonical_rcf_evidence(batch,bind,verify):
    c=batch/'canonical-seed3';i=bind(c/'inputs.json');r=bind(c/'result.json')
    assert r['status']=='CANONICAL_ROUTED_TIMING_PASS_NOT_RELEASE'
    driver=batch/'canonical-seed3-sparse-canonical.py';m=bind(driver.with_suffix('.json'))
    verify({str(driver):m['generated_sha256'],m['source']:m['source_sha256'],
        str(HERE/'canonical_sparse_snapshot_v2.py'):m['adapter_sha256']})
    assert Path(m['source'])==HERE/'canonical_snapshot_sta.py'
    assert i['runner_sha256']==sha(driver)==sha(c/'runner.py')
    assert r['inputs_sha256']==sha(c/'inputs.json')
    assert i['removed_setup_margin'] is None and i['min_setup_slack_ns']==i['min_hold_slack_ns']==0
    verify(m['sparse_evidence'])
    code=Path(m['source']).read_text()
    code=once(code,"('.xml','.sdc','.ini','.placeloc')","('.xml','.sdc','.ini','.placeloc','.rcf','.troutingtraces')")
    code=once(code,"frozen[str(canonical)]=sha(canonical)","frozen.update("+repr(m['sparse_evidence'])+")\nfrozen[str(canonical)]=sha(canonical)")
    assert code==driver.read_text()
    policy=sha(HERE/'timing_acceptance.py')
    assert i['timing_policy_sha256']==policy
    verify({str(c/'timing_acceptance.py'):policy,str(batch/'timing_acceptance.py'):policy})
    for name in ('canonical_sparse_snapshot.py','canonical_sparse_snapshot_v2.py','audit_camera_sparse_route.py',
                 'audit_camera_sparse_route_v2.py','camera_checked_gray_rcf.py','survey_pll_shift_routed_rcf.py','camera_sparse_route_launch.py'):
        verify({str(HERE/name):sha(HERE/name)})

body=(HERE/'finish_camera_checked_release_v4.py').read_text()
body=body.replace('from audit_camera_sparse_route import','from audit_camera_sparse_route_v2 import')
body=body.replace('finish_camera_checked_release_v4','finish_camera_checked_release_v5')
body=body.replace('sparse_audit(physical,bind,verify)")',r'sparse_audit(physical,bind,verify)\\n canonical_rcf_evidence(batch,bind,verify)")')
body=body.replace('CAMERA_CHECKED_RELEASE_SPARSE.md','CAMERA_CHECKED_RELEASE_SPARSE2.md')
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
exec(compile(prefix+tail,__file__+':v4-gates','exec'),globals())

import finish_local_driver_release as old_finish
original_finish=old_finish.finish_code
def with_pll_collateral():
    source,code=original_finish()
    code=once(code,"'survey_pll_shift_routed.py'","'survey_pll_shift_routed_rcf.py'")
    return source,code
old_finish.finish_code=with_pll_collateral

if __name__=='__main__':main()
