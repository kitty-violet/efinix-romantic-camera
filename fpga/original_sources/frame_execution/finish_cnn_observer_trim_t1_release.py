"""Same full release gates; route-only timing recipe is explicitly bound."""
from pathlib import Path
from control_command_repair import once
source=Path(__file__).with_name('finish_cnn_observer_trim_free_release.py')
body=source.read_text(encoding='utf8')
body=body.replace('finish_cnn_observer_trim_free_release','finish_cnn_observer_trim_t1_release')
body=body.replace('cnn_observer_trim_free_route','cnn_observer_trim_t1_route')
body=once(body,"code=code_for('minimal')","code=code_for('minimal')\ncode=once(code,\"dict(before='CONGESTION_2',after='CONGESTION_2')\",\"dict(before='CONGESTION_2',after='TIMING_1')\")")
body=once(body,'_observer_package_parent=package_code', '''    canonical=batch/'canonical-seed3'
    cm=bind(canonical/'inputs.json')
    verify({str(HERE/'canonical_snapshot_sta_isolated.py'):cm['runner_sha256'],str(HERE/'canonical_snapshot_sta.py'):cm['canonical_source_sha256']})
    assert cm['process_isolation']=='timing_then_checks_both_required'
    stages=bind(canonical/'isolated_stages.json')
    assert [s['name'] for s in stages]==['sta','checks'] and all(s['returncode']==0 for s in stages)
    for stage in stages:verify({str(canonical/(stage['name']+'.log')):stage['log_sha256']})

_observer_package_parent=package_code''')
exec(compile(body,str(source)+':timing-route-full-gates','exec'),globals())
