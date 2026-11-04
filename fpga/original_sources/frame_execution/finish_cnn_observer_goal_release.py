"""All original release gates plus exact restoration of the routing target."""
from pathlib import Path
from control_command_repair import once
source=Path(__file__).with_name('finish_cnn_observer_trim_t1_release.py')
body=source.read_text()
body=body.replace('finish_cnn_observer_trim_t1_release','finish_cnn_observer_goal_release')
body=body.replace('cnn_observer_trim_t1_route','cnn_observer_goal_route')
body=body.replace("after='TIMING_1'","after='CONGESTION_2'")
needle="    for stage in stages:verify({str(canonical/(stage['name']+'.log')):stage['log_sha256']})"
body=once(body,needle,needle+'''
    from cnn_routing_goal import verify_restored
    verify(verify_restored(physical))
    restore=bind(ROOT/'tmp/board-test/cnn-routing-goal-restore1/result.json')
    assert restore['status']=='NORMAL_AND_EXCEPTION_ROUTING_SDC_RESTORATION_PASS' and restore['cases']==2
    verify(restore['evidence'])
''')
exec(compile(body,str(source)+':strict-routing-goal-full-release','exec'),globals())
