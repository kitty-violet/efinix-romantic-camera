"""Original complete release gates, plus minimal-selector numerical evidence."""
from pathlib import Path
from run_units import ROOT,HERE,sha
from minimal_feedback_board import audit_delta
from minimal_feedback_flow import bind as bind_source

SHORTS=ROOT/'tmp/frame-execution/minimal-feedback-mapped1'
RTL=ROOT/'tmp/frame-execution/minimal-feedback-ofp3-rtl1'
EQUIV=ROOT/'tmp/frame-execution/minimal-feedback-control-equivalence1'
SERIAL=ROOT/'tmp/frame-execution/minimal-feedback-mapped1-serial/new'
RQ_SHORT=ROOT/'tmp/frame-execution/minimal-feedback-shorts1'
CLUSTER=ROOT/'tmp/frame-execution/minimal-feedback-cluster1'

def configure(module):
    for name in ('SHORTS','RTL','EQUIV','SERIAL'):setattr(module,name,globals()[name])

def parent_evidence(parent,verify):
    from finish_local_decode_release import parent_evidence as original
    return original(parent,verify)

def extra_evidence(board,batch,physical,serial,bind,verify):
    import finish_local_decode_release as original
    configure(original);original.extra_evidence(board,batch,physical,serial,bind,verify)
    assert audit_delta(board)==bind(board/'delta_audit.json')
    import audit_feedback_selection_map as selection
    selection.audit_delta=audit_delta
    assert selection.audit(board)==bind(board/'feedback_selection_map_audit.json')
    from cached_head_flow import proofs
    proofs(board,RQ_SHORT)
    result=bind(RQ_SHORT/'result.json');inp=bind(RQ_SHORT/'inputs.json')
    verify({r['path']:r['sha256'] for r in inp['snapshots']})
    for row in result['results']:
        verify({str(RQ_SHORT/row['kind']/f"simulation-{row['shared_feedback']}-{row['seed']}.log"):row['log_sha256']})
    result=bind(CLUSTER/'result.json');inp=bind(CLUSTER/'inputs.json')
    assert result['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS' and result['inputs_sha256']==sha(CLUSTER/'inputs.json')
    assert inp['frozen_board']['inputs_sha256']==sha(board/'inputs.json')
    verify({r['snapshot']:r['sha256'] for r in inp['inputs']})
    verify({str(CLUSTER/'vectors'/n):h for n,h in inp['vectors'].items()})
    verify({str(CLUSTER/'simulation.log'):result['log_sha256']})
    adapter=bind(CLUSTER.parent/(CLUSTER.name+'-driver.json'))
    verify({str(CLUSTER.parent/(CLUSTER.name+'-driver.py')):adapter['driver_sha256'],adapter['source']:adapter['source_sha256'],
            adapter['tb_source']:adapter['tb_source_sha256'],adapter['testbench']:adapter['testbench_sha256'],
            str(HERE/'validate_cached_head_cluster.py'):adapter['adapter_sha256']})
    driver=batch.parent/(batch.name+'-driver.py');meta=bind(driver.with_suffix('.json'))
    verify({str(driver):meta['driver_sha256'],meta['source']:meta['source_sha256']})
    if 'reroute_adapter' in meta:
        verify({meta['reroute_adapter']:meta['reroute_adapter_sha256']})
        inherited=bind(physical/'route_inheritance.json')
        assert inherited['status']=='EXACT_PLACEMENT_REUSED_FOR_ROUTE_PROFILE_COMPARISON'
        assert all(inherited[k] is False for k in ('placement_executed','rtl_changed','constraints_changed','seed_changed','hardware_access'))
        verify(inherited['source_files']);verify(inherited['copied_files'])
        old=Path(inherited['source'])
        verify({str(old/'physical_inputs.json'):inherited['source_physical_inputs_sha256'],
            str(old/'before_route/result.json'):inherited['source_placement_result_sha256'],
            inherited['source_route_result']:inherited['source_route_result_sha256'],
            inherited['generator']:inherited['generator_sha256']})
        assert inherited['route_profile_cli_override']==dict(before='CONGESTION_2',after='TIMING_1')
        assert 'optimization_level=TIMING_1' in (physical/'route_only.cmd').read_text()
    else:
        verify({str(HERE/'cached_head_flow.py'):meta['wrapper_sha256']})
    for name in ('minimal_feedback_board.py','minimal_feedback_flow.py','rq_feedback_select_board.py',
                 'feedback_select_flow.py','audit_feedback_selection_map.py','validate_rq_cached_head.py',
                 'finish_minimal_feedback_release.py'):
        verify({str(HERE/name):sha(HERE/name)})

def main():
    bind_source()
    import finish_local_decode_release as original
    original.audit_delta=audit_delta;configure(original)
    package=original.package_code;execute=original.execute
    def new_package():
        source,code=package()
        code=code.replace('from finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence',
                          'from finish_minimal_feedback_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence')
        code=code.replace('局部精确解码和OFP3独立DDR观察','原队列保留、寄存任务类型反馈选择和OFP3独立DDR观察')
        return source,code
    def new_execute(source,code,driver,args):
        code=code.replace("'finish_local_decode_release.py',['pll'","'finish_minimal_feedback_release.py',['pll'")
        code=code.replace("'finish_local_decode_release.py',['bundle'","'finish_minimal_feedback_release.py',['bundle'")
        return execute(source,code,driver,args)
    original.package_code=new_package;original.execute=new_execute;original.main()
if __name__=='__main__':main()
