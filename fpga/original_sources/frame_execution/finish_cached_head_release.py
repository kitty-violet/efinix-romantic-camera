"""Keep every prior release gate, adding the same-source cached-head proofs."""
import json,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from rq_cached_head_board import audit_delta
from cached_head_flow import bind as bind_source,proofs

SHORTS=ROOT/'tmp/frame-execution/cached-head-mapped1'
RTL=ROOT/'tmp/frame-execution/cached-head-ofp3-rtl1'
EQUIV=ROOT/'tmp/frame-execution/cached-head-control-equivalence1'
SERIAL=ROOT/'tmp/frame-execution/cached-head-mapped1-serial/new'
RQ_SHORT=ROOT/'tmp/frame-execution/cached-head-shorts3'
CLUSTER=ROOT/'tmp/frame-execution/cached-head-cluster3'

def parent_evidence(parent,verify):
    import finish_local_decode_release as previous
    return previous.parent_evidence(parent,verify)

def extra_evidence(board,batch,physical,serial,bind,verify):
    import finish_local_decode_release as previous
    previous.SHORTS=SHORTS;previous.RTL=RTL;previous.EQUIV=EQUIV;previous.SERIAL=SERIAL
    previous.extra_evidence(board,batch,physical,serial,bind,verify)
    from audit_cached_head_map import audit
    assert audit(board)==bind(board/'cached_head_map_audit.json')
    proofs(board,RQ_SHORT)
    for path in (RQ_SHORT/'result.json',RQ_SHORT/'inputs.json'):
        bind(path)
    short=bind(RQ_SHORT/'inputs.json')
    verify({r['path']:r['sha256'] for r in short['snapshots']})
    for row in bind(RQ_SHORT/'result.json')['results']:
        verify({str(RQ_SHORT/row['kind']/f"simulation-{row['shared_feedback']}-{row['seed']}.log"):row['log_sha256']})
    result=bind(CLUSTER/'result.json');inputs=bind(CLUSTER/'inputs.json')
    assert result['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS' and result['inputs_sha256']==sha(CLUSTER/'inputs.json')
    assert inputs['frozen_board']['inputs_sha256']==sha(board/'inputs.json')
    verify({r['snapshot']:r['sha256'] for r in inputs['inputs']})
    verify({str(CLUSTER/'vectors'/n):h for n,h in inputs['vectors'].items()})
    verify({str(CLUSTER/'simulation.log'):result['log_sha256']})
    adapter=bind(CLUSTER.parent/(CLUSTER.name+'-driver.json'))
    verify({str(CLUSTER.parent/(CLUSTER.name+'-driver.py')):adapter['driver_sha256'],adapter['source']:adapter['source_sha256'],
            adapter['tb_source']:adapter['tb_source_sha256'],adapter['testbench']:adapter['testbench_sha256'],
            str(HERE/'validate_cached_head_cluster.py'):adapter['adapter_sha256']})
    route_driver=batch.parent/(batch.name+'-driver.py');route_adapter=bind(route_driver.with_suffix('.json'))
    verify({str(route_driver):route_adapter['driver_sha256'],route_adapter['source']:route_adapter['source_sha256'],
            str(HERE/'cached_head_flow.py'):route_adapter['wrapper_sha256']})
    for name in ('rq_cached_head_board.py','validate_rq_cached_head.py','audit_cached_head_map.py','finish_cached_head_release.py'):
        verify({str(HERE/name):sha(HERE/name)})

def main():
    bind_source()
    import finish_local_decode_release as previous
    previous.audit_delta=audit_delta
    previous.SHORTS=SHORTS;previous.RTL=RTL;previous.EQUIV=EQUIV;previous.SERIAL=SERIAL
    package=previous.package_code;execute=previous.execute
    def new_package():
        source,code=package()
        code=code.replace('from finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence',
                          'from finish_cached_head_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence')
        code=code.replace('局部精确解码和OFP3独立DDR观察','RQ队首缓存、局部精确解码和OFP3独立DDR观察')
        return source,code
    def new_execute(source,code,driver,args):
        code=code.replace("'finish_local_decode_release.py',['pll'","'finish_cached_head_release.py',['pll'")
        code=code.replace("'finish_local_decode_release.py',['bundle'","'finish_cached_head_release.py',['bundle'")
        return execute(source,code,driver,args)
    previous.package_code=new_package;previous.execute=new_execute
    previous.main()
if __name__=='__main__':main()
