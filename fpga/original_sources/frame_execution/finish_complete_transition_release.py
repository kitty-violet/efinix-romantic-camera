"""Bind new controller evidence and unchanged accepted-A datapath; no board I/O."""
import inspect,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from complete_control_transition import audit_delta,BASE

SHORTS=ROOT/'tmp/frame-execution/complete-transition-mapped2'
RTL=ROOT/'tmp/frame-execution/complete-transition-ofp3-rtl2'
EQUIV=ROOT/'tmp/frame-execution/complete-transition-equivalence2'
SERIAL=ROOT/'tmp/frame-execution/complete-transition-mapped2-serial/new'
BOUNDARY=ROOT/'tmp/frame-execution/complete-transition-boundary2'
PARENT=ROOT/'output/deployment/board-full-150-accepted-tail-20261011'

def parent_evidence(parent,verify):
    assert Path(parent['board'])==BASE
    verify(parent['required_evidence'])
    return dict(status='EXACT_ACCEPTED_TAIL_PARENT_EVIDENCE_UNCHANGED',
        manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
        inherited_evidence_scope='Unchanged A arithmetic, assets, video and DDR. New controller requires its own tests and implementation.')

def configure(module):
    for n in ('SHORTS','RTL','EQUIV','SERIAL'):setattr(module,n,globals()[n])

def extra_evidence(board,batch,physical,serial,bind,verify):
    import finish_local_decode_release as original
    # Same checks with the new equivalence result's explicitly different name;
    # retain the historical implementation and its evidence unchanged.
    code=inspect.getsource(original.extra_evidence)
    old='LOCAL_EXACT_DECODE_FOURSTATE_AND_CONTROLLER_EQUIVALENCE_PASS'
    assert code.count(old)==1
    code=code.replace(old,'COMPLETE_TRANSITION_CYCLE_EQUIVALENCE_PASS')
    scope=dict(original.__dict__)
    for n in ('SHORTS','RTL','EQUIV','SERIAL'):scope[n]=globals()[n]
    exec(compile(code,str(HERE/'finish_local_decode_release.py')+':new-status','exec'),scope)
    scope['extra_evidence'](board,batch,physical,serial,bind,verify)
    assert audit_delta(board)==bind(board/'delta_audit.json')
    r=bind(BOUNDARY/'result.json');i=bind(BOUNDARY/'inputs.json')
    assert r['status']=='COMPLETE_TRANSITION_LOCAL_STRUCTURE_AND_UPSTREAM_BOUNDARY_PASS'
    assert r['inputs_sha256']==sha(BOUNDARY/'inputs.json')
    assert i['board_inputs_sha256']==sha(board/'inputs.json')
    assert r['map_sha256']==i['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    verify({str(BOUNDARY/'boolean_result.json'):r['boolean_result_sha256'],
        str(BOUNDARY/'mapped_d_test.py'):i['driver_sha256'],
        str(ROOT/'tmp/board-test/check-accepted-tail-nextstate.py'):i['source_reference_sha256'],
        str(HERE/'review_complete_transition_boundary.py'):i['reviewer_sha256'],
        str(board/'complete_transition_map_audit.json'):i['strict_audit_sha256']})
    from complete_transition_flow import proof
    folder=ROOT/'tmp/frame-execution/complete-transition-short2';proof(board,folder)
    s=bind(folder/'result.json');si=bind(folder/'inputs.json')
    verify({str(folder/n):h for n,h in si['files'].items()})
    verify({str(folder/'simulation.log'):s['log_sha256']})
    driver=batch.parent/(batch.name+'-driver.py');dm=bind(driver.with_suffix('.json'))
    verify({str(driver):dm['driver_sha256'],dm['source']:dm['source_sha256'],str(HERE/'cached_head_flow.py'):dm['wrapper_sha256']})
    for n in ('complete_control_transition.py','complete_transition_flow.py','complete_transition_physical_flow.py',
        'review_complete_transition_boundary.py','audit_complete_transition_map.py','validate_complete_transition.py',
        'run_complete_transition_evidence.py','finish_complete_transition_release.py'):
        verify({str(HERE/n):sha(HERE/n)})

def main():
    import local_control_decode
    local_control_decode.audit_delta=audit_delta
    import finish_local_decode_release as original
    original.audit_delta=audit_delta;configure(original)
    package=original.package_code;execute=original.execute
    def new_package():
        source,code=package()
        pairs=[
          ("verify({str(board/'runner.py'):mi['runner_sha256']})",
           "verify({str(HERE/'complete_control_transition.py'):mi['complete_transition_transform_sha256']})"),
          ('from local_control_decode import BASE,audit_delta','from complete_control_transition import BASE,audit_delta'),
          ('from finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence',
           'from finish_complete_transition_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence'),
          ('output/deployment/board-full-150-command-progress-20261010','output/deployment/board-full-150-accepted-tail-20261011'),
          ('局部精确解码和OFP3独立DDR观察','完整控制状态转换局部替代及OFP3观察'),
          ('保持全局打包，仅十状态精确解码使用23个单输出LUT；保留命令两级FF、首错和进度，增加DDR公开状态及事件观察。',
           '保留十状态精确解码的23个单输出LUT，下一状态使用25个单输出LUT；只改控制器状态转换，保留命令两级FF、首错、进度及DDR公开观察。全上游仍有1个写适配器COMB4，影响边界已核对；不宣称物理根因已证明。'),
          ("functional=parent['functional'],", "functional=parent['functional'],accepted_tail_profile=parent['accepted_tail_profile'],controller_transition_evidence=dict(path=str(board/'complete_transition_map_audit.json'),whole_cone_zero_comb4=False,board_test_pending=True),")]
        for before,after in pairs:
            assert code.count(before)==1,before
            code=code.replace(before,after)
        return source,code
    def new_execute(source,code,driver,args):
        code=code.replace("'finish_local_decode_release.py',['pll'","'finish_complete_transition_release.py',['pll'")
        code=code.replace("'finish_local_decode_release.py',['bundle'","'finish_complete_transition_release.py',['bundle'")
        return execute(source,code,driver,args)
    original.package_code=new_package;original.execute=new_execute;original.main()
if __name__=='__main__':main()
