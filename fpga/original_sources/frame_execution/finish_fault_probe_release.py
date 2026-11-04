"""Bind the observation-only OFP4 delta, actual mapped gates and final route.

The existing complete release validator is adapted without removing physical,
reset, CDC, accepted-A asset or supervised-programming requirements.
"""
import inspect,json,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from fault_probe_board import audit_delta,BASE
SHORTS=ROOT/'tmp/frame-execution/fault-probe-evidence1'
RTL=ROOT/'tmp/frame-execution/fault-probe-ofp4-rtl2'
SERIAL=ROOT/'tmp/frame-execution/fault-probe-evidence1-serial-fast2/new'
PARENT=ROOT/'output/deployment/board-full-150-complete-transition2-20261011'
MAP_AUDIT=ROOT/'tmp/frame-execution/fault-probe-map-audit2'
ADAPTER=ROOT/'tmp/frame-execution/fault-probe-adapter-idle4'

def parent_evidence(parent,verify):
    assert Path(parent['board'])==BASE
    verify(parent['required_evidence'])
    return dict(status='EXACT_COMPLETE_TRANSITION_PARENT_EVIDENCE_UNCHANGED',
        manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
        scope='Full video/A arithmetic, assets, controller, DDR PHY and constraints unchanged; new observation implementation separately verified.')

def extra_evidence(board,batch,physical,serial,bind,verify):
    from fault_probe_clock_cdc import map_pairs
    assert audit_delta(board)==bind(board/'delta_audit.json')
    mapped=bind(board/'result.json')
    ma=bind(MAP_AUDIT/'result.json')
    assert ma['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS'
    assert ma['map_sha256']==mapped['map_sha256'] and ma['board_inputs_sha256']==sha(board/'inputs.json')
    assert ma['next_state_luts']==25 and ma['checks']==17504
    assert ma['sync']==map_pairs((board/'outflow/ti60f225_oob.map.v').read_text())
    verify(ma['evidence'])
    from fault_probe_flow import proof
    unit=ROOT/'tmp/frame-execution/fault-probe-short2';proof(board,unit)
    ur=bind(unit/'result.json');ui=bind(unit/'inputs.json')
    verify({str(unit/n):h for n,h in ui['files'].items()})
    verify({str(unit/x['log']):x['sha256'] for x in ur['results']})
    for folder,status in ((SHORTS/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),
                          (ADAPTER,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
        r=bind(folder/'result.json');i=bind(folder/'inputs.json')
        assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
        assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped['map_sha256']
        verify({str(folder/n):h for n,h in i['files'].items()})
        verify({str(folder/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
    # Keep the parent's exact public-model inlining proof and physical-review
    # evidence validation. Only the new functional short-test set differs.
    import finish_local_decode_release as old
    code=inspect.getsource(old.extra_evidence)
    start=code.index("    si=bind(serial/'inputs.json')")
    code='def inherited_physical(board,batch,physical,serial,bind,verify):\n'+code[start:]
    code=code.replace('from inline_official_state_luts import inline','from inline_fault_probe_luts import inline')
    code=code.replace("HERE/'inline_official_state_luts.py'","HERE/'inline_fault_probe_luts.py'")
    code=code.replace("len(details['instances'])==23","len(details['instances'])==proof['primitive_instances_removed']")
    scope=dict(old.__dict__);exec(compile(code,old.__file__+':OFP4-physical','exec'),scope)
    scope['inherited_physical'](board,batch,physical,serial,bind,verify)
    driver=batch.parent/(batch.name+'-driver.py');dm=bind(driver.with_suffix('.json'))
    verify({str(driver):dm['driver_sha256'],dm['source']:dm['source_sha256'],str(HERE/'fault_probe_flow.py'):dm['wrapper_sha256']})
    for name in ('fault_probe_board.py','fault_probe_flow.py','fault_probe_clock_cdc.py','audit_fault_probe_gray.py',
        'audit_fault_probe_map.py','validate_fault_probe.py','validate_control_adapter_idle_mapped.py',
        'read_fault_probe.py','read_fault_probe_status.py','test_fault_probe_host.py','finish_fault_probe_release.py','inline_fault_probe_luts.py'):
        verify({str(HERE/name):sha(HERE/name)})

def main():
    import local_control_decode
    local_control_decode.audit_delta=audit_delta
    import finish_local_decode_release as original
    original.audit_delta=audit_delta
    for n in ('SHORTS','RTL','SERIAL'):setattr(original,n,globals()[n])
    package=original.package_code;execute=original.execute
    def new_package():
        source,code=package()
        pairs=[
          ("verify({str(board/'runner.py'):mi['runner_sha256']})",
           "verify({str(HERE/'fault_probe_board.py'):mi['fault_probe_transform_sha256']})"),
          ('from local_control_decode import BASE,audit_delta',
           'from fault_probe_board import BASE,audit_delta\nimport local_control_decode\nlocal_control_decode.audit_delta=audit_delta'),
          ('from finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence',
           'from finish_fault_probe_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence'),
          ('output/deployment/board-full-150-command-progress-20261010','output/deployment/board-full-150-complete-transition2-20261011'),
          ('from snapshot_clock_cdc import inventory','from fault_probe_clock_cdc import inventory'),
          ("'test_control_ddr_host','-v'","'test_control_ddr_host','test_fault_probe_host','-v'"),
          ("'Ran 24 tests'","'Ran 31 tests'"),
          ("for name in ('read_control_ddr_progress','read_control_ddr_status'):",
           "for name in ('read_fault_probe','read_fault_probe_status'):"),
          ("name=='read_control_ddr_status'","name=='read_fault_probe_status'"),
          ('局部精确解码和OFP3独立DDR观察','完整视频/A系统的适配器首错和DDR时钟观察'),
          ('保持全局打包，仅十状态精确解码使用23个单输出LUT；保留命令两级FF、首错和进度，增加DDR公开状态及事件观察。',
           '只新增只读观察：适配器首次错误前一周期的状态/地址/握手，DDR四时钟相对频率及复位/CKE历史；保持原功能语句。该包用于定位余下板测故障，不宣称DDR校准已修复。'),
          ("functional=parent['functional'],", "functional=parent['functional'],accepted_tail_profile=parent['accepted_tail_profile'],observation_only=True,adapter_and_ddr_board_test_pending=True,")]
        for before,after in pairs:
            assert code.count(before)==1,before
            code=code.replace(before,after)
        code=code.replace('OFP3','OFP4').replace('450位','770位')
        code=code.replace('read_control_ddr_progress.py只读','read_fault_probe.py只读').replace('read_control_ddr_status.py读','read_fault_probe_status.py读')
        return source,code
    def new_execute(source,code,driver,args):
        code=code.replace("'finish_local_decode_release.py',['pll'","'finish_fault_probe_release.py',['pll'")
        code=code.replace("'finish_local_decode_release.py',['bundle'","'finish_fault_probe_release.py',['bundle'")
        return execute(source,code,driver,args)
    original.package_code=new_package;original.execute=new_execute;original.main()
if __name__=='__main__':main()
