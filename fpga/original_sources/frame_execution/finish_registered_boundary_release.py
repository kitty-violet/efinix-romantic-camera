"""Rebind established release gates to registered datapaths; no hardware access."""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from registered_boundary_board import BOARD,BASE as DATAPATH_PARENT,audit_delta
from control_command_repair import once
from finish_split_logic_outputs_release import archived_placement_evidence

BASE=ROOT/'tmp/frame-execution/connected-board-startup-cause4'
PARENT=ROOT/'output/deployment/board-full-150-startup-cause-20261011'
SHORTS=ROOT/'tmp/frame-execution/split-registered-evidence1'
RTL=SHORTS/'rtl'
SERIAL=ROOT/'tmp/frame-execution/split-registered-evidence1-serial/new'
MAP_AUDIT=ROOT/'tmp/frame-execution/split-registered-placement-checks1/map-audit'
ADAPTER=ROOT/'tmp/frame-execution/split-registered-adapter1'
STATUS='LOCAL_DECODE_ROUTED_ORIGINAL_STA_GRAY_PASS_PENDING_RELEASE_AUDITS'

def parent_evidence(parent,verify):
    assert Path(parent['board'])==BASE
    verify(parent['required_evidence'])
    audit_delta(BOARD)
    return dict(status='RELEASE_PARENT_AND_EXPLICIT_DATAPATH_DELTA_BOUND',
        manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
        scope='Pins/clocks/protectedIP and acceptedA assets inherited; zero-COMB and two registered datapaths have separate current evidence. No inherited board or throughput PASS.')

def extra_evidence(board,batch,physical,serial,bind,verify):
    assert board==BOARD and audit_delta(board)==bind(board/'delta_audit.json')
    mapped=bind(board/'result.json');ma=bind(MAP_AUDIT/'result.json')
    assert ma['map_sha256']==mapped['map_sha256'] and ma['board_inputs_sha256']==sha(board/'inputs.json')
    assert ma['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS' and ma['checks']==17504
    verify(ma['evidence'])
    for name,checker in (('new_diagnostic_cells.json','audit_split_logic_outputs_map.py'),
                          ('held_data_structure.json','split_logic_outputs_cdc.py')):
        r=bind(MAP_AUDIT/name);assert r['map_sha256']==mapped['map_sha256']
        verify({str(HERE/checker):r['auditor_sha256']})
    # The unchanged diagnostic RTL uses its old15-case evidence. The two
    # changed datapaths have their own four-state and real-A adjacent proofs.
    unit=ROOT/'tmp/frame-execution/split-outputs-shorts2'
    ur=bind(unit/'result.json');ui=bind(unit/'inputs.json')
    assert ur['status']=='STARTUP_CAUSE_TRACE_CELLS_SERIAL_ADAPTER_PASS' and len(ur['results'])==15
    assert ur['inputs_sha256']==sha(unit/'inputs.json')
    assert ui['board_inputs_sha256']==sha(DATAPATH_PARENT/'inputs.json')
    verify({str(unit/n):h for n,h in ui['files'].items()})
    verify({str(unit/x['log']):x['sha256'] for x in ur['results']})
    assert all(x['passes'] for x in ur['results'])
    for folder,status in ((SHORTS/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),
                          (ADAPTER,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
        r=bind(folder/'result.json');i=bind(folder/'inputs.json')
        assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
        assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped['map_sha256']
        verify({str(folder/n):h for n,h in i['files'].items()})
        verify({str(folder/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
    import finish_local_decode_release as old
    code=inspect.getsource(old.extra_evidence)
    code='def inherited_physical(board,batch,physical,serial,bind,verify):\n'+code[code.index("    si=bind(serial/'inputs.json')"):]
    code=code.replace('from inline_official_state_luts import inline','from inline_fault_probe_luts import inline')
    code=code.replace("HERE/'inline_official_state_luts.py'","HERE/'inline_fault_probe_luts.py'")
    code=code.replace("len(details['instances'])==23","len(details['instances'])==proof['primitive_instances_removed']")
    code=once(code,"verify(route['frozen']);verify(route['required_evidence'])",
        "verify(route['frozen']);verify(archived_placement_evidence(route['required_evidence'],physical,bind))")
    code=once(code,"verify(review['required_evidence'])",
        "verify(archived_placement_evidence(review['required_evidence'],physical,bind))")
    needle="    verify({str(driver):driver_meta['driver_sha256'],driver_meta['source']:driver_meta['source_sha256'],str(wrapper):driver_meta['wrapper_sha256']})"
    code=once(code,needle,"    wrapper=physical.parent/(physical.name+'-place-wrapper.py')\n"+needle)
    scope=dict(old.__dict__,archived_placement_evidence=archived_placement_evidence)
    exec(compile(code,__file__+':inherited-physical','exec'),scope)
    scope['inherited_physical'](board,batch,physical,serial,bind,verify)
    driver=batch.parent/(batch.name+'-driver.py');dm=bind(driver.with_suffix('.json'))
    verify({str(driver):dm['driver_sha256'],dm['source']:dm['source_sha256'],str(HERE/'registered_boundary_route.py'):dm['wrapper_sha256']})
    wrapper=physical.parent/(physical.name+'-place-wrapper.py');wm=bind(wrapper.with_suffix('.json'))
    verify({str(wrapper):wm['driver_sha256'],wm['source']:wm['source_sha256'],str(HERE/'place_registered_boundary_board.py'):wm['wrapper_sha256']})
    inv=bind(batch/'gray-seed3/clock_crossing_inventory.json')
    assert inv['cell_probe_bundled_settling']['max_delay_ns']<13.332
    assert inv['cell_probe_structure']['map_sha256']==mapped['map_sha256']
    completed=bind(SHORTS/'result.json')
    assert completed['status']=='REGISTERED_BOUNDARY_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert completed['map_sha256']==mapped['map_sha256'];verify(completed['evidence'])
    alias=bind(SERIAL.parent.parent/(SERIAL.parent.name+'-reference-binding.json'))
    assert alias['map_sha256']==mapped['map_sha256'] and not alias['hardware_source_changed'] and not alias['assertions_removed']
    verify({str(HERE/'registered_boundary_flow.py'):alias['wrapper_sha256']})
    a=ROOT/'tmp/frame-execution/split-registered-accepted-cluster1'
    ar=bind(a/'result.json');ai=bind(a/'inputs.json')
    assert ar['status']=='REGISTERED_BOUNDARY_ACCEPTED_A_CLUSTER_PASS'
    assert ar['inputs_sha256']==sha(a/'inputs.json') and ai['board_inputs_sha256']==sha(board/'inputs.json')
    verify({str(a/'simulation.log'):ar['log_sha256']})
    verify({x['snapshot']:x['sha256'] for x in ai['inputs']})
    verify({ai['binary']:ai['binary_sha256']});verify(ai['vectors'])
    from accepted_a_assets import accepted_package,PACKAGE_SHA
    accepted_package();assert ai['package_sha256']==ar['package_sha256']==PACKAGE_SHA
    verify({str(HERE/'accepted_a_assets.py'):ai['checker_sha256'],
            str(HERE/'validate_registered_accepted_cluster.py'):ai['runner_sha256']})
    gate=ROOT/'tmp/frame-execution/split-registered-edges-shorts4'
    assert bind(gate/'result.json')['status']=='RQ_CYCLE_EQUIVALENT_AND_SOURCE_NUMERIC_SHORTS_PASS'
    for kind,name in (('rq','fixed_rq32_mixed.sv'),('source','fixed_source_columns6.sv')):
        folder=gate/kind;i=bind(folder/'inputs.json');r=bind(folder/'result.json')
        verify({x['snapshot']:x['sha256'] for x in i['inputs']})
        assert sha(folder/name)==sha(board/'assembly'/name)
        verify({str(folder/f'simulation{j}.log'):x['log_sha256'] for j,x in enumerate(r['results'])})
    for n in ('registered_boundary_board.py','registered_boundary_flow.py','registered_boundary_route.py',
              'review_registered_boundary_route.py','run_registered_boundary_evidence.py','place_registered_boundary_board.py',
              'prepare_split_registered_board.py','finish_registered_boundary_release.py','audit_split_logic_outputs_gray.py',
              'split_logic_outputs_cdc.py','read_startup_cause.py','read_startup_cause_snapshot.py','read_startup_cause_status.py'):
        verify({str(HERE/n):sha(HERE/n)})

def package_code():
    import finish_local_decode_release as old
    source,code=old.package_code()
    changes={
        'from local_control_decode import BASE,audit_delta\nfrom finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence':
        'from registered_boundary_board import audit_delta\nfrom finish_registered_boundary_release import BASE,SHORTS,RTL,SERIAL,extra_evidence,parent_evidence\nimport local_control_decode\nlocal_control_decode.audit_delta=audit_delta',
        'output/deployment/board-full-150-command-progress-20261010':'output/deployment/board-full-150-startup-cause-20261011',
        "verify({str(board/'runner.py'):mi['runner_sha256']})":"verify({str(HERE/'prepare_split_registered_board.py'):mi['registered_boundary_transform_sha256']})",
        'from snapshot_clock_cdc import inventory':'from split_logic_outputs_cdc import inventory',
        "'test_control_ddr_host','-v'":"'test_control_ddr_host','test_fault_probe_host','test_startup_cause_host','-v'",
        "'Ran 24 tests'":"'Ran 40 tests'",
        "for name in ('read_control_ddr_progress','read_control_ddr_status'):":"for name in ('read_startup_cause_snapshot','read_startup_cause_status'):",
        "name=='read_control_ddr_status'":"name=='read_startup_cause_status'",
        "str(HERE/'finish_local_decode_release.py'):driver_meta['adapter_sha256']":"str(HERE/'finish_registered_boundary_release.py'):driver_meta['adapter_sha256']",
        "functional=parent['functional'],":"functional=parent['functional'],functional_evidence_scope='Parent evidence retained; modified datapaths separately checked, no inherited throughput claim',accepted_tail_profile=parent['accepted_tail_profile'],registered_datapath_boundaries=True,independent_outputs_workaround=True,",
        '局部精确解码和OFP3独立DDR观察':'零COMB4与RQ/source寄存边界的完整视频/A候选',
        '保持全局打包，仅十状态精确解码使用23个单输出LUT；保留命令两级FF、首错和进度，增加DDR公开状态及事件观察。':
        '使用独立LUT输出，RQ反馈/输出及source增加已验证寄存边界；保留完整视频/A、150MHz、保护IP、原引脚与约束。本版本尚待监督读取实板启动与DDR状态。'}
    for before,after in changes.items():code=once(code,before,after)
    code=code.replace('OFP3','OFP5').replace('450位','1026位')
    code=code.replace('read_control_ddr_progress.py只读','read_startup_cause_snapshot.py只读')
    code=code.replace('read_control_ddr_status.py读','read_startup_cause_status.py读')
    return source,code

def finish_code():
    source=HERE/'finish_command_progress_release.py';code=source.read_text()
    code=once(code,"'COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE'",repr(STATUS))
    code=once(code,"gray=Path(terminal['gray'])","gray=batch/'gray-seed3'")
    code=once(code,"'audit_command_progress_pll.py',['--board'","'finish_registered_boundary_release.py',['pll','--board'")
    code=once(code,"'prepare_command_progress_release.py',['--board'","'finish_registered_boundary_release.py',['bundle','--board'")
    code=once(code,"'--output',a.output.resolve()],90)","'--output',a.output.resolve()],240)")
    return source,code

def main():
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='bundle':source,code=package_code();key='--output'
    else:
        assert mode=='finish';source,code=finish_code();key='--root'
    args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve()
    audit=audit_delta(BOARD);dp=BOARD/'delta_audit.json'
    if dp.exists():assert json.loads(dp.read_text())==audit
    else:dp.write_text(json.dumps(audit,indent=2),encoding='utf8')
    driver=out.parent/(out.name+'-registered-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
