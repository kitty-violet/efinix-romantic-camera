"""Reuse the full release gates with this candidate's separately bound evidence.

No hardware access. Original failed simulation/physical records stay unchanged.
The generated adapter is recorded alongside its source hash for review.
"""
import hashlib,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from local_control_decode import audit_delta,once

SHORTS=ROOT/'tmp/frame-execution/local-decode-shorts2'
RTL=ROOT/'tmp/frame-execution/local-decode-ofp3-rtl1'
EQUIV=ROOT/'tmp/frame-execution/local-decode-equivalence4'
SERIAL=ROOT/'tmp/frame-execution/local-decode-serial-inline1'
STATUS='LOCAL_DECODE_ROUTED_ORIGINAL_STA_GRAY_PASS_PENDING_RELEASE_AUDITS'

def parent_evidence(parent,verify):
    """Only the explicitly added routing-review policy supersedes its old hash."""
    policy=ROOT/'AGENTS.md';rows=dict(parent['required_evidence'])
    old_hash=rows.pop(str(policy));raw=policy.read_bytes()
    assert old_hash=='b6dec3447af96408a98ea64ee7ac8d60357d23b8e503148c4201465415e1b137'
    current='03484bb50083a44f203988c0096686e8e72115610d6a7878544e978dcf222a7d'
    assert sha(policy)==current
    start=raw.index('## 布线感知的布局前置审查'.encode('utf8'))
    end=raw.index('## 全局长仿真启动规则'.encode('utf8'))
    assert hashlib.sha256(raw[:start]+raw[end:]).hexdigest()==old_hash
    verify(rows);verify({str(policy):current})
    return dict(path=str(policy),parent_sha256=old_hash,current_sha256=current,
        change='Only the user-requested routing-aware placement review section was added; all pre-existing policy bytes unchanged.',
        historical_manifest_unchanged=True,all_other_parent_evidence_unchanged=True)

def extra_evidence(board,batch,physical,serial,bind,verify):
    """Recompute local truth table, source binding, and simulation-only expansion."""
    from audit_local_control_decode_map import audit
    assert audit(board)==bind(board/'local_decode_map_audit.json')
    mapped=bind(board/'result.json')
    for folder,status in ((EQUIV,'LOCAL_EXACT_DECODE_FOURSTATE_AND_CONTROLLER_EQUIVALENCE_PASS'),
                          (SHORTS/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS')):
        result=bind(folder/'result.json');meta=bind(folder/'inputs.json')
        assert result['status']==status and result['inputs_sha256']==sha(folder/'inputs.json')
        verify({str(folder/n):h for n,h in meta['files'].items()})
        verify({str(folder/'simulation.log'):result.get('log_sha256',result.get('simulation_sha256'))})
        if 'board_inputs_sha256' in meta:assert meta['board_inputs_sha256']==sha(board/'inputs.json')
        if 'source_map_sha256' in meta:assert meta['source_map_sha256']==mapped['map_sha256']
    si=bind(serial/'inputs.json');source=Path(si['source_replay']);proof=si['simulation_only_inline']
    original=bind(source/'inputs.json');assert sha(source/'inputs.json')==si['source_inputs_sha256']
    assert original['files']==proof['original_files']
    verify({str(source/n):h for n,h in proof['original_files'].items()})
    assert proof['transform_sha256']==sha(HERE/'inline_official_state_luts.py')
    from inline_official_state_luts import inline
    expanded,details=inline((source/'mapped_control.v').read_text(),source/'efx_lut4.v',lambda n:n.startswith('\\cpu/ctrl_decode/'))
    assert (serial/'mapped_control.v').read_text()==expanded and len(details['instances'])==23
    assert proof['official_model_sha256']==sha(serial/'efx_lut4.v')==sha(source/'efx_lut4.v')
    for n,h in proof['original_files'].items():
        if n!='mapped_control.v':assert sha(serial/n)==h
    route=bind(batch/'inputs.json');assert Path(route['physical'])==physical and Path(route['board'])==board
    assert route['board_inputs_sha256']==sha(board/'inputs.json')
    verify(route['frozen']);verify(route['required_evidence'])
    review=bind(Path(route['review']));assert sha(route['review'])==route['review_sha256']
    assert review['decision']=='PROCEED_TO_BOUNDED_ROUTE'
    verify(review['required_evidence'])
    assert review['placement_review_sha256']==sha(physical/'placement_review_inputs.json')
    placement=bind(physical/'placement_review_inputs.json')
    assert placement['status']=='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW'
    physical_meta=bind(physical/'physical_inputs.json')
    verify({str(physical/'physical_runner.py'):physical_meta['runner_sha256']})
    driver=physical.parent/(physical.name+'-place-driver.py');driver_meta=bind(driver.with_suffix('.json'))
    grouped=physical_meta.get('dsp_layout',{}).get('layout')=='four_quadrants_with_central_rq'
    wrapper=HERE/('probe_local_decode_quadrant_place.py' if grouped else
        'probe_local_decode_place_v2.py' if 'automatic_cnn_ram' in driver_meta else 'probe_local_decode_place.py')
    verify({str(driver):driver_meta['driver_sha256'],driver_meta['source']:driver_meta['source_sha256'],str(wrapper):driver_meta['wrapper_sha256']})
    if grouped:
        from audit_grouped_dsp_locations import audit as audit_grouped
        assert audit_grouped(physical)==bind(physical/'grouped_dsp_actual_locations.json')
        for path in (physical/'quadrant_dsp_plan.json',HERE/'fixed_quadrant_dsp_locations.py',HERE/'audit_grouped_dsp_locations.py'):
            verify({str(path):sha(path)})
    for n in ('route_only.cmd','cnn_ram_actual_locations.json','fixed_dsp.placeloc'):
        verify({str(physical/n):sha(physical/n)})
    # Bind the actual adapters, public-model transform, and host readers.
    for n in ('local_control_decode.py','audit_local_control_decode_map.py','validate_local_control_decode.py',
              'validate_local_decode_board.py','run_local_decode_shorts.py','probe_local_decode_place.py',
              'route_local_decode_board.py','inline_official_state_luts.py','replay_local_decode_serial_inline.py',
              'replay_mapped_serial_modelsim.py','read_control_ddr_progress.py','read_control_ddr_status.py',
              'test_control_ddr_host.py','control_ddr_clock_profile.py','finish_local_decode_release.py'):
        verify({str(HERE/n):sha(HERE/n)})

def package_code():
    source=HERE/'prepare_command_progress_release.py';code=source.read_text()
    pairs=[
        ('from control_command_repair import BASE,audit_delta','from local_control_decode import BASE,audit_delta\nfrom finish_local_decode_release import SHORTS,RTL,SERIAL,extra_evidence,parent_evidence'),
        ('output/deployment/board-full-150-ddr-shift-guard-20261010','output/deployment/board-full-150-command-progress-20261010'),
        ("verify(parent['required_evidence']);verify({str(PARENT/n):h for n,h in parent['bundle_files'].items()})",
         "policy_update=parent_evidence(parent,verify);verify({str(PARENT/n):h for n,h in parent['bundle_files'].items()})"),
        ("terminal=bind(batch/'result.json');assert terminal['status']=='COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE'",
         "terminal=bind(batch/'result.json');assert terminal['status']=='"+STATUS+"'\n    terminal={**terminal,'gray':str(batch/'gray-seed3'),'serial':str(SERIAL)}\n    assert paired==SERIAL"),
        ("    assert command_sync==bind(batch/'command_sync.json')\n    mapped_sync=bind(board/'command_sync_audit.json')\n    assert Path(mapped_sync['netlist']).resolve()==board/'outflow/ti60f225_oob.map.v'\n    assert mapped_sync['sha256']==mapped['map_sha256']\n    assert {k:v for k,v in mapped_sync.items() if k not in ('netlist','sha256')}==command_sync",
         "    mapped_sync=bind(board/'local_decode_map_audit.json')\n    assert mapped_sync['map_sha256']==mapped['map_sha256']\n    assert mapped_sync['command_sync']==command_sync"),
        ("short=ROOT/'tmp/frame-execution/command-progress-short2'","short=RTL"),
        ("'COMMAND_PROGRESS_RTL_SERIAL_RESET_PASS'","'CONTROL_DDR_RTL_SERIAL_EVENTS_RESET_PASS'"),
        ("(batch/'startup','COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT')","(SHORTS/'startup','COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT')"),
        ("'REAL_MAPPED_PROGRESS_COUNTERS_PASS'","'CONTROL_DDR_MAPPED_SERIAL_IDLE_RESET_PASS'"),
        ("'test_stable_jtag_reply','-v'","'test_stable_jtag_reply','test_control_ddr_host','-v'"),
        ("'Ran 18 tests'","'Ran 24 tests'"),
        ("    out.mkdir(parents=True);shutil.copy2(bit,out/bit.name)",
         "    extra_evidence(board,batch,physical,serial,bind,verify)\n    driver_meta=bind(Path(__file__).with_suffix('.json'))\n    verify({str(Path(__file__)):driver_meta['generated_sha256'],driver_meta['source']:driver_meta['source_sha256'],str(HERE/'finish_local_decode_release.py'):driver_meta['adapter_sha256']})\n    out.mkdir(parents=True);shutil.copy2(bit,out/bit.name)"),
        ("for name in ('read_control_progress','read_command_progress_status'):","for name in ('read_control_ddr_progress','read_control_ddr_status'):"),
        ("name=='read_command_progress_status'","name=='read_control_ddr_status'")]
    for before,after in pairs:code=once(code,before,after)
    code=code.replace("serial/'new/result.json'","serial/'result.json'").replace("paired/'new/result.json'","paired/'result.json'")
    code=code.replace("serial/'new'","serial").replace("paired/'new'","paired")
    code=code.replace('OFP2','OFP3').replace('386位','450位')
    code=once(code,'functional=parent[\'functional\'],','functional=parent[\'functional\'],parent_policy_revalidation=policy_update,')
    code=once(code,'命令同步器修正和OFP3当前状态','局部精确解码和OFP3独立DDR观察')
    code=once(code,'命令两级同步器强制映射为两个真实FF；低192位首错记录保留，新增128位当前状态与命令计数。',
        '保持全局打包，仅十状态精确解码使用23个单输出LUT；保留命令两级FF、首错和进度，增加DDR公开状态及事件观察。')
    code=code.replace('read_control_progress.py只读','read_control_ddr_progress.py只读').replace('read_command_progress_status.py读','read_control_ddr_status.py读')
    return source,code

def execute(source,code,driver,args):
    assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*map(str,args)];runpy.run_path(str(driver),run_name='__main__')

def main():
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='bundle':
        args=sys.argv[1:];out=Path(args[args.index('--output')+1]).resolve()
        source,code=package_code();execute(source,code,out.parent/(out.name+'-package-driver.py'),args);return
    assert mode=='finish'
    args=sys.argv[1:];root=Path(args[args.index('--root')+1]).resolve()
    source=HERE/'finish_command_progress_release.py';code=source.read_text()
    code=once(code,"'COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE'",repr(STATUS))
    code=once(code,"gray=Path(terminal['gray'])","gray=batch/'gray-seed3'")
    code=once(code,"'audit_command_progress_pll.py',['--board'","'finish_local_decode_release.py',['pll','--board'")
    code=once(code,"'prepare_command_progress_release.py',['--board'","'finish_local_decode_release.py',['bundle','--board'")
    execute(source,code,root.parent/(root.name+'-local-driver.py'),args)

if __name__=='__main__':main()
