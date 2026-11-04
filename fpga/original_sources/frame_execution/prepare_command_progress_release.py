"""Bind the exact full OFP2 implementation and offline proofs; never download."""
import argparse,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import BASE,audit_delta
from audit_canonical_multiseed import audit as canonical_audit
from audit_functional_jtag_paths import audit as functional_cdc
from audit_snapshot_jtag_paths import audit as snapshot_cdc
from snapshot_clock_cdc import inventory

PARENT=ROOT/'output/deployment/board-full-150-ddr-shift-guard-20261010'

def main():
    p=argparse.ArgumentParser()
    for name in ('board','batch','paired','shiftpaths','bitroot','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();board=a.board.resolve();batch=a.batch.resolve();paired=a.paired.resolve();shift=a.shiftpaths.resolve();bitroot=a.bitroot.resolve();out=a.output.resolve()
    assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
    delta=audit_delta(board);evidence={}
    def bind(path):
        evidence[str(path)]=sha(path);return json.loads(path.read_text())
    def verify(rows):
        for path,digest in rows.items():assert sha(path)==digest,('Evidence changed',path);evidence[str(path)]=digest
    parent=bind(PARENT/'manifest.json');assert Path(parent['board'])==BASE
    verify(parent['required_evidence']);verify({str(PARENT/n):h for n,h in parent['bundle_files'].items()})
    terminal=bind(batch/'result.json');assert terminal['status']=='COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE'
    selected=canonical_audit(Path(terminal['selected']['canonical']));assert selected==terminal['selected'] and selected['current_timing_pass']
    physical=Path(selected['physical']);canonical=Path(selected['canonical']);gray=Path(terminal['gray'])
    mapped=bind(board/'result.json');mi=bind(board/'inputs.json')
    assert selected['identity']['map']==mapped['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert mapped['inputs_sha256']==sha(board/'inputs.json') and not mapped['comb_loop_warning'] and not mapped['new_undriven_signatures']
    verify({r['snapshot']:r['sha256'] for r in mi['inputs']});verify({str(board/'runner.py'):mi['runner_sha256']})
    from audit_command_toggle_sync import audit as sync
    command_sync=sync((board/'outflow/ti60f225_oob.map.v').read_text())
    assert command_sync==bind(batch/'command_sync.json')
    mapped_sync=bind(board/'command_sync_audit.json')
    assert Path(mapped_sync['netlist']).resolve()==board/'outflow/ti60f225_oob.map.v'
    assert mapped_sync['sha256']==mapped['map_sha256']
    assert {k:v for k,v in mapped_sync.items() if k not in ('netlist','sha256')}==command_sync
    short=ROOT/'tmp/frame-execution/command-progress-short2'
    serial=Path(terminal['serial'])
    for folder,status in ((short,'COMMAND_PROGRESS_RTL_SERIAL_RESET_PASS'),(batch/'startup','COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT'),(serial/'new','MAPPED_SERIAL_APB_OFS_RESET_PASS'),(paired/'new','MAPPED_SERIAL_APB_OFS_RESET_PASS')):
        result=bind(folder/'result.json');inputs=bind(folder/'inputs.json')
        assert result['status']==status and result['inputs_sha256']==sha(folder/'inputs.json')
        verify({str(folder/n):h for n,h in inputs['files'].items()})
        verify({str(folder/'simulation.log'):result.get('log_sha256',result.get('simulation_sha256'))})
        if 'original_map_sha256' in inputs:assert inputs['original_map_sha256']==mapped['map_sha256']
        if 'source_map_sha256' in inputs:assert inputs['source_map_sha256']==mapped['map_sha256']
        if 'board_inputs_sha256' in inputs:assert inputs['board_inputs_sha256']==sha(board/'inputs.json')
    counter=bind(serial/'progress_counter_checks.json');assert counter['status']=='REAL_MAPPED_PROGRESS_COUNTERS_PASS'
    assert counter['source_result_sha256']==sha(serial/'new/result.json')
    pair=bind(paired/'paired_reply_result.json');assert pair['status']=='MAPPED_PAIRED_REPLY_STARTUP_FAULT_RESET_PASS'
    assert pair['source_map_sha256']==mapped['map_sha256'] and pair['result_sha256']==sha(paired/'new/result.json')
    import audit_control_state_implementation as impl
    impl.audit_delta=audit_delta
    current=impl.audit(board,physical);recorded=bind(batch/'implementation.json')
    assert current==recorded;verify(recorded['required_evidence'])
    cr=bind(canonical/'result.json');cm=bind(canonical/'inputs.json')
    verify(cm['original']);verify({str(canonical/'outflow'/n):h for n,h in cr['reports'].items()})
    g=bind(gray/'result.json');gm=bind(gray/'inputs.json');gi=bind(gray/'clock_crossing_inventory.json')
    assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE' and Path(gm['base'])==physical
    assert g['inputs_sha256']==sha(gray/'inputs.json') and g['buses']==10 and g['endpoints']==70 and g['max_skew_ns']<2.5
    assert gi==json.loads(json.dumps(inventory(gray))) and g['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
    j=bind(gray/'jtag_cdc_audit.json');assert j==snapshot_cdc(gray)
    f=bind(gray/'functional_jtag_cdc.json');assert f==functional_cdc(gray)
    verify(gm['original']);verify(gm['scripts']);verify(j['reports']);verify(f['reports']);verify({r['path']:r['sha256'] for r in gi['reports']})
    sb=bind(shift/'bound.json');si=bind(shift/'inputs.json');sr=bind(shift/'result.json')
    assert sb['status']=='PLL_SHIFT_ROUTED_SETUP_HIGH_BOUND_PASS' and Path(si['base'])==canonical
    assert sb['source_inputs_sha256']==sha(board/'inputs.json') and sb['reports_inputs_sha256']==sr['inputs_sha256']==sha(shift/'inputs.json')
    from prepare_ddr_shift_guard_board import CAL,proof
    proof();assert sb['vendor_result_sha256']==sha(CAL/'result.json')
    verify(si['frozen']);verify(sb['reports'])
    gap=bind(Path(sb['min_gap_result']));assert sb['min_gap_result_sha256']==sha(sb['min_gap_result']) and gap['status']=='PLL_SHIFT_MINIMUM_GAP_PASS'
    from audit_pll_shift_routed import bounds
    b=sb['bounds'];assert bounds(shift,b['logical_setup_ns'],b['logical_high_ns'],b['logical_low_ns'])==b
    assert b['setup_lower_ns']>=2.5 and min(b['common_control_high_lower_ns'],b['common_control_low_lower_ns'])>=10
    bm=bind(bitroot/'generation_inputs.json');br=bind(bitroot/'result.json')
    assert br['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED' and br['inputs_sha256']==sha(bitroot/'generation_inputs.json')
    assert Path(bm['base'])==physical and Path(bm['canonical'])==canonical
    bit=bitroot/'outflow/ti60f225_oob.bit';item=br['artifacts']['outflow/ti60f225_oob.bit'] if 'outflow/ti60f225_oob.bit' in br['artifacts'] else br['artifacts']['outflow\\ti60f225_oob.bit']
    assert sha(bit)==item['sha256'] and bit.stat().st_size==item['bytes']
    verify({str(physical/'outflow'/n):h for n,h in bm['input_geometry'].items()})
    for name in ('work_pnr/ti60f225_oob.lbf','outflow/ti60f225_oob.lpf','outflow/ti60f225_oob.raminfo.pb'):
        assert sha(physical/name)==sha(bitroot/name);verify({str(physical/name):sha(physical/name),str(bitroot/name):sha(bitroot/name)})
    bi=bind(batch/'inputs.json');verify({str(batch/'runner.py'):bi['runner_sha256']})
    if (batch/'route_continuation.json').exists():
        continuation=bind(batch/'route_continuation.json')
        assert terminal['route_continuation_sha256']==sha(batch/'route_continuation.json')
        assert all(continuation[k] is False for k in ('rtl_changed','map_changed','placement_changed','constraints_changed'))
        verify(continuation['frozen'])
        verify({str(physical/'route_only.cmd'):continuation['command_sha256'],continuation['official_doc']:continuation['official_doc_sha256']})
    if (batch/'reused_short_tests.json').exists():
        reuse=bind(batch/'reused_short_tests.json');assert reuse['map_sha256']==mapped['map_sha256']
        verify({str(Path(reuse['startup'])/'result.json'):reuse['startup_result_sha256'],str(Path(reuse['serial'])/'new/result.json'):reuse['serial_result_sha256']})
        driver=batch.parent/(batch.name+'-driver.py');dm=bind(driver.with_suffix('.json'))
        verify({str(driver):dm['generated_sha256'],dm['source']:dm['source_sha256'],str(HERE/'run_command_progress_seed.py'):dm['wrapper_sha256']})
    # Control/reset/snapshot structure is recomputed and bound above through
    # batch/implementation.json. This candidate emits no legacy per-board
    # control_state_map_audit.json or snapshot_sync_audit.json files.
    for path in (batch/'stages.json',physical/'physical_inputs.json',board/'delta_audit.json',bitroot/'outflow/ti60f225_oob.pgm.out'):
        evidence[str(path)]=sha(path)
    scripts=('control_command_repair.py','prepare_control_command_board.py','read_control_progress.py','read_command_progress_status.py','stable_jtag_reply.py','test_control_progress_host.py','test_command_progress_status.py','test_stable_jtag_reply.py','audit_command_toggle_sync.py','audit_functional_jtag_paths.py','prepare_command_progress_release.py')
    for name in scripts:evidence[str(HERE/name)]=sha(HERE/name)
    testlog=batch/'host_release_tests.log';assert not testlog.exists()
    test=subprocess.run([sys.executable,'-m','unittest','test_control_progress_host','test_command_progress_status','test_stable_jtag_reply','-v'],cwd=HERE,capture_output=True,text=True,timeout=30)
    testlog.write_text(test.stdout+test.stderr);assert test.returncode==0 and 'Ran 18 tests' in testlog.read_text() and testlog.read_text().rstrip().endswith('OK');evidence[str(testlog)]=sha(testlog)
    out.mkdir(parents=True);shutil.copy2(bit,out/bit.name)
    for name in ('coefficients-padded.bin','compute.bin','dma.bin'):shutil.copy2(PARENT/name,out/name)
    for name in ('read_control_progress','read_command_progress_status'):
        (out/(name+'.py')).write_text('import sys\nsys.path.insert(0,'+repr(str(HERE))+')\nfrom '+name+' import main\nif __name__=="__main__":main()\n') if name=='read_command_progress_status' else (out/(name+'.py')).write_text('import runpy,sys\nsys.path.insert(0,'+repr(str(HERE))+')\nif __name__=="__main__":runpy.run_path('+repr(str(HERE/(name+'.py')))+',run_name="__main__")\n')
    (out/'README.md').write_text('''# 完整150MHz系统：命令同步器修正和OFP2当前状态

保留完整摄像头、DDR、B1-A和HDMI硬件；频率、引脚及原约束不变。
命令两级同步器强制映射为两个真实FF；低192位首错记录保留，新增128位当前状态与命令计数。
所有本版离线门禁和资源/时序绑定于manifest。它是可监督临时下载的包，不是已通过板测的结论。
program.py默认只校验；只有用户在旁监督时才临时JTAG，不写Flash。
初次使用read_control_progress.py只读独立快照，再使用read_command_progress_status.py读身份和启动/DDR状态；后者需两次连续应答一致，超时仍尝试取独立快照，宿主总限20秒。
禁止混用旧OFS1的258位读取器；本版为OFP2/386位。
本轮不发DDR数据请求、不装载权重、不运行CNN。确认控制和DDR校准后另行放行受限CRC与A装载。
尚未证明本版实板启动、DDR校准、真实画面或30fps；附A资产保持部署同源，不代表这些项已通过。
''',encoding='utf-8')
    manifest=dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',board=str(board),batch=str(batch),selected_seed=selected['seed'],bitstream=str(out/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
        resources=selected['resources'],wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'],compute_clock_mhz=150,repair_delta=delta,observation_format='OFP2',functional_parent=str(PARENT),functional=parent['functional'],
        gray_max_skew_ns=g['max_skew_ns'],held_payload_max_ns=j['max_held_data_delay_ns'],command_payload_max_ns=f['command_payload_max_ns'],functional_jtag_host_contract=f['host_contract'],pll_setup_lower_ns=b['setup_lower_ns'],pll_high_lower_ns=b['common_control_high_lower_ns'],pll_low_lower_ns=b['common_control_low_lower_ns'],
        full_video_hardware_preserved=True,required_evidence=evidence,user_supervision_required=True,hardware_access=False,automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,a_weight_load_released=False,full_system_accepted=False,
        bundle_files={n.name:sha(n) for n in out.iterdir() if n.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));verify(evidence)
    wrapper=(PARENT/'program.py').read_text()
    for old,new in ((sha(PARENT/'manifest.json'),sha(out/'manifest.json')),(parent['bitstream_sha256'],sha(bit))):
        assert wrapper.count(old)==1;wrapper=wrapper.replace(old,new)
    compile(wrapper,str(out/'program.py'),'exec');(out/'program.py').write_text(wrapper)
    print(json.dumps(dict(status=manifest['status'],bundle=str(out),wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'],bit_sha256=sha(bit))))

if __name__=='__main__':main()
