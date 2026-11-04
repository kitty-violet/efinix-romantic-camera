"""Strict offline bundle for OFS1 observation of the unchanged full system."""
import argparse,json,re,shutil,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from first_fault_observer import BASE,audit_delta
from prepare_first_fault_board import PROOF,check_proof
from audit_canonical_multiseed import audit
from audit_snapshot_jtag_paths import audit as audit_jtag
from snapshot_clock_cdc import inventory

PARENT=ROOT/'output/deployment/board-full-150-ownership-20261010'

def main():
    p=argparse.ArgumentParser()
    for n in ('board','batch','bitroot','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();board=a.board.resolve();batch=a.batch.resolve();bitroot=a.bitroot.resolve();out=a.output.resolve()
    assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
    delta=audit_delta(board);check_proof();evidence={}
    def bind(p):
        evidence[str(p)]=sha(p);return json.loads(p.read_text())
    def verify(rows):
        for p,h in rows.items():assert sha(p)==h,('Evidence mismatch',p);evidence[p]=h
    parent=bind(PARENT/'manifest.json')
    assert parent['board']==str(BASE) and parent['compute_clock_mhz']==150
    verify(parent['required_evidence']);verify({str(PARENT/n):h for n,h in parent['bundle_files'].items()})
    terminal=bind(batch/'result.json');assert terminal['status']=='FIRST_FAULT_SELECTED_ROUTE_PASS_PENDING_OFFLINE_BUNDLE'
    selected=audit(Path(terminal['selected']['canonical']));assert selected==terminal['selected'] and selected['current_timing_pass']
    physical=Path(selected['physical']);gray=Path(terminal['gray'])
    if (physical/'cnn_ram_placement.json').exists():
        from fixed_cnn_ram_locations import verify_placed
        assert bind(physical/'cnn_ram_actual_locations.json')==verify_placed(physical)
        ram=bind(physical/'cnn_ram_placement.json')
        reference=Path(ram['reference_canonical']);reference_physical=Path(ram['reference_physical'])
        assert audit(reference)['current_timing_pass']
        verify({str(reference/'result.json'):ram['reference_canonical_result_sha256'],
                str(reference_physical/'outflow/ti60f225_oob.out.placeloc'):ram['reference_placement_sha256'],
                str(reference_physical/'outflow/ti60f225_oob.map.v'):ram['reference_map_sha256'],
                str(physical/'fixed_dsp.placeloc'):ram['assignments_sha256'],
                str(HERE/'fixed_cnn_ram_locations.py'):ram['generator_sha256'],
                str(physical/'cnn_ram_generator.py'):ram['generator_sha256']})
        driver=batch.parent/(batch.name+'-reference-runner.py');receipt=bind(driver.with_suffix('.json'))
        verify({str(driver):receipt['generated_sha256'],str(batch/'runner.py'):receipt['generated_sha256'],
                receipt['source']:receipt['source_sha256'],
                str(HERE/'run_first_fault_reference_physical.py'):receipt['wrapper_sha256']})
        assert Path(receipt['reference'])==reference and receipt['reference_result_sha256']==sha(reference/'result.json')
        reused=bind(batch/'reused_startup.json');origin=Path(reused['root'])
        assert origin==Path(receipt['reuse_startup'])
        verify({str(origin/'result.json'):receipt['reuse_result_sha256']})
        assert reused['result_sha256']==receipt['reuse_result_sha256']
        assert reused['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
        probe=physical.parent/(physical.name+'-ram-probe.py');pr=bind(probe.with_suffix('.json'))
        verify({str(probe):pr['generated_sha256'],str(physical/'physical_runner.py'):pr['generated_sha256'],
                pr['original_runner']:pr['original_sha256'],
                str(HERE/'probe_control_state_ram_physical.py'):pr['wrapper_sha256']})
        evidence[str(physical/'efx_pnr_settings.ini')]=sha(physical/'efx_pnr_settings.ini')
    mapped=bind(board/'result.json');meta=bind(board/'inputs.json')
    assert mapped['status']=='FIRST_FAULT_FULL_BOARD_MAP_NOT_RELEASE'
    assert selected['identity']['map']==mapped['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert mapped['inputs_sha256']==sha(board/'inputs.json')
    for row in meta['inputs']:verify({row['snapshot']:row['sha256']})
    for root,status in ((PROOF,'FIRST_FAULT_FOUR_STATE_SHORT_PASS'),
                        (ROOT/'tmp/frame-execution/first-fault-scan1','FIRST_FAULT_SCAN_CDC_HOST_PASS'),
                        (batch/'startup','FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS')):
        result=bind(root/'result.json');inputs=bind(root/'inputs.json')
        assert result['status']==status and result['inputs_sha256']==sha(root/'inputs.json')
        assert sha(root/'simulation.log')==result.get('log_sha256',result.get('simulation_sha256'))
        evidence[str(root/'simulation.log')]=sha(root/'simulation.log')
        verify({str(root/n):h for n,h in inputs['files'].items()});verify(inputs.get('original',{}))
        if root==batch/'startup':assert inputs['original_map_sha256']==mapped['map_sha256']
    # Repeat structural reset/periphery checks on this exact map; no inherited
    # physical implementation is allowed to substitute for the new one.
    import audit_control_state_implementation as impl
    impl.audit_delta=audit_delta
    current=impl.audit(board,physical);recorded=bind(batch/'implementation.json')
    for key in ('status','reset_ff','snapshot_sync','pins','delta'):assert recorded[key]==current[key]
    assert current['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS'
    verify(recorded['required_evidence'])
    g=bind(gray/'result.json');gm=bind(gray/'inputs.json');gi=bind(gray/'clock_crossing_inventory.json')
    assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
    assert Path(gm['base'])==physical and g['inputs_sha256']==sha(gray/'inputs.json')
    assert g['buses']==10 and g['endpoints']==70 and g['max_skew_ns']<g['limit_ns']==2.5
    assert gi==json.loads(json.dumps(inventory(gray))) and g['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
    j=bind(gray/'jtag_cdc_audit.json');assert j==audit_jtag(gray) and j['unknown_new_endpoints']==0
    verify(gm['original']);verify(j['reports'])
    for row in gi['reports']:verify({row['path']:row['sha256']})
    bm=bind(bitroot/'generation_inputs.json');br=bind(bitroot/'result.json')
    assert br['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED' and br['inputs_sha256']==sha(bitroot/'generation_inputs.json')
    assert Path(bm['base'])==physical and Path(bm['canonical'])==Path(selected['canonical'])
    bit=bitroot/'outflow/ti60f225_oob.bit'
    item=next(v for k,v in br['artifacts'].items() if Path(k).name=='ti60f225_oob.bit')
    assert sha(bit)==item['sha256'] and bit.stat().st_size==item['bytes']
    verify({str(physical/'outflow'/n):h for n,h in bm['input_geometry'].items()})
    # PGM consumes LBF/LPF, so bind those actual inputs as well as route geometry.
    for name in ('work_pnr/ti60f225_oob.lbf','outflow/ti60f225_oob.lpf',
                 'outflow/ti60f225_oob.raminfo.pb'):
        assert sha(physical/name)==sha(bitroot/name),('PGM input differs from selected route',name)
        evidence[str(physical/name)]=sha(physical/name)
        evidence[str(bitroot/name)]=sha(bitroot/name)
    evidence[str(bitroot/'outflow/ti60f225_oob.pgm.out')]=sha(bitroot/'outflow/ti60f225_oob.pgm.out')
    cr=bind(Path(selected['canonical'])/'result.json');cm=bind(Path(selected['canonical'])/'inputs.json')
    verify(cm['original']);verify({str(Path(selected['canonical'])/'outflow'/n):h for n,h in cr['reports'].items()})
    for path in (physical/'physical_inputs.json',batch/'inputs.json',batch/'stages.json',board/'delta_audit.json',
                 board/'control_state_map_audit.json',board/'snapshot_sync_audit.json',ROOT/'tmp/board-test/program-bringup-status.py'):
        evidence[str(path)]=sha(path)
    for name in ('first_fault_observer.py','prepare_first_fault_board.py','read_first_fault.py','test_first_fault_host.py',
                 'validate_first_fault_scan.py','validate_first_fault_mapped.py','run_first_fault_physical.py',
                 'read_control_snapshot.py','prepare_first_fault_release.py'):
        evidence[str(HERE/name)]=sha(HERE/name)
    tests=batch/'first_fault_host_tests.log';assert not tests.exists()
    run=subprocess.run([sys.executable,'-m','unittest','test_first_fault_host','-v'],cwd=HERE,capture_output=True,text=True,timeout=30)
    tests.write_text(run.stdout+run.stderr)
    assert run.returncode==0 and 'Ran 9 tests' in tests.read_text() and tests.read_text().rstrip().endswith('OK')
    evidence[str(tests)]=sha(tests)
    out.mkdir(parents=True)
    shutil.copy2(bit,out/'ti60f225_oob.bit')
    for n in ('coefficients-padded.bin','compute.bin','dma.bin'):shutil.copy2(PARENT/n,out/n)
    (out/'README.md').write_text('''# 完整150MHz系统：首次故障只读观察包

保留摄像头、DDR、B1-A CNN与HDMI主链，时钟、引脚和主功能控制不变。
本次仅把原来事后观察改为OFS1首次故障记录：原始10位状态、前一拍状态/地址/握手、启动状态、时刻。
同源数值/协议证据沿用经精确差异审计的父版本，新增记录四态、跨时钟读出、实际新网表启动/捕获及新布局时序单独检查。
64拍停顿记录是提前取证触发，不等同于新增功能故障判决。首次记录冻结至SW1复位。
用户监督下临时JTAG下载后，只使用read_first_fault.py读取selectors2/3；不发APB命令、不写DDR、不装载权重、不运行CNN、不写Flash。
已附A资产仅保留部署一致性，不能据此跳过启动检查。当前尚未证明板上正常启动，30fps、完整画面及其他剩余职责未通过。
program.py默认仅离线校验。USB每次重新枚举物理B，不固定COM编号。
''',encoding='utf-8')
    reader="import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(HERE))+")\nfrom read_first_fault import main\nif __name__=='__main__':main()\n"
    (out/'read_first_fault.py').write_text(reader)
    manifest=dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',board=str(board),batch=str(batch),
        selected_seed=selected['seed'],bitstream=str(out/'ti60f225_oob.bit'),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
        resources=selected['resources'],wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'],compute_clock_mhz=150,
        observation_delta=delta,observation_format='OFS1',functional_parent=str(PARENT),functional=parent['functional'],
        gray_max_skew_ns=g['max_skew_ns'],held_payload_max_ns=j['max_held_data_delay_ns'],unknown_new_crossings=0,
        full_video_hardware_preserved=True,required_evidence=evidence,user_supervision_required=True,hardware_access=False,
        automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,a_weight_load_released=False,full_system_accepted=False,
        bundle_files={n:sha(out/n) for n in ('README.md','ti60f225_oob.bit','coefficients-padded.bin','compute.bin','dma.bin','read_first_fault.py')})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');verify(evidence)
    # The existing exercised official programming wrapper is identical except
    # for frozen bundle/manifest/bit hashes. Its validate() enforces this bundle.
    wrapper=(PARENT/'program.py').read_text()
    old=json.loads((PARENT/'manifest.json').read_text())
    for previous,new in ((sha(PARENT/'manifest.json'),sha(out/'manifest.json')),(old['bitstream_sha256'],sha(bit))):
        assert wrapper.count(previous)==1;wrapper=wrapper.replace(previous,new)
    compile(wrapper,str(out/'program.py'),'exec');(out/'program.py').write_text(wrapper,encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],bundle=str(out),bit_sha256=sha(bit),wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'])))
if __name__=='__main__':main()
