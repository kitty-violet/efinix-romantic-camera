"""Offline DCO1/OFS1 bundle. Never access USB, Flash or board memory."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from ddr_cal_observer import BASE,audit_delta
from prepare_ddr_observer_board import PROOF,check_proof
from audit_canonical_multiseed import audit
from audit_snapshot_jtag_paths import audit as audit_jtag
from audit_ddr_observer import inventory,mapped as audit_observer
from fixed_cnn_ram_locations import verify_placed

PARENT=ROOT/'output/deployment/board-full-150-first-fault-20261010'


def main():
    p=argparse.ArgumentParser()
    for n in ('board','batch','bitroot','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();board=a.board.resolve();batch=a.batch.resolve();bitroot=a.bitroot.resolve();out=a.output.resolve()
    assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
    delta=audit_delta(board);check_proof();evidence={}
    def bind(path):
        evidence[str(path)]=sha(path)
        return json.loads(path.read_text())
    def verify(rows):
        for path,h in rows.items():
            assert sha(path)==h,('Evidence mismatch',path)
            evidence[str(path)]=h
    parent=bind(PARENT/'manifest.json')
    assert parent['board']==str(BASE) and parent['compute_clock_mhz']==150
    verify(parent['required_evidence']);verify({str(PARENT/n):h for n,h in parent['bundle_files'].items()})
    evidence[str(PARENT/'program.py')]=sha(PARENT/'program.py')
    terminal=bind(batch/'result.json')
    assert terminal['status']=='DDR_OBSERVER_SELECTED_ROUTE_PASS_PENDING_BUNDLE'
    if (batch/'reused_short.json').exists():
        for row in bind(batch/'reused_short.json'):
            source=Path(row['source']);verify({str(source/'result.json'):row['result_sha256']})
            assert sha(batch/row['name']/'result.json')==row['result_sha256']
    selected=audit(Path(terminal['selected']['canonical']))
    assert selected==terminal['selected'] and selected['current_timing_pass']
    physical=Path(selected['physical']);canonical=Path(selected['canonical']);gray=Path(terminal['gray'])
    if (batch/'route_continuation.json').exists():
        continuation=bind(batch/'route_continuation.json');source=Path(continuation['source'])
        assert terminal['route_continuation_sha256']==sha(batch/'route_continuation.json')
        assert bind(physical/'result.json')['continuation_sha256']==sha(batch/'route_continuation.json')
        assert continuation['hardware_access'] is False
        assert all(continuation[n] is False for n in ('rtl_changed','map_changed','constraints_changed'))
        verify(continuation['source_gate'])
        if continuation['source_kind']=='host_timeout':
            assert 'TimeoutExpired' in json.loads((source/'failure.json').read_text())['error']
        else:
            assert continuation['source_kind']=='legal_route_timing_fail'
            old_gate=audit(source.parent/'canonical-seed1')
            assert not old_gate['current_timing_pass'] and old_gate['whs_ns']>=0
        verify(continuation['copied_placement_and_inputs'])
        for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb',
                     'ti60f225_oob.sdc','efx_pnr_settings.ini','fixed_dsp.placeloc'):
            assert sha(source/name)==sha(physical/name),('Continuation changed frozen input',name)
            evidence[str(physical/name)]=sha(physical/name)
        change=continuation['optimization_change']
        if change:
            import xml.etree.ElementTree as ET
            tree=ET.parse(source/'ti60f225_oob.xml');nodes=[n for n in tree.iter() if n.get('name')=='optimization_level']
            assert len(nodes)==1 and nodes[0].get('value')==change['before']
            assert change['after'] in ('CONGESTION_1','CONGESTION_3')
            nodes[0].set('value',change['after'])
            assert ET.tostring(tree.getroot())==ET.tostring(ET.parse(physical/'ti60f225_oob.xml').getroot())
        else:assert sha(source/'ti60f225_oob.xml')==sha(physical/'ti60f225_oob.xml')
        evidence[str(physical/'ti60f225_oob.xml')]=sha(physical/'ti60f225_oob.xml')
        verify({str(physical/'route_only.cmd'):continuation['command_sha256']})
        command=continuation['command']
        assert command[-1]=='--route' and '--place' not in command
        assert command[command.index('--prj')+1]==str(physical/'ti60f225_oob.xml')
        from run_units import EF
        expected='@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\n'+subprocess.list2cmdline(command)+'\nexit /b %errorlevel%\n'
        assert (physical/'route_only.cmd').read_text()==expected
        route_log=(batch/'route_only.log').read_text()
        assert 'Routing took' in route_log and 'Placement took' not in route_log
        assert 'Finished writing bitstream file work_pnr' in route_log, 'Route-only flow must generate a fresh LBF'
        if continuation['source_kind']=='legal_route_timing_fail':
            for name in ('outflow/ti60f225_oob.troutingtraces','work_pnr/ti60f225_oob.lbf'):
                assert sha(source/name)!=sha(physical/name),('Reject stale routing/bitstream geometry',name)
                evidence[str(source/name)]=sha(source/name)
                evidence[str(physical/name)]=sha(physical/name)
        evidence[str(batch/'route_only.log')]=sha(batch/'route_only.log')
        continuation_meta=bind(batch/'inputs.json')
        verify({str(batch/'runner.py'):continuation_meta['runner_sha256'],
                str(HERE/'resume_ddr_observer_route.py'):continuation_meta['runner_sha256']})
    resource=selected['resources']
    assert resource  # audit() enforces device capacities and original constraints.
    ram=bind(physical/'cnn_ram_placement.json')
    assert bind(physical/'cnn_ram_actual_locations.json')==verify_placed(physical)
    reference=Path(ram['reference_canonical']);reference_physical=Path(ram['reference_physical'])
    assert audit(reference)['current_timing_pass']
    verify({str(reference/'result.json'):ram['reference_canonical_result_sha256'],
            str(reference_physical/'outflow/ti60f225_oob.out.placeloc'):ram['reference_placement_sha256'],
            str(reference_physical/'outflow/ti60f225_oob.map.v'):ram['reference_map_sha256'],
            str(physical/'fixed_dsp.placeloc'):ram['assignments_sha256'],
            str(HERE/'fixed_cnn_ram_locations.py'):ram['generator_sha256'],
            str(physical/'cnn_ram_generator.py'):ram['generator_sha256']})
    probe=physical.parent/(physical.name+'-ram-probe.py');pr=bind(probe.with_suffix('.json'))
    verify({str(probe):pr['generated_sha256'],str(physical/'physical_runner.py'):pr['generated_sha256'],
            pr['original_runner']:pr['original_sha256'],str(HERE/'probe_control_state_ram_physical.py'):pr['wrapper_sha256']})
    current_map=bind(board/'result.json');meta=bind(board/'inputs.json')
    assert current_map['status']=='DDR_OBSERVER_FULL_BOARD_MAP_NOT_RELEASE'
    assert selected['identity']['map']==current_map['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert current_map['inputs_sha256']==sha(board/'inputs.json')
    for row in meta['inputs']:verify({row['snapshot']:row['sha256']})
    for root,status in ((PROOF,'DDR_OBSERVER_FOUR_STATE_SHORT_PASS'),
                        (ROOT/'tmp/frame-execution/ddr-observer-jtag2','DDR_OBSERVER_REAL_JTAG_APB_PASS'),
                        (batch/'observer-mapped','DDR_OBSERVER_ACTUAL_MAP_SHORT_PASS'),
                        (batch/'startup','FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS'),
                        (batch/'mapped-jtag2','DDR_OBSERVER_MAPPED_JTAG_APB_PASS')):
        result=bind(root/'result.json');inputs=bind(root/'inputs.json')
        assert result['status']==status and result['inputs_sha256']==sha(root/'inputs.json')
        verify({str(root/'simulation.log'):result.get('log_sha256',result.get('simulation_sha256'))})
        verify({str(root/n):h for n,h in inputs['files'].items()})
        verify(inputs.get('original',{}))
        if root.is_relative_to(batch):assert inputs['original_map_sha256']==current_map['map_sha256']
    # Original control pins/reset, original snapshot CDC and added observer CDC
    # are audited again against this exact mapped/routed implementation.
    import audit_control_state_implementation as impl
    impl.audit_delta=audit_delta
    actual=impl.audit(board,physical);recorded=bind(batch/'implementation.json')
    for key in ('status','reset_ff','snapshot_sync','pins','delta'):assert recorded[key]==actual[key]
    assert actual['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS'
    verify(recorded['required_evidence'])
    observer=audit_observer(board/'outflow/ti60f225_oob.map.v')
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
    assert Path(bm['base'])==physical and Path(bm['canonical'])==canonical
    bit=bitroot/'outflow/ti60f225_oob.bit'
    artifact=next(v for k,v in br['artifacts'].items() if Path(k).name=='ti60f225_oob.bit')
    assert sha(bit)==artifact['sha256'] and bit.stat().st_size==artifact['bytes']
    verify({str(physical/'outflow'/n):h for n,h in bm['input_geometry'].items()})
    for name in ('work_pnr/ti60f225_oob.lbf','outflow/ti60f225_oob.lpf','outflow/ti60f225_oob.raminfo.pb'):
        assert sha(physical/name)==sha(bitroot/name),('PGM input differs',name)
        evidence[str(physical/name)]=sha(physical/name);evidence[str(bitroot/name)]=sha(bitroot/name)
    evidence[str(bitroot/'outflow/ti60f225_oob.pgm.out')]=sha(bitroot/'outflow/ti60f225_oob.pgm.out')
    cr=bind(canonical/'result.json');cm=bind(canonical/'inputs.json')
    verify(cm['original']);verify({str(canonical/'outflow'/n):h for n,h in cr['reports'].items()})
    for path in (physical/'physical_inputs.json',physical/'efx_pnr_settings.ini',batch/'inputs.json',batch/'stages.json',
                 batch/'runner.py',board/'delta_audit.json',board/'control_state_map_audit.json',board/'snapshot_sync_audit.json'):
        evidence[str(path)]=sha(path)
    for name in ('ddr_cal_observer.py','prepare_ddr_observer_board.py','validate_ddr_observer.py',
                 'validate_ddr_observer_jtag.py','validate_ddr_observer_mapped.py','validate_ddr_observer_mapped_jtag.py','audit_ddr_observer.py',
                 'run_ddr_observer_stage.py','run_ddr_observer_offline.py','prepare_ddr_observer_release.py',
                 'read_ddr_observer.py','test_ddr_observer_host.py'):
        evidence[str(HERE/name)]=sha(HERE/name)
    tests=batch/'ddr_observer_host_tests.log';assert not tests.exists()
    run=subprocess.run([sys.executable,'-m','unittest','test_ddr_observer_host','-v'],cwd=HERE,capture_output=True,text=True,timeout=30)
    tests.write_text(run.stdout+run.stderr)
    assert run.returncode==0 and 'Ran 8 tests' in tests.read_text() and tests.read_text().rstrip().endswith('OK')
    evidence[str(tests)]=sha(tests)
    out.mkdir(parents=True)
    shutil.copy2(bit,out/'ti60f225_oob.bit')
    for n in ('coefficients-padded.bin','compute.bin','dma.bin'):shutil.copy2(PARENT/n,out/n)
    (out/'README.md').write_text('''# 完整150MHz系统：DDR校准公开端口观察包

保留摄像头、DDR、B1-A CNN、HDMI及原OFS1首次故障记录。引脚、频率、PHY逻辑和原时序约束不变。
本次新增四路时钟活动计数、移相见过的8相位/选择位、DDR复位边沿及持续时间。
唯一控制功能变化是CPU只读APB诊断地址扩展至0x94；原主机禁止写APB等保护保持。

## 下一次监督板测

1. 用户在场、供电/USB已连接后重新枚举FT4232H物理B，不依赖COM号或历史USB地址。
2. program.py默认仅离线校验；临时JTAG下载前再次校验全部证据。不能写Flash。
3. 使用read_ddr_observer.py读取DCO1和OFS1。固定只读地址，遇错停止，宿主20秒超时，保存有效前缀。
4. 首次读取不发DDR载荷、不装载A、不启动CNN；内部厂商校准过程本身仍会访问DDR。
5. 根据校准阶段、时钟与复位证据定位问题。校准、受限CRC、A装载、画面和帧率必须逐级另行通过。

## 只读寄存器（APB基址0xF8100000）

70=DCO1；74/78/7C/80=core/tdqss/tac/twd分频边沿计数；84=150MHz/256计时器；
88=DDR复位下降/上升次数；8C=上次高/低持续ticks；90=相位/选择/状态；94=系统复位次数/当前DDR电平持续ticks。
每字同步读取，多字不是原子快照。主机使用前后时间界限给出频率范围；饱和时长只能作为下界。
这些是数字观察，不证明模拟电气质量或PHY最终正常。观察计数跨SW1保留、重新配置后清零。

模块四态、真实JTAG到APB短链、实际观察状态网表、实际控制启动/复位、原约束P&R/STA及CDC分别绑定此包。
实际观察状态网表短测的读mux使用冻结RTL，真实JTAG/APB链另测；不能据此宣称全板长仿真或实板PASS。
当前包只待监督取证，DDR校准和完整系统尚未通过。附A资产仅保留部署一致性。
''',encoding='utf-8')
    for name in ('read_first_fault','read_ddr_observer'):
        (out/(name+'.py')).write_text('import sys\nsys.path.insert(0,'+repr(str(HERE))+')\nfrom '+name+' import main\nif __name__=="__main__":main()\n')
    names=('README.md','ti60f225_oob.bit','coefficients-padded.bin','compute.bin','dma.bin','read_first_fault.py','read_ddr_observer.py')
    manifest=dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',board=str(board),batch=str(batch),selected_seed=selected['seed'],
        bitstream=str(out/'ti60f225_oob.bit'),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
        resources=resource,wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'],compute_clock_mhz=150,
        observation_delta=delta,observation_format='DCO1+OFS1',observer_sync=observer,functional_parent=str(PARENT),functional=parent['functional'],
        gray_max_skew_ns=g['max_skew_ns'],held_payload_max_ns=j['max_held_data_delay_ns'],unknown_new_crossings=0,
        full_video_hardware_preserved=True,required_evidence=evidence,user_supervision_required=True,hardware_access=False,
        automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,a_weight_load_released=False,full_system_accepted=False,
        bundle_files={n:sha(out/n) for n in names})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');verify(evidence)
    wrapper=(PARENT/'program.py').read_text()
    for old,new in ((sha(PARENT/'manifest.json'),sha(out/'manifest.json')),(parent['bitstream_sha256'],sha(bit))):
        assert wrapper.count(old)==1;wrapper=wrapper.replace(old,new)
    compile(wrapper,str(out/'program.py'),'exec');(out/'program.py').write_text(wrapper,encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],bundle=str(out),bit_sha256=sha(bit),wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'])))


if __name__=='__main__':main()
