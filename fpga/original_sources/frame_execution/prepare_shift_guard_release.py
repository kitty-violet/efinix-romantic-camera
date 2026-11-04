"""Offline full-system package; a PLL setup repair, not a board PASS."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_ddr_shift_guard_board import BASE,CAL,UNIT,audit_delta,proof
from audit_canonical_multiseed import audit
from snapshot_clock_cdc import inventory
from audit_snapshot_jtag_paths import audit as audit_jtag

PARENT=ROOT/'output/deployment/board-full-150-first-fault-20261010'

def main():
 p=argparse.ArgumentParser()
 for n in ('board','batch','bitroot','shiftpaths','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();batch=a.batch.resolve();bitroot=a.bitroot.resolve();shiftpaths=a.shiftpaths.resolve();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
 delta=audit_delta(board);proof();evidence={}
 def bind(path):
  evidence[str(path)]=sha(path);return json.loads(path.read_text())
 def verify(rows):
  for f,h in rows.items():assert sha(f)==h,('Evidence mismatch',f);evidence[f]=h
 parent=bind(PARENT/'manifest.json');assert parent['board']==str(BASE) and parent['compute_clock_mhz']==150
 verify(parent['required_evidence']);verify({str(PARENT/n):h for n,h in parent['bundle_files'].items()})
 terminal=bind(batch/'result.json');assert terminal['status']=='SHIFT_GUARD_PHYSICAL_PASS_PENDING_VENDOR_REPLAY_AND_BUNDLE'
 selected=audit(Path(terminal['selected']['canonical']));assert selected==terminal['selected'] and selected['current_timing_pass']
 physical=Path(selected['physical']);canonical=Path(selected['canonical']);gray=Path(terminal['gray'])
 if 'route_continuation_sha256' in terminal:
  continuation=bind(batch/'route_continuation.json')
  assert terminal['route_continuation_sha256']==sha(batch/'route_continuation.json')
  assert not continuation['rtl_changed'] and not continuation['map_changed']
  margin_change=continuation.get('optimization_margin_change')
  if margin_change:
   assert continuation['constraints_changed'] and continuation['acceptance_constraints_changed'] is False
   assert margin_change['before_extra_ns']==0.3 and margin_change['after_extra_ns']==0 and margin_change['canonical_acceptance_unchanged']
   unchanged_sdc=Path(margin_change['acceptance_sdc'])
   assert sha(unchanged_sdc)==margin_change['acceptance_sdc_sha256']==selected['identity']['canonical_sdc']
   assert sha(physical/'ti60f225_oob.sdc')==sha(unchanged_sdc)
   source_phys=Path(continuation['source'])
   extra='\n# Reserve 0.300ns during optimization; final acceptance is unchanged.\nset_clock_uncertainty -setup -to [get_clocks {*}] 0.300\n'
   assert (source_phys/'ti60f225_oob.sdc').read_text()==unchanged_sdc.read_text()+extra
   verify({str(source_phys/'ti60f225_oob.sdc'):margin_change['old_optimization_sdc_sha256'],str(unchanged_sdc):margin_change['acceptance_sdc_sha256']})
   driver=batch.parent/(batch.name+'-driver.py');driver_meta=bind(driver.with_suffix('.json'))
   verify({str(driver):driver_meta['generated_sha256'],driver_meta['original']:driver_meta['original_sha256'],driver_meta['generator']:driver_meta['generator_sha256']})
   assert sha(batch/'runner.py')==sha(driver)
  else:
   assert not continuation['constraints_changed']
  verify(continuation['source_gate']);verify(continuation['copied_placement_and_inputs'])
  assert continuation['command'][-1]=='--route'
  verify({str(physical/'route_only.cmd'):continuation['command_sha256']})
  source_phys=Path(continuation['source'])
  for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc','efx_pnr_settings.ini','fixed_dsp.placeloc'):
   if margin_change and n=='ti60f225_oob.sdc':continue
   assert sha(physical/n)==sha(source_phys/n)
  fi=bind(batch/'finisher_inputs.json');assert terminal['finisher_inputs_sha256']==sha(batch/'finisher_inputs.json')
  verify({str(batch/'finisher.py'):fi['runner_sha256'],str(batch/'route_result.json'):fi['route_result_sha256']})
  for f in (batch/'route_only.log',batch/'signoff_stages.json'):evidence[str(f)]=sha(f)
 mapped=bind(board/'result.json');meta=bind(board/'inputs.json')
 assert mapped['status']=='SHIFT_GUARD_FULL_BOARD_MAP_NOT_RELEASE'
 assert selected['identity']['map']==mapped['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 assert mapped['inputs_sha256']==sha(board/'inputs.json')
 verify({row['snapshot']:row['sha256'] for row in meta['inputs']})
 verify({str(board/'runner.py'):meta['runner_sha256']})
 batch_meta=bind(batch/'inputs.json');verify({str(batch/'runner.py'):batch_meta['runner_sha256']})
 c=bind(CAL/'result.json');cm=bind(CAL/'inputs.json');u=bind(UNIT/'result.json')
 assert c['status']=='GUARDED_SHIFT_VENDOR_CALIBRATION_RW_PASS' and u['status']=='PLL_SHIFT_GUARD_FOURSTATE_PASS'
 verify({str(CAL/n):h for n,h in cm['files'].items()});verify({str(CAL/'simulation.log'):c['log_sha256']})
 if 'post_audit' in c:
  verify({str(CAL/'failure.json'):c['post_audit']['original_failure_sha256'],str(HERE/'audit_ddr_shift_completed_log.py'):c['post_audit']['auditor_sha256']})
 verify({str(UNIT/n):h for n,h in u['inputs'].items()});verify({str(UNIT/'simulation.log'):u['log_sha256']})
 start=bind(batch/'startup/result.json');sm=bind(batch/'startup/inputs.json')
 assert start['status']=='FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS' and start['inputs_sha256']==sha(batch/'startup/inputs.json')
 assert sm['original_map_sha256']==mapped['map_sha256']
 verify({str(batch/'startup'/n):h for n,h in sm['files'].items()});verify(sm.get('original',{}))
 verify({str(batch/'startup/simulation.log'):start.get('log_sha256',start.get('simulation_sha256'))})
 import audit_control_state_implementation as impl
 impl.audit_delta=lambda b:audit_delta(b,False)
 current=impl.audit(board,physical);recorded=bind(batch/'implementation.json')
 for key in ('status','reset_ff','snapshot_sync','pins','delta'):assert recorded[key]==current[key]
 assert current['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS';verify(recorded['required_evidence'])
 from fixed_cnn_ram_locations import verify_placed
 assert bind(physical/'cnn_ram_actual_locations.json')==verify_placed(physical)
 ram=bind(physical/'cnn_ram_placement.json');reference=Path(ram['reference_canonical']);refphys=Path(ram['reference_physical'])
 assert audit(reference)['current_timing_pass']
 verify({str(reference/'result.json'):ram['reference_canonical_result_sha256'],str(refphys/'outflow/ti60f225_oob.out.placeloc'):ram['reference_placement_sha256'],str(refphys/'outflow/ti60f225_oob.map.v'):ram['reference_map_sha256'],str(physical/'fixed_dsp.placeloc'):ram['assignments_sha256'],str(HERE/'fixed_cnn_ram_locations.py'):ram['generator_sha256'],str(physical/'cnn_ram_generator.py'):ram['generator_sha256']})
 probe=physical.parent/(physical.name+'-ram-probe.py');pr=bind(probe.with_suffix('.json'))
 wrapper_path=Path(pr.get('wrapper_runner',str(HERE/'probe_shift_guard_physical.py'))).resolve()
 assert wrapper_path in {HERE/'probe_shift_guard_physical.py',HERE/'probe_feedback_shift_guard_physical.py',HERE/'probe_input_feedback_shift_guard_physical.py',HERE/'probe_local_shift_guard_physical.py',HERE/'probe_ddr_source_shift_guard_physical.py'}
 verify({str(probe):pr['generated_sha256'],str(physical/'physical_runner.py'):pr['generated_sha256'],pr['original_runner']:pr['original_sha256'],str(wrapper_path):pr['wrapper_sha256'],str(HERE/'shift_guard_clock_profile.py'):pr['clock_profile_sha256']})
 if wrapper_path.name in ('probe_feedback_shift_guard_physical.py','probe_input_feedback_shift_guard_physical.py','probe_local_shift_guard_physical.py','probe_ddr_source_shift_guard_physical.py'):
  from fixed_cnn_feedback_locations import verify_placed as verify_feedback
  assert bind(physical/'cnn_feedback_actual_locations.json')==verify_feedback(physical)
  feedback=bind(physical/'cnn_feedback_placement.json')
  assert feedback['primitive_count']==512 and feedback['truth_table_cases']==2048 and not feedback['clock_or_logic_changed']
  assert Path(feedback['reference'])==reference
  verify({str(HERE/'fixed_cnn_feedback_locations.py'):feedback['generator_sha256'],str(physical/'cnn_feedback_generator.py'):feedback['generator_sha256'],str(physical/'fixed_dsp.placeloc'):feedback['assignments_sha256']})
  for name in ('run_ddr_shift_guard_feedback_physical.py','probe_feedback_shift_guard_physical.py'):
   evidence[str(HERE/name)]=sha(HERE/name)
 if wrapper_path.name=='probe_input_feedback_shift_guard_physical.py':
  from fixed_cnn_input_locations import verify_placed as verify_input
  assert bind(physical/'cnn_input_actual_locations.json')==verify_input(physical)
  inputs_layout=bind(physical/'cnn_input_placement.json')
  assert inputs_layout['primitive_count']==768 and not inputs_layout['clock_or_logic_changed'] and Path(inputs_layout['reference'])==reference
  verify({str(HERE/'fixed_cnn_input_locations.py'):inputs_layout['generator_sha256'],str(physical/'cnn_input_generator.py'):inputs_layout['generator_sha256'],str(physical/'fixed_dsp.placeloc'):inputs_layout['assignments_sha256']})
  for name in ('run_ddr_shift_guard_input_physical.py','probe_input_feedback_shift_guard_physical.py'):
   evidence[str(HERE/name)]=sha(HERE/name)
 if wrapper_path.name in ('probe_local_shift_guard_physical.py','probe_ddr_source_shift_guard_physical.py'):
  local_generator=HERE/'fixed_shift_guard_local_repair.py'
  if wrapper_path.name=='probe_ddr_source_shift_guard_physical.py':
   from fixed_shift_guard_ddr_source_repair import verify_placed as verify_local
   local_generator=HERE/'fixed_shift_guard_ddr_source_repair.py'
   evidence[str(HERE/'fixed_shift_guard_local_repair.py')]=sha(HERE/'fixed_shift_guard_local_repair.py')
  else:
   from fixed_shift_guard_local_repair import verify_placed as verify_local
  assert bind(physical/'local_repair_actual_locations.json')==verify_local(physical)
  local=bind(physical/'local_repair_placement.json')
  assert local['primitive_count']==56835 and not local['rtl_or_map_changed']
  assert local['moved_cell']=='LUT__133530' and local['before']==[61,171,0] and local['after']==[56,167,0]
  verify({str(local_generator):local['generator_sha256'],str(physical/'local_repair_generator.py'):local['generator_sha256'],str(physical/'fixed_dsp.placeloc'):local['assignments_sha256'],str(Path(local['reference'])/'result.json'):local['reference_result_sha256'],local['destination_evidence']:local['destination_evidence_sha256']})
  if wrapper_path.name=='probe_ddr_source_shift_guard_physical.py':
   move=local['ddr_source_move'];assert move['protected_head_assigned'] is False
   assert move['source_before']==[64,2,2] and move['source_after']==[31,2,2] and move['driver_after']==[31,2,0]
   verify({str(Path(move['reference'])/'result.json'):move['reference_result_sha256'],move['placement']:move['placement_sha256']})
  verify(local['docs'])
 if (batch/'reused_startup.json').exists():
  reuse=bind(batch/'reused_startup.json');assert reuse['map_sha256']==mapped['map_sha256']
  verify({str(Path(reuse['root'])/'result.json'):reuse['result_sha256']})
 g=bind(gray/'result.json');gm=bind(gray/'inputs.json');gi=bind(gray/'clock_crossing_inventory.json')
 assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE' and Path(gm['base'])==physical
 assert g['inputs_sha256']==sha(gray/'inputs.json') and g['buses']==10 and g['endpoints']==70
 assert g['max_skew_ns']<g['limit_ns']==2.5
 assert gi==json.loads(json.dumps(inventory(gray))) and g['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
 j=bind(gray/'jtag_cdc_audit.json');assert j==audit_jtag(gray) and j['unknown_new_endpoints']==0
 verify(gm['original']);verify(gm['scripts']);verify(j['reports']);verify({row['path']:row['sha256'] for row in gi['reports']})
 # This is distinct from ordinary WNS: the original PLL false path is kept.
 sb=bind(shiftpaths/'bound.json');si=bind(shiftpaths/'inputs.json');sr=bind(shiftpaths/'result.json')
 assert sb['status']=='PLL_SHIFT_ROUTED_SETUP_HIGH_BOUND_PASS' and Path(si['base'])==canonical
 assert sb['source_inputs_sha256']==sha(board/'inputs.json') and sb['vendor_result_sha256']==sha(CAL/'result.json')
 assert sb['reports_inputs_sha256']==sr['inputs_sha256']==sha(shiftpaths/'inputs.json')
 verify(si['frozen']);verify(sb['reports'])
 gap_path=Path(sb['min_gap_result']);assert sha(gap_path)==sb['min_gap_result_sha256'];gap=bind(gap_path)
 assert gap['status']=='PLL_SHIFT_MINIMUM_GAP_PASS' and gap['low_ns']==sb['bounds']['logical_low_ns']
 verify({str(gap_path.parent/n):h for n,h in gap['inputs'].items()});verify({str(gap_path.parent/'simulation.log'):gap['log_sha256']})
 from audit_pll_shift_routed import bounds
 assert bounds(shiftpaths,sb['bounds']['logical_setup_ns'],sb['bounds']['logical_high_ns'],sb['bounds']['logical_low_ns'])==sb['bounds']
 assert sb['bounds']['setup_lower_ns']>=2.5 and sb['bounds']['common_control_high_lower_ns']>=10 and sb['bounds']['common_control_low_lower_ns']>=10
 bm=bind(bitroot/'generation_inputs.json');br=bind(bitroot/'result.json')
 assert br['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED' and br['inputs_sha256']==sha(bitroot/'generation_inputs.json')
 assert Path(bm['base'])==physical and Path(bm['canonical'])==canonical
 bit=bitroot/'outflow/ti60f225_oob.bit';item=next(v for k,v in br['artifacts'].items() if Path(k).name==bit.name)
 assert sha(bit)==item['sha256'] and bit.stat().st_size==item['bytes']
 verify({str(physical/'outflow'/n):h for n,h in bm['input_geometry'].items()})
 for name in ('work_pnr/ti60f225_oob.lbf','outflow/ti60f225_oob.lpf','outflow/ti60f225_oob.raminfo.pb'):
  assert sha(physical/name)==sha(bitroot/name);evidence[str(physical/name)]=sha(physical/name);evidence[str(bitroot/name)]=sha(bitroot/name)
 cr=bind(canonical/'result.json');cm=bind(canonical/'inputs.json');verify(cm['original']);verify({str(canonical/'outflow'/n):h for n,h in cr['reports'].items()})
 for f in (physical/'physical_inputs.json',batch/'inputs.json',batch/'stages.json',board/'delta_audit.json',board/'control_state_map_audit.json',board/'snapshot_sync_audit.json',ROOT/'tmp/board-test/program-bringup-status.py',bitroot/'outflow/ti60f225_oob.pgm.out'):
  evidence[str(f)]=sha(f)
 for n in ('prepare_ddr_shift_guard_board.py','run_ddr_shift_guard_physical.py','validate_shift_guard_mapped.py','survey_pll_shift_routed.py','audit_pll_shift_routed.py','prepare_shift_guard_release.py','shift_guard_clock_profile.py','probe_shift_guard_physical.py'):
  evidence[str(HERE/n)]=sha(HERE/n)
 tests=batch/'first_fault_host_tests.log';assert not tests.exists()
 test=subprocess.run([sys.executable,'-m','unittest','test_first_fault_host','-v'],cwd=HERE,capture_output=True,text=True,timeout=30)
 tests.write_text(test.stdout+test.stderr);assert test.returncode==0 and 'Ran 9 tests' in tests.read_text() and tests.read_text().rstrip().endswith('OK')
 evidence[str(tests)]=sha(tests)
 out.mkdir(parents=True);shutil.copy2(bit,out/bit.name)
 for n in ('coefficients-padded.bin','compute.bin','dma.bin','read_first_fault.py'):shutil.copy2(PARENT/n,out/n)
 (out/'README.md').write_text('''# 完整150MHz系统：DDR移相接口建立时间修正

保留摄像头、DDR、已接受的B1-A与HDMI链，时钟/电压/引脚不变。
唯一功能差异为公开PLL控制接口：捕获移相码，等待一拍后使能。
正常厂商模型校准及读写、控制器四态短测、新网表启动/复位、新布局原约束时序、CDC与实际控制输出延迟分别核验。
这些离线结果不代表已解决实板校准。下载前必须等待用户在旁监督；临时JTAG，不写Flash。
首次只读启动与DDR校准状态；不发DDR数据请求、不装载A、不运行CNN。
确认实板done/pass正常后，另行决定受限DDR CRC与A装载测试。
program.py默认仅离线校验；USB按物理接口枚举，不固定COM口。
''',encoding='utf-8')
 manifest=dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',board=str(board),batch=str(batch),selected_seed=selected['seed'],bitstream=str(out/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
  resources=selected['resources'],wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'],compute_clock_mhz=150,repair_delta=delta,observation_format='OFS1',functional_parent=str(PARENT),functional=parent['functional'],
  vendor_calibration=c['status'],pll_setup_lower_ns=sb['bounds']['setup_lower_ns'],pll_high_lower_ns=sb['bounds']['common_control_high_lower_ns'],pll_low_lower_ns=sb['bounds']['common_control_low_lower_ns'],gray_max_skew_ns=g['max_skew_ns'],held_payload_max_ns=j['max_held_data_delay_ns'],unknown_new_crossings=0,
  full_video_hardware_preserved=True,required_evidence=evidence,user_supervision_required=True,hardware_access=False,automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,a_weight_load_released=False,full_system_accepted=False,
  bundle_files={n:sha(out/n) for n in ('README.md','ti60f225_oob.bit','coefficients-padded.bin','compute.bin','dma.bin','read_first_fault.py')})
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');verify(evidence)
 wrapper=(PARENT/'program.py').read_text()
 for old,new in ((sha(PARENT/'manifest.json'),sha(out/'manifest.json')),(parent['bitstream_sha256'],sha(bit))):
  assert wrapper.count(old)==1;wrapper=wrapper.replace(old,new)
 compile(wrapper,str(out/'program.py'),'exec');(out/'program.py').write_text(wrapper,encoding='utf-8')
 print(json.dumps(dict(status=manifest['status'],bundle=str(out),wns_ns=selected['wns_ns'],whs_ns=selected['whs_ns'],bit_sha256=sha(bit))))

if __name__=='__main__':main()
