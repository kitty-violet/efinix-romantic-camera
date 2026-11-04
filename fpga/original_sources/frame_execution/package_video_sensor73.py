"""Release only the actual sensor73 route and affected-interface evidence."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from video_service_board import bound
from audit_canonical_multiseed import audit
from place_video_sensor73 import audit_delta
from package_video_service import local_imports

def main():
 p=argparse.ArgumentParser()
 for n in ('board','pin-test','route','canonical','gray','pll','bits','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();pins=a.pin_test.resolve();canonical=a.canonical.resolve();gray=a.gray.resolve();pll=a.pll.resolve();bits=a.bits.resolve();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
 parent=ROOT/'output/deployment/board-full-150-video-service71-exposure-20261017'
 assert sha(parent/'manifest.json')=='eca4bda3858e5bc4775a1e3577e27c1bb92af3c021bf5a3f70fc8b6b7794fcae'
 pm=bound(parent/'manifest.json');evidence=dict(pm['required_evidence'])
 def bindfile(f,h=None):
  f=Path(f).resolve();actual=sha(f)
  if h is not None:assert actual==h,('evidence',str(f))
  if str(f) in evidence:assert evidence[str(f)]==actual,('historical_changed',str(f))
  evidence[str(f)]=actual
  return json.loads(f.read_text(encoding='utf8')) if f.suffix=='.json' else None
 def embedded(obj):
  if isinstance(obj,dict):
   if isinstance(obj.get('snapshot'),str) and isinstance(obj.get('sha256'),str):bindfile(obj['snapshot'],obj['sha256'])
   for k,v in obj.items():
    if isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) and Path(k).is_absolute():bindfile(k,v)
    elif isinstance(v,(dict,list)):embedded(v)
  elif isinstance(obj,list):
   for v in obj:embedded(v)
 bindfile(parent/'manifest.json');bindfile(parent/'program.py')
 for n,h in pm['bundle_files'].items():bindfile(parent/n,h)
 proof=audit_delta(board);embedded(proof);bindfile(board/'sensor73-map-audit.json')
 wr=bindfile(board/'sensor73-warning-review.json');embedded(wr)
 assert wr['status']=='SENSOR73_WARNINGS_REVIEWED_NO_NEW_UNCLASSIFIED'
 meta=bindfile(board/'inputs.json');embedded(meta);native=Path(meta['native'])
 delta=bindfile(native/'inputs.json');embedded(delta)
 assert delta['gain_register']==52 and delta['exposure_half_lines']==1120
 assert delta['source_sha256']==sha(native/'native_source.v')==sha(board/'native_source.v')
 assert delta['all_unlisted_source_byte_equal'] and delta['new_luts']==8
 assert delta['script_sha256']==sha(HERE/'prepare_video_sensor73_original_input.py')
 assert delta['base_generator_sha256']==sha(HERE/'prepare_video_sensor_rom_verify.py')
 sequence=bindfile(native/'inputs.json')['verification_entries']
 assert sequence[-1]==dict(address=0x3e09,expected=52,mask=255)
 bindfile(native/'expected.hex')
 words=[int(x,16) for x in (native/'expected.hex').read_text().splitlines()]
 assert len(words)==164 and all(w>>24==0x60 for w in words)
 registers={w>>8&65535:w&255 for w in words}
 assert [registers[r] for r in (0x3e00,0x3e01,0x3e02,0x3e08,0x3e09)]==[0,70,0,128,52]
 pr=bindfile(pins/'result.json');assert pr['status'] in ('SENSOR73_MAPPED_I2C_PINS_PASS','SENSOR73_IDENTICAL_MAPPED_I2C_PINS_EVIDENCE_PASS') and [r['mode'] for r in pr['cases']]==[0,10,4,6,8]
 pi=bindfile(pins/'inputs.json',pr['inputs_sha256']);embedded(pi)
 prepared=Path(pi['prepared']);si=bindfile(prepared/'inputs.json',pi['prepared_sha256']);embedded(si)
 assert si['map_sha256']==proof['map_sha256']
 test_origin=pins
 if pr['status']=='SENSOR73_IDENTICAL_MAPPED_I2C_PINS_EVIDENCE_PASS':
  assert pr['reexecuted'] is False
  bindfile(HERE/'reuse_video_sensor73_identical_pins.py',pi['runner_sha256'])
  test_origin=Path(pi['reference_test']);opr=bindfile(test_origin/'result.json');opi=bindfile(test_origin/'inputs.json',opr['inputs_sha256']);embedded(opi)
  assert opr['status']=='SENSOR73_MAPPED_I2C_PINS_PASS' and opr['cases']==pr['cases']
  original=Path(opi['prepared']);bindfile(original/'inputs.json',opi['prepared_sha256'])
  assert all(sha(Path(f))==h==sha(original/Path(f).name) for f,h in si['files'].items())
  bindfile(HERE/'test_video_sensor73.py',opi['runner_sha256']);bindfile(HERE/'test_video_sensor72.py',opi['base_runner_sha256'])
 else:bindfile(HERE/'test_video_sensor73.py',pi['runner_sha256'])
 for r in pr['cases']:bindfile(test_origin/f"mode{r['mode']}.log",r['log_sha256']);assert r['exit_code']==0
 bindfile(test_origin/'compile.log');bindfile(test_origin/'run.do')
 cdc=bindfile(board/'sensor73-cdc-delta.json');embedded(cdc)
 assert cdc['status']=='SENSOR73_CDC_DELTA_CLASSIFIED_NOT_ROUTED_PASS' and cdc['new_cross_domain_cones']==0
 bindfile(board/'service-cdc-inventory-expanded.json',cdc['current_inventory_sha256'])
 t=audit(canonical);assert t['current_timing_pass'] and t['identity']['map']==proof['map_sha256']
 physical=Path(t['physical']);embedded(bindfile(physical/'physical_inputs.json'))
 rr=bindfile(a.route/'result.json');ri=bindfile(a.route/'inputs.json',rr['inputs_sha256']);embedded(ri)
 assert rr['status']=='VIDEO_SERVICE_LEGAL_ROUTE_PASS_NOT_RELEASE' and Path(ri['physical'])==physical
 assert rr['trace_sha256']==sha(physical/'outflow/ti60f225_oob.troutingtraces')
 bindfile(a.route/'route.log');bindfile(a.route/'stages.json')
 if 'routing_only_override' in ri:
  override=ri['routing_only_override'];assert override['value']=='TIMING_1' and override['placement_recipe']=='CONGESTION_2' and override['cli_override_only']
  assert not override['original_sdc_changed'] and not override['source_or_placement_changed']
  bindfile(HERE/'route_video_service.py',ri['base_route_runner_sha256'])
  runner=ROOT/'tmp/board-test/video-service-review1/route-sensor73-timing1.py'
  bindfile(runner,ri['runner_sha256']);bindfile(a.route/'runner.py',ri['runner_sha256'])
  bindfile(runner.with_name('route-reused-timing1.py'))
  reuse=ri['placement_reuse'];origin=Path(reuse['source'])
  bindfile(origin/'physical_inputs.json',reuse['physical_inputs_sha256'])
  bindfile(origin/'before_route/result.json',reuse['source_placement_result_sha256'])
  for n,h in reuse['geometry'].items():
   bindfile(origin/'outflow'/n,h);bindfile(physical/'outflow'/n,h)
  assert reuse['copied_not_reexecuted']
 else:bindfile(HERE/'route_video_service.py',ri['runner_sha256'])
 rb=bindfile(physical/'original-gray-endpoints.json');embedded(rb)
 assert rb['status']=='EXACT_ORIGINAL_GRAY_ENDPOINT_NAMES_PASS' and rb['buses']==10 and rb['destination_bits']==70 and rb['missing']==0
 assert rb['renamed_endpoints']==rb['changed_constraints']==0
 assert rb['original_sdc_sha256']==sha(physical/'ti60f225_oob.sdc')
 per=bindfile(physical/'original-periphery-output-check.json');embedded(per)
 assert per['status']=='ORIGINAL_PERIPHERY_INPUT_AND_GENERATED_PINS_MATCH_EXCEPT_DATE'
 reference=ROOT/'tmp/frame-execution/video-service-place50-input-local'
 old_csv=(reference/'outflow/ti60f225_oob.interface.csv').read_text().splitlines();new_csv=(physical/'outflow/ti60f225_oob.interface.csv').read_text().splitlines()
 assert len(old_csv)==len(new_csv)
 different=[(x,y) for x,y in zip(old_csv,new_csv) if x!=y]
 assert len(different)==1 and all(x.startswith('# Date: ') for x in different[0])
 assert sha(reference/'ti60f225_oob.peri.xml')==sha(physical/'ti60f225_oob.peri.xml')
 embedded(bindfile(physical/'route-review.json'));bindfile(physical/'result.json')
 for f in ('placement_review_inputs.json','service_clock_profile.json','cnn_ram_placement.json','hard-anchor-check.json'):embedded(bindfile(physical/f))
 embedded(bindfile(canonical/'inputs.json'));bindfile(canonical/'result.json')
 for f in (canonical/'outflow').glob('canonical_*.rpt'):bindfile(f)
 gr=bindfile(gray/'result.json');assert gr['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
 bindfile(gray/'inputs.json',gr['inputs_sha256'])
 sr=bindfile(gray/'service_cdc.json',gr['service_cdc_sha256']);assert sr['status']=='VIDEO_SERVICE_ROUTED_CDC_SETTLING_PASS_NOT_RELEASE' and sr['map_sha256']==proof['map_sha256']
 for f in gray.glob('*.json'):embedded(bindfile(f))
 for f in (gray/'outflow').glob('*.rpt'):bindfile(f)
 pl=bindfile(pll/'bound.json');embedded(pl);assert pl['status']=='PLL_SHIFT_ROUTED_SETUP_HIGH_BOUND_PASS'
 for k,v in [('setup_lower_ns',2.5),('common_control_high_lower_ns',10),('common_control_low_lower_ns',10)]:assert pl['bounds'][k]>=v
 embedded(bindfile(pll/'inputs.json'));bindfile(pll/'result.json')
 for suffix in ('.map.v','.place','.troutingtraces','.vdb'):
  name='outflow/ti60f225_oob'+suffix
  assert sha(gray/name)==sha(pll/name)==sha(canonical/name)==sha(physical/name)==sha(bits/name)
  bindfile(bits/name)
 br=bindfile(bits/'result.json');assert br['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED'
 gi=bindfile(bits/'generation_inputs.json',br['inputs_sha256']);assert Path(gi['canonical'])==canonical
 bit=bits/'outflow/ti60f225_oob.bit'
 from deployment_artifact_paths import checked_artifact
 art=checked_artifact(br['artifacts'],bits,bit);assert art['sha256']==sha(bit) and art['bytes']==bit.stat().st_size
 bindfile(bit)
 plan_root=ROOT/'tmp/frame-execution/video-sensor72-native2'
 plan=bindfile(plan_root/'comparison-isp.json');bindfile(plan_root/'comparison-isp.bin',plan['sha256'])
 assert plan['gamma']==2.3 and plan['black']==8 and plan['gains']==[1.1431,.73435,1.4135] and not plan['hardware_applied']
 runtime=dict(pm['runtime_sha256'])
 for name in ('video_service_input_isp_prepared.py','video_service_restore_preview.py','video_service_stage_preview_coefficients.py'):local_imports(HERE/name,runtime)
 for receipt,helper,tests in [('isp-prepared-tests1.json','video_service_input_isp_prepared.py',14),('restore-preview-tests1.json','video_service_restore_preview.py',9),('preview-coeff-host-tests1.json','video_service_stage_preview_coefficients.py',6)]:
  hr=bindfile(ROOT/'tmp/board-test/video-service69-board1'/receipt)
  assert hr['status']=='PASS' and hr['tests']==tests
  bindfile(HERE/helper,hr['helper_sha256']);bindfile(HERE/('test_'+helper),hr['test_sha256'])
  if 'prepare_sha256' in hr:bindfile(HERE/'video_service_reprepare_preview.py',hr['prepare_sha256'])
 for f in (Path(__file__),HERE/'video_sensor73_physical.py',HERE/'video_sensor72_sta.py',HERE/'video_sensor73_pll.py',HERE/'prepare_video_sensor73_original_input.py',HERE/'prepare_video_sensor_rom_verify.py',HERE/'audit_video_sensor73_source_scope.py',HERE/'audit_video_sensor73_original_cdc.py'):bindfile(f)
 out.mkdir();shutil.copy2(bit,out/bit.name)
 for n in ('coefficients-padded.bin','compute.bin','dma.bin'):shutil.copy2(parent/n,out/n)
 for n in ('comparison-isp.bin','comparison-isp.json'):shutil.copy2(plan_root/n,out/n)
 shutil.copy2(native/'expected.hex',out/'sensor_write_sequence.hex')
 (out/'sensor_parameters.json').write_text(json.dumps(dict(registers=registers,verification_entries=sequence,analog_gain=2.503,exposure_half_lines=1120),indent=2))
 (out/'README.md').write_text(f'''# 73：传感器增益与曝光同步调整

模拟增益3e08/3e09=80/34（官方2.503倍，用户要求2.5倍）；曝光3e00/01/02=00/46/00（1120半行，比71的1494再减少约25%）。增益启动读回期望从同一ROM的校验区取得，当前为34，旧2a会被拒绝。后续同类参数变化可使用ROM更新流程，仍需同源验证。

原综合入口除摄像头差分外逐字保持；当前实际新映射/新路由包含全部修改。未声称完整网表形式等价证明通过；受影响实际摄像头原语五场景已验证。没有沿用71的旧ROM覆盖来冒称新源。固定B1-A、150MHz、引脚和原时序要求保持。原约束WNS {t['wns_ns']:+.3f}ns，WHS {t['whs_ns']:+.3f}ns；XLR {t['resources']['XLRs']['used']}/60800。

输入颜色表目标Gamma2.3、黑位8、RGB增益1.1431/0.73435/1.4135、R/B交换开启，CRC {plan['crc32']}。comparison-isp.bin是需要在停止排空状态上传并完整读回的运行表，不能把它误认为位流默认值已经改变。

下载前确认用户仍在板旁监督；仅临时JTAG，不写Flash。先读取身份、摄像头配置及DDR状态。新配置会清除DDR，必须重新上传A的20480字节系数并全读回，再LOAD到INPUT_PREVIEW；上传上述颜色表768字节并验证CRC，随后明确关彩条、清inspection、重新LOAD、LIVE。本轮先预览验色，不把预览当作CNN性能结果。

所有数值和视觉效果仍须实板确认。降低曝光并增加增益不保证消除白边，也可能增加暗部噪声。A权重和compute/DMA保持原包；不能以HDMI30Hz冒称新AI30fps。15fps、丢帧、长期稳定及其余系统职责仍未完成验收。
''',encoding='utf8')
 m=dict(pm)
 for k in ('native_parameter_overlay','effective_native_sha256','rom_overlay','rom_overlay_sha256'):m.pop(k,None)
 m.update(native=str(native),board=str(board),map_sha256=proof['map_sha256'],bitstream=str(out/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
  canonical=str(canonical),wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],resources=t['resources'],selected_seed=t['seed'],required_evidence=evidence,runtime_sha256=runtime,
  implementation_basis='fresh official map and routed sensor73 implementation; runtime ISP upload separate',
  base_manifest=str(parent/'manifest.json'),base_manifest_sha256=sha(parent/'manifest.json'),
  current_short_proofs=dict(retained_circuit=str(board/'sensor73-map-audit.json'),affected_pin_level=str(pins/'result.json'),cdc_delta=str(board/'sensor73-cdc-delta.json')),
  sensor_parameters=dict(analog_gain=2.503,exposure_half_lines=1120,gain_registers=[128,52],exposure_registers=[0,70,0]),
  runtime_isp_target=plan,inherited_board_pass=False,full_system_accepted=False,hardware_access=False,
  pending=['supervised73 startup/readback/visual comparison','actual exposure/highlight/shadow noise','same-epoch AI fps/drop accounting','RAW and long-run stability','remaining baseline system duties'],
  bundle_files={f.name:sha(f) for f in out.iterdir() if f.is_file()})
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
 template=(parent/'program.py').read_text().replace(sha(parent/'manifest.json'),sha(out/'manifest.json')).replace(pm['bitstream_sha256'],sha(bit))
 (out/'program.py').write_text(template,encoding='utf8');bound(out/'manifest.json')
 print(json.dumps(dict(status=m['status'],bundle=str(out),manifest_sha256=sha(out/'manifest.json'),bitstream_sha256=sha(bit),hardware_access=False)),flush=True)

if __name__=='__main__':main()
