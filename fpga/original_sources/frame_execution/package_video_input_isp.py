"""Input ISP release: actual source/map proof plus current physical evidence.

Retained67 tests are reused only through the complete actual-map/source delta.
This candidate must supply its own STA, routed CDC, PLL and generated bitstream.
No programming or USB calls; the resulting package requires supervised use.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_canonical_multiseed import audit as timing_audit
from place_video_input_isp import audit_delta
from fixed_jtag_loader import validate_payload
from package_video_service import local_imports
from deployment_artifact_paths import checked_artifact


def main():
 p=argparse.ArgumentParser()
 for n in ('board','canonical','gray','pll','bitstream-root','output'):
  p.add_argument('--'+n,type=Path,required=n in ('board','output'))
 p.add_argument('--proof-only',action='store_true')
 a=p.parse_args();board=a.board.resolve();out=a.output.resolve()
 assert not out.exists()
 if a.proof_only:
  assert out.is_relative_to(ROOT/'tmp') and out.suffix=='.json'
 else:
  assert all((a.canonical,a.gray,a.pll,a.bitstream_root))
  canonical=a.canonical.resolve();gray=a.gray.resolve();pll=a.pll.resolve();bits=a.bitstream_root.resolve()
  assert out.is_relative_to(ROOT/'output/deployment')
 parent=ROOT/'output/deployment/board-full-150-video-service67-color-20261017'
 evidence={}
 def bind(path,expected=None):
  f=Path(path).resolve();h=evidence.get(str(f)) or sha(f)
  if expected is not None:assert h==expected,('changed_evidence',str(f))
  evidence[str(f)]=h
  return json.loads(f.read_text(encoding='utf-8')) if f.suffix=='.json' else None
 def file_map(d,base=None):
  if isinstance(d,dict):
   if isinstance(d.get('snapshot'),str) and isinstance(d.get('sha256'),str):bind(d['snapshot'],d['sha256'])
   for k,v in d.items():
    if isinstance(v,str) and re.fullmatch('[a-f0-9]{64}',v):
     f=Path(k)
     if f.is_absolute():bind(f,v)
     elif base is not None and (base/f).is_file():bind(base/f,v)
    elif isinstance(v,(dict,list)):file_map(v,base)
  elif isinstance(d,list):
   for v in d:file_map(v,base)
 pm=bind(parent/'manifest.json','b42c3b6044e2f947851c882f30011105563c0da9465600e15d3aea47b44390a3')
 assert pm['status']=='VIDEO_SERVICE_OFFLINE_RELEASED_FOR_SUPERVISED_TEST'
 # Only these host files have been superseded by schema6. All historical
 # hardware evidence must still match exactly. Current host regression binds
 # the replacement implementations; no historical host PASS is reused for them.
 host=ROOT/'tmp/frame-execution/video-input-host1'
 hr=bind(host/'result.json');assert hr['status']=='PASS' and hr['tests']==119
 file_map(hr);bind(host/'tests.log',hr['log_sha256'])
 extra_host=ROOT/'tmp/frame-execution/video-input-frame-budget1'
 extra=bind(extra_host/'result.json');assert extra['status']=='PASS' and extra['tests']==6
 file_map(extra);bind(extra_host/'tests.log',extra['log_sha256'])
 replaced={}
 host_names={'video_service_board.py','video_service_observation.py','read_video_service_pages.py','test_video_service_observation.py'}
 for name,h in pm['required_evidence'].items():
  f=Path(name)
  if f.parent==HERE and f.name in host_names:
   assert hr['files'][str(f)]==sha(f)
   replaced[str(f)]=dict(historical_sha256=h,current_sha256=sha(f),current_proof=str(host/'result.json'))
  else:bind(f,h)
 assert {Path(f).name for f in replaced}==host_names
 for n,h in pm['bundle_files'].items():bind(parent/n,h)
 audit_delta(board)
 bm=bind(board/'inputs.json');br=bind(board/'result.json');file_map(bm)
 native=Path(bm['native']);assert Path(bm['parent'])==Path(pm['board'])
 mapped=board/'outflow/ti60f225_oob.map.v';bind(mapped,br['map_sha256'])
 delta=bind(board/'input-isp-map-audit.json');file_map(delta)
 source_delta=bind(native/'input-isp-edit.json');file_map(source_delta)
 assert delta['source_sha256']==source_delta['candidate_sha256']==sha(native/'native_source.v')
 assert delta['source']['changed_native_cells']==179 and delta['retained']==54240
 assert delta['source']['protected_input_byte_identical']
 inventory=bind(board/'service-cdc-inventory-expanded.json');file_map(inventory)
 assert inventory['map_sha256']==sha(mapped) and not inventory['unclassified']
 cdc_delta=bind(board/'input-isp-cdc-delta.json');file_map(cdc_delta)
 assert cdc_delta['status']=='INPUT_ISP_CDC_CLASSIFICATION_PASS_NOT_ROUTED_RELEASE'
 assert cdc_delta['map_sha256']==sha(mapped)
 for name in ('schema6-pages.json','schema6-counter.json'):
  proof=bind(board/name);assert 'PASS' in proof['status'] and proof['map_sha256']==sha(mapped)
  file_map(proof)
 tests={}
 for name in ('video-input-color4','video-input-lut2','video-input-mapped1','video-input-native2','video-input-continuous-isp2','video-input-recovery-wide3'):
  unit=ROOT/'tmp/frame-execution'/name
  numeric=bind(unit/'result.json');assert numeric['status']=='PASS'
  file_map(bind(unit/'inputs.json',numeric['inputs_sha256']),unit);file_map(numeric,unit)
  for f in unit.glob('*.log'):bind(f)
  tests[name]=dict(result_sha256=sha(unit/'result.json'),inputs_sha256=sha(unit/'inputs.json'))
 if a.proof_only:
  out.write_text(json.dumps(dict(status='INPUT_ISP_PACKAGE_LOGICAL_GATES_PASS_NOT_RELEASE',map_sha256=sha(mapped),
   required_evidence=evidence,superseded_historical_host=replaced,tests=tests,hardware_access=False),indent=2))
  print('INPUT_ISP_PACKAGE_LOGICAL_GATES_PASS_NOT_RELEASE',len(evidence));return
 t=timing_audit(canonical);assert t['current_timing_pass'] and t['identity']['map']==sha(mapped)
 bind(canonical/'result.json');file_map(bind(canonical/'inputs.json'))
 for f in (canonical/'outflow').glob('canonical_*.rpt'):bind(f)
 physical=Path(t['physical']);file_map(bind(physical/'route-review.json'))
 for f in ('placement_review_inputs.json','control-network-survey.json','physical_inputs.json','service_clock_profile.json'):
  file_map(bind(physical/f))
 gr=bind(gray/'result.json');assert gr['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
 bind(gray/'inputs.json',gr['inputs_sha256'])
 for field,name in [('inventory_sha256','clock_crossing_inventory.json'),('jtag_audit_sha256','jtag_cdc_audit.json'),('skew_audit_sha256','gray_skew_audit.json'),('service_cdc_sha256','service_cdc.json')]:bind(gray/name,gr[field])
 service=bind(gray/'service_cdc.json')
 assert service['status']=='VIDEO_SERVICE_ROUTED_CDC_SETTLING_PASS_NOT_RELEASE' and service['map_sha256']==sha(mapped)
 for f in gray.glob('*.json'):file_map(bind(f),gray)
 for f in (gray/'outflow').glob('*.rpt'):bind(f)
 pr=bind(pll/'bound.json');assert pr['status']=='PLL_SHIFT_ROUTED_SETUP_HIGH_BOUND_PASS'
 for k,v in [('setup_lower_ns',2.5),('common_control_high_lower_ns',10),('common_control_low_lower_ns',10)]:assert pr['bounds'][k]>=v
 file_map(pr);file_map(bind(pll/'inputs.json'));bind(pll/'result.json')
 for suffix in ('.map.v','.place','.vdb','.troutingtraces'):
  name='ti60f225_oob'+suffix
  assert sha(gray/'outflow'/name)==sha(pll/'outflow'/name)==sha(canonical/'outflow'/name)
 generated=bind(bits/'result.json');assert generated['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED'
 gi=bind(bits/'generation_inputs.json',generated['inputs_sha256']);assert Path(gi['canonical'])==canonical
 for name,h in gi['input_geometry'].items():assert sha(bits/'outflow'/name)==sha(canonical/'outflow'/name)==h
 bit=bits/'outflow/ti60f225_oob.bit';art=checked_artifact(generated['artifacts'],bits,bit)
 assert art['sha256']==sha(bit) and art['bytes']==bit.stat().st_size
 integration=Path(pm['a_integration']);payload=validate_payload(integration)
 for name,data in payload.items():assert data==(parent/(name+'.bin')).read_bytes()
 runtime={}
 for name in ('video_service_board.py','video_service_input_isp.py','video_service_prepare_preview.py','video_service_timed_read.py','video_service_frame_accounting.py','video_service_drained_receipt.py','video_service_frame_budget.py'):
  local_imports(HERE/name,runtime)
 for f in (Path(__file__),HERE/'deployment_artifact_paths.py',HERE/'video_input_pll.py',ROOT/'tmp/board-test/program-bringup-status.py'):bind(f)
 out.mkdir();shutil.copy2(bit,out/bit.name)
 for name,data in payload.items():(out/(name+'.bin')).write_bytes(data)
 (out/'README.md').write_text(f'''# B1-A 150MHz：RAW排列与输入颜色校准监督板测包

- 固定B1拓扑、A系数和计算/DMA程序；修正Debayer前RAW像素对顺序，并在输入端增加三通道256项查表。
- 默认Gamma2.2、R/B方向与成功Demo一致；曝光保持00/ba/d0。默认亮度和白平衡尚须真实场景验收。
- J4摄像头、HDMI TX 1080p30。原约束WNS {t['wns_ns']:+.3f}ns、WHS {t['whs_ns']:+.3f}ns。
- XLR {t['resources']['XLRs']['used']}/60800；容量内不等于达到95%余量目标。
- 位流SHA：{sha(bit)}。默认program.py只离线校验；下载前须用户现场监督，不写Flash。
- 权重限制guarded-A-zero-v1。临时下载后先复核启动/DDR/摄像头，再用prepare_preview核验resident系数并选择INPUT_PREVIEW，然后发LIVE开始原图预览；该阶段绕过CNN，不代表DDR程序已验证。
- 用户确认颜色后STOP排空并保存校准CRC；预览保留前台帧，不能直接切LIVE_A。按SW1清空旧槽位（合法输入查表保留），读取CRC/模式0和DDR恢复状态，再上传A系数及计算/DMA程序并完整读回，随后连续启动。不得把prepare_preview成功当成CNN装载放行。
- SW0（第二颗白键）连续启停；SW1（第三颗）逻辑复位。临时位流及DDR内容掉电丢失。
- 输入查表只能STOP排空后配置。video_service_input_isp.py上传768字节并按256索引完整读回三通道，CRC一致才接受；不自动启动CNN。
- 新schema6为112请求/轮，保留原状态并新增ISP页；必须使用本manifest绑定的宿主程序。
- 67实板新AI结果约10fps、HDMI30Hz；不是本版实测。15fps未达，未以重复显示充当AI新帧。
- RAW溢出取消及偶发预览fault7尚未证明完全消除，下一板需同epoch、停止排空前后统计；画面正常不能忽略RAW告警。
- 本包仅放行监督测试。真实颜色、丢帧、连续稳定性、同帧数值、手势/恶灵效果/FPS叠字和目标性能仍待验证。
''',encoding='utf-8')
 keep=('diagnostic_schema','service_diagnostic_positions','camera_diagnostic_slots','frequency_mhz','flash_allowed','a_integration','coefficient_policy','observation_counter_contract')
 m={k:pm[k] for k in keep}
 m.update(status='VIDEO_SERVICE_OFFLINE_RELEASED_FOR_SUPERVISED_TEST',native=str(native),board=str(board),map_sha256=sha(mapped),
  bitstream=str(out/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,resources=t['resources'],
  wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],selected_seed=t['seed'],canonical=str(canonical),
  required_evidence=evidence,runtime_sha256=runtime,
  current_short_proofs=dict(retained67=dict(status='REUSED_THROUGH_EXACT_SOURCE_AND_MAP_DELTA',parent_manifest=str(parent/'manifest.json'),parent_manifest_sha256=sha(parent/'manifest.json')),
    input_source_sha256=sha(native/'input-isp-edit.json'),input_map_sha256=sha(board/'input-isp-map-audit.json'),tests=tests,
    current_host_regression_sha256=sha(host/'result.json'),superseded_historical_host=replaced),
  inherited_numerical_evidence=str(parent/'manifest.json'),inherited_board_pass=False,
  user_supervision_required=True,automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,a_weight_load_released=False,
  full_system_accepted=False,hardware_access=False,bundle_files={f.name:sha(f) for f in out.iterdir() if f.is_file()},
  pending=['supervised69 cold start/reset and corrected RAW/color comparison','real exposure/sensor cadence and brightness','RAW loss and timed AI/display rate','same-frame board numerical evidence','gesture/evil effect/FPS overlay and target performance'])
 m['diagnostic_schema']=6
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
 template=(parent/'program.py').read_text()
 assert template.count(pm['bitstream_sha256']) and template.count(sha(parent/'manifest.json'))
 template=template.replace(pm['bitstream_sha256'],sha(bit)).replace(sha(parent/'manifest.json'),sha(out/'manifest.json'))
 (out/'program.py').write_text(template,encoding='utf-8')
 from video_service_board import bound
 bound(out/'manifest.json')
 print(json.dumps(dict(status=m['status'],bundle=str(out),resources=t['resources'],wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],hardware_access=False)))


if __name__=='__main__':main()
