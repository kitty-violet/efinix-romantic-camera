"""Package only a separately verified84 implementation for supervised testing."""
import argparse,json,shutil,importlib.util
from pathlib import Path
from run_units import ROOT,sha
from video_service_board import bound
from audit_canonical_multiseed import audit as timing_audit
from video_control86_physical import audit,BOARD
from video_bandwidth83_physical import audit as audit83,BOARD as BOARD83

def main():
 p=argparse.ArgumentParser()
 for n in ('route','canonical','gray','pll','bits','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();route=a.route.resolve();canonical=a.canonical.resolve();gray=a.gray.resolve();pll=a.pll.resolve();bits=a.bits.resolve();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
 parent=ROOT/'output/deployment/board-full-150-video-gain82-20261018';assert sha(parent/'manifest.json')=='1466619da9c2d8560223e8d217be0298e9548c92d342d994a373e16460cfbcba'
 pm=bound(parent/'manifest.json');evidence=dict(pm['required_evidence'])
 def bind(f,h=None):
  f=Path(f).resolve();actual=sha(f)
  if h is not None:assert actual==h,('hash',f)
  if str(f) in evidence:assert evidence[str(f)]==actual,('historical changed',f)
  evidence[str(f)]=actual
  return json.loads(f.read_text(encoding='utf8')) if f.suffix=='.json' else None
 def embedded(o):
  if isinstance(o,dict):
   if isinstance(o.get('snapshot'),str) and isinstance(o.get('sha256'),str):bind(o['snapshot'],o['sha256'])
   for k,v in o.items():
    if isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) and Path(k).is_absolute():bind(k,v)
    elif isinstance(v,(dict,list)):embedded(v)
  elif isinstance(o,list):
   for x in o:embedded(x)
 bind(parent/'manifest.json');bind(parent/'program.py')
 for n,h in pm['bundle_files'].items():bind(parent/n,h)
 proof=audit();embedded(bind(BOARD/'control86-preplace-audit.json'))
 embedded(bind(ROOT/'tmp/frame-execution/connected-board-video-control84-map1/control84-preplace-audit.json'))
 # These unchanged capture/bridge tests are inherited source evidence, not
 # evidence that the new implementation has passed its physical gates.
 audit83();embedded(bind(BOARD83/'bandwidth83-preplace-audit.json'))
 for rel,status,files_key in (
  ('tmp/board-test/video-gain82-offline1/fullcapture83-1','CAPTURE_720P_EXACT_BYTES_AND_ABORT_RECOVERY_PASS','inputs'),
  ('tmp/frame-execution/video-bandwidth83-wrapper32','ACTUAL_NATIVE_CAPTURE_BOUNDARY_PASS','files')):
  test=ROOT/rel;result=bind(test/'result.json');embedded(result)
  if status:assert result['status']==status
  for name,digest in result[files_key].items():bind(test/name,digest)
 bridge=ROOT/'tmp/board-test/video-gain82-offline1/split-bridge-protocol1'
 test=bind(bridge/'result.json');assert test['status']=='DDR516_AXI_BRIDGE_PASS'
 embedded(bind(bridge/'inputs.json',test['inputs_sha256']))
 reset=bind(BOARD/'control-reset-fanout-audit.json');embedded(reset)
 assert reset['status']=='CONTROL86_FACTORED_CONE_SYNCHRONOUS_RESET_SINKS_PASS_NOT_ROUTED' and reset['map_sha256']==proof['map_sha256']
 assert reset['all_special_sinks_unchanged'] and reset['asynchronous_reset_sinks']==0 and reset['clock_sinks']==0
 embedded(bind(BOARD/'inputs.json'));bind(BOARD/'result.json');bind(BOARD/'native_source.v')
 cdc=bind(BOARD/'control86-cdc-delta.json');assert cdc['status']=='CONTROL86_CDC_DELTA_CLASSIFIED_NOT_ROUTED_PASS' and cdc['new_cross_domain_cones']==0;embedded(cdc)
 bind(BOARD/'service-cdc-inventory-expanded.json',cdc['current_inventory_sha256'])
 pins=ROOT/'tmp/frame-execution/video-control86-pin1';pr=bind(pins/'result.json');assert pr['status']=='CONTROL86_MAPPED_I2C_PINS_PASS'
 assert [x['mode'] for x in pr['cases']]==[0,10,4,6,8] and all(x['exit_code']==0 for x in pr['cases'])
 pi=bind(pins/'inputs.json',pr['inputs_sha256']);embedded(pi)
 prepared=Path(pi['prepared']);q=bind(prepared/'inputs.json',pi['prepared_sha256']);embedded(q);assert q['map_sha256']==proof['map_sha256']
 for row in pr['cases']:bind(pins/f"mode{row['mode']}.log",row['log_sha256'])
 joint=ROOT/'tmp/board-test/video-gain82-offline1/joint-full1080-83-1/result.json';j=bind(joint);embedded(j)
 assert j['status']=='DIAGNOSTIC_COMPLETE' and j['parameters']['lines']==1080 and len(j['observations'])==1
 assert 'completed=1 failed=0 raw=000 aborts=0' in j['observations'][0]
 t=timing_audit(canonical);assert t['current_timing_pass'] and t['identity']['map']==proof['map_sha256'];physical=Path(t['physical'])
 rr=bind(route/'result.json');ri=bind(route/'inputs.json',rr['inputs_sha256']);embedded(ri)
 assert rr['status']=='VIDEO_SERVICE_LEGAL_ROUTE_PASS_NOT_RELEASE' and Path(ri['physical'])==physical
 assert rr['trace_sha256']==sha(physical/'outflow/ti60f225_oob.troutingtraces')
 for n in ('physical_inputs.json','result.json','placement_review_inputs.json','global-bandwidth-review.json','control86-path-review.json','route-review.json','service_clock_profile.json','cnn_ram_placement.json','hard-anchor-check.json'):embedded(bind(physical/n))
 endpoint=bind(physical/'original-gray-endpoints.json');embedded(endpoint)
 assert endpoint['status']=='EXACT_ORIGINAL_GRAY_ENDPOINT_NAMES_PASS' and endpoint['buses']==10 and endpoint['destination_bits']==70 and endpoint['missing']==0
 per=bind(physical/'original-periphery-output-check.json');embedded(per);assert per['status']=='ORIGINAL_PERIPHERY_INPUT_AND_GENERATED_PINS_MATCH_EXCEPT_DATE'
 embedded(bind(canonical/'inputs.json'));bind(canonical/'result.json')
 for f in (canonical/'outflow').glob('canonical_*.rpt'):bind(f)
 gr=bind(gray/'result.json');assert gr['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
 bind(gray/'inputs.json',gr['inputs_sha256']);sr=bind(gray/'service_cdc.json',gr['service_cdc_sha256'])
 assert sr['status']=='VIDEO_SERVICE_ROUTED_CDC_SETTLING_PASS_NOT_RELEASE' and sr['map_sha256']==proof['map_sha256']
 for f in gray.glob('*.json'):embedded(bind(f))
 for f in (gray/'outflow').glob('*.rpt'):bind(f)
 pl=bind(pll/'bound.json');embedded(pl);assert pl['status']=='PLL_SHIFT_ROUTED_SETUP_HIGH_BOUND_PASS'
 for k,v in [('setup_lower_ns',2.5),('common_control_high_lower_ns',10),('common_control_low_lower_ns',10)]:assert pl['bounds'][k]>=v
 embedded(bind(pll/'inputs.json'));bind(pll/'result.json')
 for f in (pll/'outflow').glob('*.rpt'):bind(f)
 for suffix in ('.map.v','.place','.troutingtraces','.vdb'):
  n='outflow/ti60f225_oob'+suffix;assert sha(gray/n)==sha(pll/n)==sha(canonical/n)==sha(physical/n)==sha(bits/n);bind(bits/n)
 br=bind(bits/'result.json');assert br['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED'
 gi=bind(bits/'generation_inputs.json',br['inputs_sha256']);assert Path(gi['canonical'])==canonical
 from deployment_artifact_paths import checked_artifact
 bit=bits/'outflow/ti60f225_oob.bit';art=checked_artifact(br['artifacts'],bits,bit);assert art['sha256']==sha(bit) and art['bytes']==bit.stat().st_size;bind(bit)
 bind(Path(__file__));bind(Path(__file__).with_name('video_control86_physical.py'))
 out.mkdir();shutil.copy2(bit,out/'ti60f225_oob.bit')
 for n in ('coefficients-padded.bin','compute.bin','dma.bin','comparison-isp.bin','comparison-isp.json','sensor_parameters.json','sensor_write_sequence.hex'):shutil.copy2(parent/n,out/n)
 assert sha(out/'coefficients-padded.bin')=='ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523'
 (out/'README.md').write_text(f'''# 86：采集带宽与两条控制链修正

采集突发16→32、预存64拍；解除独立读写RAM之间的两处多余互斥，实际原生网表对应四个LUT。另外将DDR33控制链作512组合等价重排，并将DMA使能1152链作2048组合等价重排，关键地址检查分别由3级减至1级、4级减至3级；综合器另有自动逻辑重组，按源范围、实际目标控制短证明和接口测试验证，未宣称全map形式等价。完整B1-A、150MHz、引脚、82传感器配置和已接受的曲线80均保持。

本包自身原约束WNS {t['wns_ns']:+.3f}ns、WHS {t['whs_ns']:+.3f}ns，XLR {t['resources']['XLRs']['used']}/60800；其他seed未新跑，不冒称全部seed通过。资源95%目标仍未达到。

采集模块两张720p/5529600字节及取消恢复通过；完整1080行受控并发采集RAW溢出0。后者用理想DDR响应器和人工CNN流量，不是整机实测。当前86新路由CDC/Gray、PLL及摄像头实际映射接口短测均绑定在manifest。没有全网表形式等价声明。

下载前等待用户在旁监督，仅临时JTAG、不写Flash。按既有流程检查身份/DDR/摄像头，上传A及程序完整读回后，再恢复曲线80并连续运行。先记录冷启动计数，再按相同时间窗测AI提交、新显示帧和帧损失。旧版约10fps，新版是否达到15fps未知。

SW1恢复历史缺陷未在本轮宣称修复；长期稳定、真实RAW丢帧、15fps以及手势/恶灵叠加等其余职责仍待验证。不要因为短测通过就跳过实际观察。
''',encoding='utf8')
 m=dict(pm)
 for k in ('native_parameter_overlay','effective_native_sha256','rom_overlay','rom_overlay_sha256'):m.pop(k,None)
 m.update(native=str(ROOT/'tmp/frame-execution/video-control86-input1'),board=str(BOARD),map_sha256=proof['map_sha256'],bitstream=str(out/'ti60f225_oob.bit'),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
  resources=t['resources'],wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],selected_seed=t['seed'],canonical=str(canonical),required_evidence=evidence,
  base_manifest=str(parent/'manifest.json'),base_manifest_sha256=sha(parent/'manifest.json'),implementation_basis='fresh86 map/place/route with scoped capture/bridge change and control factoring and accepted82 sensorROM; separate unchanged curve80 upload',
  current_short_proofs=dict(source_clock=str(BOARD/'control86-preplace-audit.json'),mapped_sensor=str(pins/'result.json'),cdc_delta=str(BOARD/'control86-cdc-delta.json'),full_height_contention=str(joint)),
  inherited_board_pass=False,full_system_accepted=False,hardware_access=False,pending=['supervised86 startup and continuous A','actual15fps and same-epoch frame-loss accounting','SW1 input recovery','long-run RAW/display stability','remaining gesture/overlay baseline duties'],
  bundle_files={f.name:sha(f) for f in out.iterdir() if f.is_file()})
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
 template=(parent/'program.py').read_text().replace(sha(parent/'manifest.json'),sha(out/'manifest.json')).replace(pm['bitstream_sha256'],sha(bit));(out/'program.py').write_text(template)
 bound(out/'manifest.json');spec=importlib.util.spec_from_file_location('offline86_validator',out/'program.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.validate();(bits/'offline86-validation.json').write_text(json.dumps(result,indent=2));print('OFFLINE86_READY_WAITING_SUPERVISION',out)
if __name__=='__main__':main()
