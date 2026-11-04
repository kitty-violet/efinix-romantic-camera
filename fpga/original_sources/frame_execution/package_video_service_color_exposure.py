"""Incremental color release: exact retained proof plus own routed evidence.

Retained64 tests are reused only through the complete actual-map/source delta.
This candidate must supply its own STA, routed CDC, PLL and generated bitstream.
No programming or USB calls; the resulting package requires supervised use.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_canonical_multiseed import audit as timing_audit
from place_video_service_color_exposure import audit_delta
from fixed_jtag_loader import validate_payload
from package_video_service import local_imports
from deployment_artifact_paths import checked_artifact


def main():
 p=argparse.ArgumentParser()
 for n in ('board','canonical','gray','pll','bitstream-root','output'):
  p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();canonical=a.canonical.resolve();gray=a.gray.resolve()
 pll=a.pll.resolve();bits=a.bitstream_root.resolve();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
 parent=ROOT/'output/deployment/board-full-150-video-service64-clock-20261017'
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
 pm=bind(parent/'manifest.json','b879ed32018a1d08690e1d25b14cdb84ac2ef7b039202c1e45477142e80f076d')
 assert pm['status']=='VIDEO_SERVICE_OFFLINE_RELEASED_FOR_SUPERVISED_TEST'
 file_map(pm['required_evidence']);file_map(pm['runtime_sha256'])
 for n,h in pm['bundle_files'].items():bind(parent/n,h)
 audit_delta(board)
 bm=bind(board/'inputs.json');br=bind(board/'result.json');file_map(bm)
 native=Path(bm['native']);assert Path(bm['parent'])==Path(pm['board'])
 mapped=board/'outflow/ti60f225_oob.map.v';bind(mapped,br['map_sha256'])
 delta=bind(board/'color-map-audit.json');file_map(delta)
 source_delta=bind(native/'color-exposure-independent-audit.json');file_map(source_delta)
 assert delta['source_sha256']==source_delta['source_sha256']==sha(native/'native_source.v')
 assert delta['changed_public_cells']==17 and delta['all_other_public_cells_and_parameters_identical']
 warnings=bind(board/'color-warnings-review.json');assert warnings['status']=='COLOR_INCREMENTAL_WARNINGS_REVIEWED';file_map(warnings)
 bind(board/'color-warnings-audit.json')
 inventory=bind(board/'service-cdc-inventory-expanded.json');file_map(inventory)
 assert inventory['map_sha256']==sha(mapped) and not inventory['unclassified']
 unit=ROOT/'tmp/frame-execution/video-service-color-arithmetic1'
 numeric=bind(unit/'result.json');assert numeric['status']=='ACTUAL_MAPPED_BAYER_EXHAUSTIVE_ARITHMETIC_PASS'
 assert numeric['comparisons']==131072
 file_map(bind(unit/'inputs.json',numeric['inputs_sha256']));file_map(numeric,unit)
 for f in unit.glob('*.log'):bind(f)
 cfa=ROOT/'tmp/board-test/video-service64-color-audit4';bind(cfa/'result.json')
 for f in cfa.glob('*'):
  if f.is_file() and f.suffix in ('.v','.json','.log','.py'):bind(f)
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
 runtime=local_imports(HERE/'video_service_board.py')
 for f in (Path(__file__),HERE/'deployment_artifact_paths.py',HERE/'video_service_color_pll.py',ROOT/'tmp/board-test/program-bringup-status.py'):bind(f)
 out.mkdir();shutil.copy2(bit,out/bit.name)
 for name,data in payload.items():(out/(name+'.bin')).write_bytes(data)
 (out/'README.md').write_text(f'''# B1-A 150MHz：Bayer与曝光修正版监督板测包

- 固定B1和A系数/程序保持；只修正一组红蓝插值连接及摄像头曝光初始化。
- 曝光3e00/01/02改为00/ba/d0，禁止把ROM校验当成传感器实板读回。
- J4摄像头、HDMI TX 1080p30；原约束WNS {t['wns_ns']:+.3f}ns，WHS {t['whs_ns']:+.3f}ns。
- 当前位流SHA：{sha(bit)}。
- 权重限制仍为guarded-A-zero-v1；不支持任意替换权重。上板前必须用户现场监督，不写Flash。
- 临时下载后先检查启动、DDR与摄像头，再上传本包A系数/程序并完整读回；LOAD成功后连续运行。
- SW0（Type-C向HDMI数第二颗白键）启动/停止；SW1第三颗为逻辑复位，复位后须重新准备。
- 当前尚无67实板画面、RAW损失和吞吐验证；64实测约6.34fps，不能当作67结果或宣称15fps已达成。
- 本包默认program.py只离线核验。正常断电会丢失临时位流和DDR权重。

板测比较同一场景的偏色/曝光，核查RAW新取消、准入/采集失败/CNN提交、显示新帧与重复帧，所有计数保留各自采样时间。不得用30Hz显示重复刷新冒充AI FPS。
''',encoding='utf-8')
 keep=('diagnostic_schema','service_diagnostic_positions','camera_diagnostic_slots','frequency_mhz','flash_allowed','a_integration','coefficient_policy','observation_counter_contract')
 m={k:pm[k] for k in keep}
 m.update(status='VIDEO_SERVICE_OFFLINE_RELEASED_FOR_SUPERVISED_TEST',native=str(native),board=str(board),map_sha256=sha(mapped),
  bitstream=str(out/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,resources=t['resources'],
  wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],selected_seed=t['seed'],canonical=str(canonical),
  required_evidence=evidence,runtime_sha256=runtime,
  current_short_proofs=dict(retained64=dict(status='REUSED_THROUGH_EXACT_SOURCE_AND_MAP_DELTA',parent_manifest=str(parent/'manifest.json'),parent_manifest_sha256=sha(parent/'manifest.json')),
    color_source_sha256=sha(native/'color-exposure-independent-audit.json'),color_map_sha256=sha(board/'color-map-audit.json'),bayer_arithmetic=numeric),
  inherited_numerical_evidence=str(parent/'manifest.json'),inherited_board_pass=False,
  user_supervision_required=True,automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,a_weight_load_released=False,
  full_system_accepted=False,hardware_access=False,bundle_files={f.name:sha(f) for f in out.iterdir() if f.is_file()},
  pending=['supervised67 cold start/reset and picture comparison','real exposure/sensor cadence and brightness','RAW loss and timed AI/display rate','same-frame board numerical evidence','gesture/evil effect/FPS overlay and target performance'])
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
 template=(parent/'program.py').read_text()
 assert template.count(pm['bitstream_sha256']) and template.count(sha(parent/'manifest.json'))
 template=template.replace(pm['bitstream_sha256'],sha(bit)).replace(sha(parent/'manifest.json'),sha(out/'manifest.json'))
 (out/'program.py').write_text(template,encoding='utf-8')
 from video_service_board import bound
 bound(out/'manifest.json')
 print(json.dumps(dict(status=m['status'],bundle=str(out),resources=t['resources'],wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],hardware_access=False)))


if __name__=='__main__':main()
