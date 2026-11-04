"""Offline-only release of the selected complete video/A implementation.

This assembles a supervised-test bundle only after current physical gates and
the changed-boundary proofs pass. It neither imports USB nor downloads.
"""
import argparse,ast,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_canonical_multiseed import audit as timing_audit
from place_video_service_epoch import audit_delta
from fixed_jtag_loader import validate_payload

PROOFS={
 'video-service-output2':'VIDEO_SERVICE_SPATIAL_DISPLAY_RECOVERY_PASS',
 'video-service-codec-frame1':'FULL_FRAME_CODEC_SOURCE_RGB_PASS',
 'video-service-display-boundary11':'EXACT_NATIVE_DISPLAY_EXTENSION_BOUNDARY_PASS',
 'video-service-units10':'VIDEO_SERVICE_UNITS_PASS_NOT_INTEGRATED',
 'video-service-ram-units1':'VIDEO_SERVICE_UNITS_PASS_NOT_INTEGRATED',
 'video-service-host7':'PASS',
 'video-service-mapped-boot13':'VIDEO_SERVICE_ACTUAL_MAP_BOOT_PASS',
 'video-service-owner-boundary4':'ACTUAL_MAP_OWNER_FAILURE_BOUNDARY_PASS',
 'video-service-compact-map-ram-test1':'VIDEO_SERVICE_INDEXED_SERIAL_NATIVE_PASS',
}

def local_imports(path,found=None):
 found={} if found is None else found;path=path.resolve()
 if str(path) in found:return found
 found[str(path)]=sha(path)
 for n in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
  names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else []
  for name in names:
   other=HERE/(name.split('.')[0]+'.py')
   if other.is_file():local_imports(other,found)
 return found

def main():
 p=argparse.ArgumentParser()
 for name in ('board','canonical','gray','pll','bitstream-root','output'):p.add_argument('--'+name,type=Path,required=name in ('board','output'))
 p.add_argument('--proof-only',action='store_true')
 a=p.parse_args();board=a.board.resolve();out=a.output.resolve()
 assert not out.exists()
 if a.proof_only:assert out.is_relative_to(ROOT/'tmp/board-test') and out.suffix=='.json'
 else:
  assert out.is_relative_to(ROOT/'output/deployment')
  assert all((a.canonical,a.gray,a.pll,a.bitstream_root))
  canonical=a.canonical.resolve();gray=a.gray.resolve();pll=a.pll.resolve();bits=a.bitstream_root.resolve()
 evidence={}
 def bind(path,expected=None):
  path=Path(path).resolve();h=sha(path)
  if expected is not None:assert h==expected,('changed_evidence',str(path))
  evidence[str(path)]=h
  return json.loads(path.read_text()) if path.suffix=='.json' else None
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
 audit_delta(board);bm=bind(board/'inputs.json');br=bind(board/'result.json')
 for row in bm['inputs']:bind(row['snapshot'],row['sha256'])
 mapped=board/'outflow/ti60f225_oob.map.v';bind(mapped,br['map_sha256'])
 graph=bind(board/'retained-map-audit.json');assert graph['map_sha256']==sha(mapped) and graph['conflicts']==0
 native=Path(bm['native']);ng=bind(native/'native-graph-audit.json');file_map(ng)
 graft=bind(native/'graft.json');bind(native/'native_source.v',graph['source_sha256'])
 warning=bind(board/'synthesis-warning-audit.json');assert warning['status']=='ALL_CURRENT_SYNTHESIS_WARNINGS_CLASSIFIED_PASS' and warning['map_sha256']==sha(mapped);file_map(warning['evidence'])
 for name in ('pages-proof.json','nested-epoch-proof.json'):
  r=bind(board/name);assert 'PASS' in r['status'] and r['map_sha256']==sha(mapped);file_map(r)
 current_proofs={}
 for name,status in PROOFS.items():
  root=ROOT/'tmp/frame-execution'/name;r=bind(root/'result.json');assert r['status']==status,(name,r['status'])
  current_proofs[name]=dict(status=status,result_sha256=sha(root/'result.json'))
  if (root/'inputs.json').exists():
   i=bind(root/'inputs.json',r.get('inputs_sha256'));file_map(i,root)
   if 'parent_inputs' in i:
    pi=bind(i['parent_inputs'],i['parent_inputs_sha256']);file_map(pi,Path(i['parent_inputs']).parent)
  for field in ('inputs','sources','files'):
   if field in r:file_map(r[field],root)
  for f in root.glob('*warning*audit*.json'):file_map(bind(f),root)
  logs={sha(f) for f in root.glob('*.log')}
  def expected_logs(node):
   if isinstance(node,dict):
    for k,v in node.items():
     if k=='log_sha256':assert v in logs,(name,'changed_log',v)
     elif isinstance(v,(dict,list)):expected_logs(v)
   elif isinstance(node,list):
    for v in node:expected_logs(v)
  expected_logs(r)
  for f in root.glob('*.log'):bind(f)
 assert bind(ROOT/'tmp/frame-execution/video-service-owner-boundary4/inputs.json')['map_sha256']==sha(mapped)
 boot=bind(ROOT/'tmp/frame-execution/video-service-mapped-boot13/result.json');assert boot['inputs'][str(mapped)]==sha(mapped)
 ram_unit=ROOT/'tmp/frame-execution/video-service-ram-units1/video_dualclock_sample.sv'
 assert sha(ram_unit)==sha(board/'video_dualclock_sample.sv')
 if a.proof_only:
  out.write_text(json.dumps(dict(status='CURRENT_SERVICE_FUNCTIONAL_RELEASE_INPUTS_CHECKED_NOT_PHYSICAL_RELEASE',proofs=current_proofs,evidence=evidence,checker_sha256=sha(__file__),map_sha256=sha(mapped),hardware_access=False,board_release=False),indent=2))
  print('CURRENT_SERVICE_FUNCTIONAL_RELEASE_INPUTS_CHECKED_NOT_PHYSICAL_RELEASE');return
 t=timing_audit(canonical);assert t['current_timing_pass'] and t['identity']['map']==sha(mapped)
 bind(canonical/'result.json');file_map(bind(canonical/'inputs.json'))
 for f in (canonical/'outflow').glob('canonical_*.rpt'):bind(f)
 gr=bind(gray/'result.json');assert gr['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE' and gr.get('hardware_access',False) is False
 bind(gray/'inputs.json',gr['inputs_sha256'])
 for field,name in [('inventory_sha256','clock_crossing_inventory.json'),('jtag_audit_sha256','jtag_cdc_audit.json'),('skew_audit_sha256','gray_skew_audit.json'),('service_cdc_sha256','service_cdc.json')]:bind(gray/name,gr[field])
 # Both the unchanged native crossings and the added service crossings are
 # mandatory. An inventory alone is never an acceptable release receipt.
 service=bind(gray/'service_cdc.json');assert service['status']=='VIDEO_SERVICE_ROUTED_CDC_SETTLING_PASS_NOT_RELEASE' and service['map_sha256']==sha(mapped)
 for f in gray.glob('*.json'):file_map(bind(f),gray)
 for f in (gray/'outflow').glob('*.rpt'):bind(f)
 assert sha(gray/'outflow/ti60f225_oob.map.v')==sha(mapped)
 pr=bind(pll/'bound.json');assert pr['status']=='PLL_SHIFT_ROUTED_SETUP_HIGH_BOUND_PASS'
 for k,v in [('setup_lower_ns',2.5),('common_control_high_lower_ns',10),('common_control_low_lower_ns',10)]:assert pr['bounds'][k]>=v
 file_map(pr);file_map(bind(pll/'inputs.json'));bind(pll/'result.json')
 for suffix in ('.map.v','.place','.vdb','.troutingtraces'):
  name='ti60f225_oob'+suffix
  assert sha(gray/'outflow'/name)==sha(pll/'outflow'/name)==sha(canonical/'outflow'/name)
 generated=bind(bits/'result.json');assert generated['status']=='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED'
 gi=bind(bits/'generation_inputs.json',generated['inputs_sha256']);assert Path(gi['canonical'])==canonical
 for name,h in gi['input_geometry'].items():assert sha(bits/'outflow'/name)==sha(canonical/'outflow'/name)==h
 bit=bits/'outflow/ti60f225_oob.bit';assert generated['artifacts']['outflow/ti60f225_oob.bit']['sha256']==sha(bit)
 parent=ROOT/'output/deployment/board-full-150-continuous-20261015';pm=bind(parent/'manifest.json')
 # Historical tests apply only to the exact retained subgraph. They do not
 # claim this changed control/display system has already worked on hardware.
 for f,h in pm['required_evidence'].items():assert sha(f)==h,('parent_evidence_changed',f)
 bind(ROOT/'tmp/board-test/video-service-review1/parent-release-revalidation1.json')
 integration=ROOT/'tmp/frame-execution/accepted-a-integration1';payload=validate_payload(integration)
 for f in (integration/'result.json',integration/'descriptors/result.json',integration/'descriptors/schema.json'):bind(f)
 runtime=local_imports(HERE/'video_service_board.py')
 programmer=ROOT/'tmp/board-test/program-bringup-status.py';bind(programmer);bind(Path(__file__))
 out.mkdir();shutil.copy2(bit,out/bit.name)
 for name,data in payload.items():(out/(name+'.bin')).write_bytes(data)
 (out/'README.md').write_text(f'''# 完整视频/B1-A监督板测包

这是本机证据绑定的临时JTAG部署包。离线门禁通过不等于实板画面已通过。

- 内部计算150MHz，J4摄像头，HDMI TX 1920×1080@30Hz；HDMI刷新率不是AI实测FPS。
- 原约束建立裕量 {t['wns_ns']:+.3f}ns，保持裕量 {t['whs_ns']:+.3f}ns；实际资源与全部证据见manifest.json。
- 位流SHA-256：`{sha(bit)}`。
- A系数、compute、DMA为已验收集成包；包内*.bin用于硬件装载，不用.pt文件。
- 不写Flash。断电后临时位流和DDR内权重会丢失，需重新下载、装载。

## 使用顺序

1. 用户在旁监督，确认供电、USB、J4摄像头和HDMI屏幕。下载前保持全部按键松开。
2. 先运行program.py的离线核验；确认现场后才使用其显式监督下载入口。program.py默认不访问USB。
3. 临时下载后，以同一manifest及下载回执调用video_service_board.py的observe入口，核查身份、DDR校准及摄像头状态。
4. 必要时用同一位流的TX_PATTERN/DDR_PATTERN/INPUT_PREVIEW隔离显示链；不因请求被接受就宣称屏幕显示成功。
5. prepare入口完成限定DDR测试、A权重及程序上传、完整读回和LOAD校验；READY后才允许启动。
6. SW0（从Type-C向HDMI数第二颗白键）按一下开始连续运行，再按一下请求停止并排空。无需电脑逐帧发命令。
7. SW1（第三颗白键）为逻辑复位，会使READY失效；不要用第一颗CRESET_N代替。

宿主入口为 `{HERE/'video_service_board.py'}`，默认仅离线核验。硬件操作必须同时指定当前下载回执、独立报告路径、execute及supervised。诊断切换需满足停止/排空条件；锁定帧窗单次只读最多4096字节。

实板仍需确认：冷启动及SW1恢复、正常实时画面、同帧数值、连续稳定性和实际AI提交间隔。本包不宣称已达30fps，也不宣称手势/恶灵效果基线已完成。
''',encoding='utf8')
 manifest=dict(status='VIDEO_SERVICE_OFFLINE_RELEASED_FOR_SUPERVISED_TEST',diagnostic_schema=4,service_diagnostic_positions=list(range(192)),
  camera_diagnostic_slots=pm['camera_diagnostic_slots'],frequency_mhz=150,flash_allowed=False,a_integration=str(integration),
  native=str(native),board=str(board),map_sha256=sha(mapped),bitstream=str(out/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
  resources=t['resources'],wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],selected_seed=t['seed'],canonical=str(canonical),
  required_evidence=evidence,runtime_sha256=runtime,current_short_proofs=current_proofs,
  observation_counter_contract=graft['native_binding']['observation_counter_contract'],
  inherited_numerical_evidence=str(parent/'manifest.json'),inherited_board_pass=False,
  user_supervision_required=True,automatic_download_allowed=False,flash_write_allowed=False,ddr_test_released=False,a_weight_load_released=False,full_system_accepted=False,hardware_access=False,
  bundle_files={f.name:sha(f) for f in out.iterdir() if f.is_file()},
  pending=['supervised cold-start and SW1 recovery','actual screen in same-bitstream diagnostic modes and LIVE_A','same-frame numerical readback and actual throughput','electrical HDMI/PHY behavior; unavailable raw internal IP signals are not inferred'])
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 template=(parent/'program.py').read_text()
 template=template.replace(pm['bitstream_sha256'],sha(bit)).replace(sha(parent/'manifest.json'),sha(out/'manifest.json')).replace('OFFLINE_READY_AWAIT_USER_SUPERVISION',manifest['status'])
 assert manifest['bitstream_sha256'] in template and sha(out/'manifest.json') in template
 (out/'program.py').write_text(template,encoding='utf8')
 from video_service_board import bound
 bound(out/'manifest.json')
 print(json.dumps(dict(status=manifest['status'],bundle=str(out),resources=t['resources'],wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],hardware_access=False),ensure_ascii=False))

if __name__=='__main__':main()
