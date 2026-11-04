"""Release frozen73 geometry plus a fully audited exposure-only ROM overlay."""
import hashlib,importlib.util,json,shutil,zlib
from pathlib import Path
from run_units import ROOT,HERE,sha
from video_service_board import bound
from update_video_sensor_gain_bram import decode_read,image
from audit_canonical_multiseed import audit
from video_input_isp_control import table

PARENT=ROOT/'output/deployment/board-full-150-video-service73-sensor-20261018'
CANDIDATE=ROOT/'tmp/frame-execution/video-exposure76-candidate1'
OVERLAY=ROOT/'tmp/frame-execution/video-exposure76-bram1'
PINS=ROOT/'tmp/frame-execution/video-exposure76-pin1'
OUT=ROOT/'output/deployment/board-full-150-video-exposure76-20261018'

def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
 assert not OUT.exists()
 assert sha(PARENT/'manifest.json')=='e7f98802962b86e924ac11d9dd8f9800615fae48363918b4556c51cbc6f84a80'
 pm=bound(PARENT/'manifest.json');evidence=dict(pm['required_evidence'])
 def bindfile(path,expected=None):
  p=Path(path).resolve();h=sha(p)
  if expected is not None:assert h==expected,('Changed evidence',str(p))
  if str(p) in evidence:assert evidence[str(p)]==h
  evidence[str(p)]=h
  return json.loads(p.read_text(encoding='utf8')) if p.suffix=='.json' else None
 bindfile(PARENT/'manifest.json');bindfile(PARENT/'program.py')
 for n,h in pm['bundle_files'].items():bindfile(PARENT/n,h)
 cr=bindfile(CANDIDATE/'result.json');br=bindfile(OVERLAY/'result.json');pr=bindfile(PINS/'result.json')
 assert cr['status']=='EXPOSURE76_ROM_EXACT_DELTA_PASS_NOT_RELEASED'
 assert cr['changed_addresses']==[116] and cr['changes']=={'116':dict(register=0x3e01,old=0x46,new=0x54)}
 assert cr['gain_before']==cr['gain_after']==2.503 and cr['verification_bank_unchanged']
 assert (cr['exposure_before_half_lines'],cr['exposure_after_half_lines'])==(1120,1344)
 for f,h in cr['evidence'].items():bindfile(f,h)
 assert br['status']=='OFFICIAL_BRAM_EXPOSURE76_OVERLAY_PASS_NOT_RELEASED'
 assert br['candidate_sha256']==sha(CANDIDATE/'result.json') and br['candidate_native_sha256']==cr['candidate_native_sha256']
 assert br['all_logical_memories_compared']==147 and br['unchanged_memories']==146
 assert br['changed_addresses']==[116] and br['original_payload_exact_roundtrip'] and br['repeat_updated_payload_exact']
 bindfile(HERE/'update_video_exposure76_bram.py',br['runner_sha256'])
 bindfile(HERE/'update_video_sensor_gain_bram.py',br['base_runner_sha256'])
 bindfile(br['header_format_source'],br['header_format_source_sha256'])
 bindfile('E:/efinity-2026.1.132-windows-x64/efinity/2026.1/bin/efx_bram_edit.exe',br['tool_sha256'])
 for name,h in br['base_geometry'].items():
  bindfile(Path(br['base'])/name,h);bindfile(OVERLAY/name,h)
 canonical=Path(pm['canonical']);t=audit(canonical)
 assert t['current_timing_pass'] and (t['wns_ns'],t['whs_ns'])==(pm['wns_ns'],pm['whs_ns'])
 for suffix in ('.map.v','.place','.troutingtraces','.vdb'):
  n='outflow/ti60f225_oob'+suffix
  assert sha(canonical/n)==br['base_geometry'][n]
 assert sha(canonical/'ti60f225_oob.sdc')==br['base_geometry']['ti60f225_oob.sdc']
 # Complete logical memory comparison, including the verification bank.
 before={};after={}
 for label,d in [('before',before),('after',after)]:
  for f in sorted(OVERLAY.glob(label+'-*.log')):
   part=decode_read(f.read_text());assert not set(part)&set(d);d.update(part);bindfile(f)
 assert len(before)==len(after)==147 and set(before)==set(after)
 assert [n for n in before if before[n]!=after[n]]==[br['target']]
 for n in before:
  assert hashlib.sha256(before[n].encode()).hexdigest()==br['memory_before'][n]
  assert hashlib.sha256(after[n].encode()).hexdigest()==br['memory_after'][n]
 target=br['target'];assert before[target][256*20:]==after[target][256*20:]
 low=after[target];high=after['camera_checked/config_rom/ROM__D$1']
 actual=[int(high[5*i:5*(i+1)][::-1],2)<<20|int(low[20*i:20*(i+1)][::-1],2) for i in range(164)]
 assert actual==br['configuration_words']==cr['configuration_words']
 for f in OVERLAY.glob('*.log'):bindfile(f)
 for n in ('sensor-gain-low20.hex','updated.lbf','original.lbf','generate-overlay.cmd'):bindfile(OVERLAY/n)
 bit=OVERLAY/'outflow/ti60f225_oob.bit';bindfile(bit,br['bitstream_sha256'])
 assert bit.stat().st_size==br['bitstream_bytes']
 assert hashlib.sha256(image(bit)[1]).hexdigest()==br['updated_payload_sha256']
 assert pr['status']=='SENSOR73_MAPPED_I2C_PINS_PASS' and pr['unmodified_official_models']
 assert [c['mode'] for c in pr['cases']]==[0,10,4,6,8]
 pi=bindfile(PINS/'inputs.json',pr['inputs_sha256']);prepared=Path(pi['prepared'])
 meta=bindfile(prepared/'inputs.json',pi['prepared_sha256'])
 assert meta['candidate_sha256']==sha(CANDIDATE/'result.json') and meta['map_sha256']==pm['map_sha256']
 bindfile(HERE/'prepare_video_exposure76.py',meta['overlay_builder_sha256'])
 bindfile(Path(meta['original_prepared'])/'inputs.json',meta['original_prepared_sha256'])
 for f,h in meta['files'].items():bindfile(f,h)
 for f,h in pi['files'].items():bindfile(f,h)
 bindfile(HERE/'test_video_sensor73.py',pi['runner_sha256'])
 bindfile(HERE/'test_video_sensor72.py',pi['base_runner_sha256'])
 for c in pr['cases']:
  assert c['exit_code']==0;bindfile(PINS/f"mode{c['mode']}.log",c['log_sha256'])
 for n in ('compile.log','run.do'):bindfile(PINS/n)
 bindfile(Path(__file__))
 data=table(3.0,[1.1431,.73435,1.4135],8)
 assert hashlib.sha256(data).hexdigest()=='3ed37a9cc745220b7ea90dc39dcc7de55e22761423cc2b1ae8f8108f010f805b'
 plan=dict(gamma=3.0,black=8,gains=[1.1431,.73435,1.4135],swap_rb=True,
  sha256=hashlib.sha256(data).hexdigest(),crc32=f'{zlib.crc32(data):08x}',hardware_applied=False)
 OUT.mkdir();shutil.copy2(bit,OUT/bit.name)
 for n in ('coefficients-padded.bin','compute.bin','dma.bin'):shutil.copy2(PARENT/n,OUT/n)
 (OUT/'comparison-isp.bin').write_bytes(data);save(OUT/'comparison-isp.json',plan)
 shutil.copy2(CANDIDATE/'result.json',OUT/'source_overlay.json')
 (OUT/'sensor_write_sequence.hex').write_text(''.join(f'60{w>>9:04x}{(w>>1)&255:02x}\n' for w in actual))
 sp=dict(analog_gain=2.503,exposure_half_lines=1344,gain_registers=[128,52],exposure_registers=[0,84,0])
 save(OUT/'sensor_parameters.json',sp)
 (OUT/'README.md').write_text(f'''# 76：曝光增加20%，保持73完整电路

只改SC431HAI启动配置3e01：46→54，曝光1120→1344半行。3e00/3e02仍00/00，增益80/34（2.503倍）不变。官方BRAM更新工具仅改变一个逻辑ROM地址，全部147逻辑存储逐位对照、其余146不变，包含增益读回期望的校验区不变；原始位流恢复对照和新位流重复生成一致。全部164次I2C配置与正常/旧增益拒绝/错误读回/NACK/复位恢复五场景PASS。

本包有效设计是原73冻结实现加source_overlay.json，不将保留原值的旧map单独称为新设计。原map/place/route/VDB/约束逐字相同，沿用同一选定实现的合法route、原约束WNS {pm['wns_ns']:+.3f}ns、WHS {pm['whs_ns']:+.3f}ns及CDC/Gray/PLL证据；没有重新布局布线、改变频率/引脚或增加时序例外。XLR60462/60800，RAM245/256，DSP160/160，未达95%优化目标。

运行颜色表保持用户当前Gamma3.0、黑位8、RGB1.1431/0.73435/1.4135、swapRB开启。CRC22001da7；需单独上传全768字节并读回，位流默认颜色表没有改变。

用户在旁监督时仅临时JTAG下载，不写Flash。新下载清DDR：先读启动/DDR/摄像头，再上传20KB A系数并完整读回，prepare INPUT_PREVIEW，写入颜色表并完整读回，clear/prepare/live恢复原图。当前摄像头启动诊断验证11项读回，不直接读回曝光三个寄存器；曝光写序列有逐字差分和实际I2C原语短测证据，实板视觉仍需确认。

B1-A/150MHz/计算和DMA程序不变。预览不代表CNN运行或15fps，AI性能、同epoch丢帧、长期稳定及其余既定职责仍待验证。
''',encoding='utf8')
 m=dict(pm);m.update(bitstream=str(OUT/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
  required_evidence=evidence,implementation_basis='frozen73 physical implementation plus official exposure-only ROM overlay76',
  base_manifest=str(PARENT/'manifest.json'),base_manifest_sha256=sha(PARENT/'manifest.json'),
  native_parameter_overlay=str(CANDIDATE/'result.json'),effective_native_sha256=cr['candidate_native_sha256'],
  rom_overlay=str(OVERLAY/'result.json'),rom_overlay_sha256=sha(OVERLAY/'result.json'),
  sensor_parameters=sp,runtime_isp_target=plan,inherited_board_pass=False,full_system_accepted=False,hardware_access=False,
  current_short_proofs=dict(parent73_unchanged_circuitry=str(PARENT/'manifest.json'),official_overlay=str(OVERLAY/'result.json'),affected_pin_level=str(PINS/'result.json')),
  pending=['supervised76 startup and actual brightness comparison','exposure registers not directly read back on board','AI15fps and same-epoch drop accounting','long-run stability and remaining system duties'],
  bundle_files={f.name:sha(f) for f in OUT.iterdir() if f.is_file()})
 save(OUT/'manifest.json',m)
 template=(PARENT/'program.py').read_text().replace(sha(PARENT/'manifest.json'),sha(OUT/'manifest.json')).replace(pm['bitstream_sha256'],sha(bit))
 (OUT/'program.py').write_text(template,encoding='utf8');bound(OUT/'manifest.json')
 spec=importlib.util.spec_from_file_location('exposure76_program',OUT/'program.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 validation=mod.validate();save(OVERLAY/'release-validation.json',validation)
 print(json.dumps(validation),flush=True)

if __name__=='__main__':main()
