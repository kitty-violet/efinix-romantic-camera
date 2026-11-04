"""Bind a sensor-data-only official BRAM overlay to an unchanged released route.

This deliberately does not claim that the old map contains the updated ROM.
Release = frozen implementation + full-memory audited overlay + affected pin test.
"""
import argparse,hashlib,json,re,shutil,zlib
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_canonical_multiseed import audit
from package_video_service import local_imports
from video_service_board import bound
from update_video_sensor_gain_bram import decode_read,image
from video_input_isp_control import table


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 assert out.is_relative_to(ROOT/'output/deployment') and not out.exists()
 parent=ROOT/'output/deployment/board-full-150-video-service69-input-isp-20261017'
 pm=bound(parent/'manifest.json');assert sha(parent/'manifest.json')=='a8e467fde5ca08bfb21cf39d76d85eb8450db3f4bc3da676cc22e5df2788fd65'
 candidate=ROOT/'tmp/frame-execution/video-sensor-exposure71-candidate1'
 overlay=ROOT/'tmp/frame-execution/video-sensor-exposure71-bram1'
 pins=ROOT/'tmp/frame-execution/video-sensor-exposure71-pin8'
 canonical=ROOT/'tmp/frame-execution/video-sensor-gain70-canonical1'
 evidence=dict(pm['required_evidence'])
 def bindfile(f,expected=None):
  f=Path(f).resolve();h=sha(f)
  if expected is not None:assert h==expected,('changed evidence',str(f))
  if str(f) in evidence:assert evidence[str(f)]==h
  evidence[str(f)]=h
  return json.loads(f.read_text(encoding='utf8')) if f.suffix=='.json' else None
 bindfile(parent/'manifest.json');bindfile(parent/'program.py')
 for n,h in pm['bundle_files'].items():bindfile(parent/n,h)
 cr=bindfile(candidate/'result.json');br=bindfile(overlay/'result.json');pr=bindfile(pins/'result.json')
 assert cr['status']=='OFFLINE_EXPOSURE_ROM_EXACT_DELTA_PASS_NOT_RELEASED'
 assert cr['gain_readback_contract_unchanged'] and cr['gain_before']==cr['gain_after']==2.021
 assert cr['exposure_before_half_lines']==2989 and cr['exposure_after_half_lines']==1494
 assert br['candidate_sha256']==sha(candidate/'result.json') and br['candidate_native_sha256']==cr['candidate_native_sha256']
 assert br['all_logical_memories_compared']==147 and br['unchanged_memories']==146
 assert br['changed_addresses']==[116,117] and br['original_payload_exact_roundtrip'] and br['repeat_updated_payload_exact']
 for f,h in cr['evidence'].items():bindfile(f,h)
 bindfile(candidate/'candidate_i2c_master_reg_rom.v',cr['candidate_rom_sha256'])
 bindfile(HERE/'update_video_sensor_gain_bram.py',br['runner_sha256'])
 bindfile(br['header_format_source'],br['header_format_source_sha256'])
 bindfile(Path('E:/efinity-2026.1.132-windows-x64/efinity/2026.1/bin/efx_bram_edit.exe'),br['tool_sha256'])
 for name,h in br['base_geometry'].items():
  bindfile(Path(br['base'])/name,h);bindfile(overlay/name,h)
 t=audit(canonical);assert t['current_timing_pass'] and t['wns_ns']==pm['wns_ns'] and t['whs_ns']==pm['whs_ns']
 assert t['identity']['map']==pm['map_sha256']==br['base_geometry']['outflow/ti60f225_oob.map.v']
 for suffix in ('.map.v','.place','.troutingtraces','.vdb'):
  name='outflow/ti60f225_oob'+suffix
  assert sha(canonical/name)==br['base_geometry'][name]
  bindfile(canonical/name)
 assert sha(canonical/'ti60f225_oob.sdc')==br['base_geometry']['ti60f225_oob.sdc']
 for f in canonical.glob('*.json'):bindfile(f)
 for f in (canonical/'outflow').glob('canonical_*.rpt'):bindfile(f)
 for f in (canonical/'sta.log',canonical/'checks.log',canonical/'ti60f225_oob.sdc'):bindfile(f)
 before={};after={}
 for label,d in [('before',before),('after',after)]:
  for f in sorted(overlay.glob(label+'-*.log')):
   part=decode_read(f.read_text());assert not set(part)&set(d);d.update(part);bindfile(f)
 assert len(before)==len(after)==147 and set(before)==set(after)
 assert [n for n in before if before[n]!=after[n]]==[br['target']]
 for n in before:
  assert hashlib.sha256(before[n].encode()).hexdigest()==br['memory_before'][n]
  assert hashlib.sha256(after[n].encode()).hexdigest()==br['memory_after'][n]
 low=after[br['target']];high=after['camera_checked/config_rom/ROM__D$1']
 actual=[int(high[5*i:5*(i+1)][::-1],2)<<20|int(low[20*i:20*(i+1)][::-1],2) for i in range(164)]
 assert actual==br['configuration_words']
 assert [actual[i] for i in (115,116,117,161,162)]==[0x3e00<<9,0x3e01<<9|0x5d<<1,0x3e02<<9|0x60<<1,0x3e08<<9|0x80<<1,0x3e09<<9|0x2a<<1]
 for f in overlay.glob('*.log'):bindfile(f)
 for f in ('sensor-gain-low20.hex','updated.lbf','original.lbf','generate-overlay.cmd'):bindfile(overlay/f)
 bit=overlay/'outflow/ti60f225_oob.bit';bindfile(bit,br['bitstream_sha256'])
 assert bit.stat().st_size==br['bitstream_bytes']
 assert hashlib.sha256(image(bit)[1]).hexdigest()==br['updated_payload_sha256']
 assert pr['status']=='ACTUAL_MAPPED_EXPOSURE_CONTROLLER_PIN_TESTS_PASS' and [c['mode'] for c in pr['cases']]==[0,4,6,8] and pr['unmodified_official_models']
 pi=bindfile(pins/'inputs.json',pr['inputs_sha256'])
 assert pi['candidate_sha256']==sha(candidate/'result.json') and pi['base_map_sha256']==pm['map_sha256']
 bindfile(HERE/'test_video_sensor_rom_overlay_modelsim.py',pi['runner_sha256'])
 bindfile(HERE/'test_video_sensor_rom_overlay.py',pi['extraction_runner_sha256'])
 source=bindfile(pi['prepared_inputs'],pi['prepared_inputs_sha256'])
 assert source['candidate_sha256']==pi['candidate_sha256'] and source['base_map_sha256']==pi['base_map_sha256']
 for f,h in source['files'].items():bindfile(f,h)
 for f,h in pi['files'].items():bindfile(f,h)
 bindfile(pins/'run.do')
 for c in pr['cases']:
  assert c['exit_code']==0;bindfile(pins/f"mode{c['mode']}.log",c['log_sha256'])
 bindfile(pins/'compile.log')
 host=ROOT/'tmp/board-test/video-service69-board1'
 coeff=bindfile(host/'preview-coeff-host-tests1.json');restore=bindfile(host/'restore-preview-tests1.json')
 assert coeff['status']==restore['status']=='PASS' and coeff['tests']==6 and restore['tests']==9
 for receipt,helper,test in [(coeff,'video_service_stage_preview_coefficients.py','test_video_service_stage_preview_coefficients.py'),(restore,'video_service_restore_preview.py','test_video_service_restore_preview.py')]:
  bindfile(HERE/helper,receipt['helper_sha256']);bindfile(HERE/test,receipt['test_sha256'])
 bindfile(HERE/'video_service_reprepare_preview.py',restore['prepare_sha256'])
 runtime=dict(pm['runtime_sha256'])
 for n in ('video_service_stage_preview_coefficients.py','video_service_reprepare_preview.py','video_service_restore_preview.py','video_service_scene_sample.py'):
  local_imports(HERE/n,runtime)
 bindfile(Path(__file__))
 # This is the exact last measured table, not a new WB fit or visual PASS.
 plan=bindfile(host/'contrast-calibration1-plan.json')
 data=table(plan['gamma'],plan['gains_rgb'],plan['black'])
 assert hashlib.sha256(data).hexdigest()==plan['table_sha256'] and f'{zlib.crc32(data):08x}'==plan['table_crc32']
 out.mkdir();shutil.copy2(bit,out/bit.name)
 for n in ('coefficients-padded.bin','compute.bin','dma.bin'):shutil.copy2(parent/n,out/n)
 (out/'comparison-isp.bin').write_bytes(data)
 shutil.copy2(candidate/'candidate_i2c_master_reg_rom.v',out/'sensor_configuration.v')
 shutil.copy2(candidate/'result.json',out/'source_overlay.json')
 shutil.copy2(host/'contrast-calibration1-plan.json',out/'comparison-isp.json')
 readme=f'''# 71：曝光降低后的完整150MHz视频/A监督测试包

仅两项传感器ROM值变化：3e01 ba→5d、3e02 d0→60；曝光2989→1494半行（约原先50%）。3e00=00、模拟增益3e08/3e09=80/2a（2.021倍）与实际控制器读回期望保持一致。低增益70候选不放行。

这是原69的冻结实现加官方BRAM内容覆盖。原map/native/VDB保留原ROM值；source_overlay.json、sensor_configuration.v、官方更新及147存储读回记录共同定义新源。不能把旧map单独当作71的全部功能定义。B1-A、权重、计算/DMA程序、150MHz、引脚、复位和时钟约束不变；逻辑电路与连线完全未改。

原约束重新核验WNS {t['wns_ns']:+.3f}ns、WHS {t['whs_ns']:+.3f}ns；原合法route、CDC/Gray、PLL通过几何逐字一致复用。147逻辑存储中146不变，目标ROM仅上述两个地址变化；全部164配置字核对。当前实际映射控制器短测覆盖完整164次配置/13次读回、错误读回拒绝、NACK停止和配置中复位恢复。

XLR60608/60800（99.68%，剩192），RAM245/256，DSP160/160。未达到约95%资源目标；其他seed历史不改写。位流SHA {sha(bit)}。

## 下次监督板测

1. 先确认用户已在板旁，电源/USB/J4/HDMI接好；本次没有访问板子。program.py默认只做离线核验。得到现场确认后才临时JTAG下载；不写Flash。旧69的program3回执不可用于本包。
2. 只读确认新配置成功、DDR校准、时钟/复位及错误状态。使用新的71回执。先运行video_service_stage_preview_coefficients.py，明确上传并完整读回20KB系数；必须给preview-coeff-host-tests1.json，不能假定掉电后DDR仍有系数。
3. video_service_prepare_preview.py选择INPUT_PREVIEW并LOAD到READY。只在停止排空状态，用video_service_input_isp.py上传comparison-isp.json指定的Gamma1.6、RGB增益1.1431/0.773/1.4135、black0、swap=yes；完整768字节读回CRC dbe557c4。这是延续上次条件以单独比较曝光，不代表这张表已通过画质验收。位流默认表仍为69的Gamma2.2中性表，不能混为同一条件。
4. 使用video_service_restore_preview.py顺序执行明确关彩条/清inspection、重新LOAD恢复READY、LIVE并验证新提交；传restore-preview-tests1.json。该入口不冻结画面。让用户看到实时画面并调好角度；只有明确回复可以冻结之后才能STOP采样。
5. 颜色比较完成后，再转A完整装载：STOP排空保存记录；按SW1释放旧槽，确认模式0/DDR恢复/ISP CRC；系数、compute、DMA全部上传、完整读回及CRC验证，之后连续运行。原图预览不等于CNN已装载。
6. FPS与丢帧使用同一次复位/配置epoch：保存停止排空边界，LIVE后定时30秒读数，再STOP排空并重复读取稳定计数；分别记录源事件、准入/取消、CNN启动/完成、新显示/重复显示及RAW标志。不得拿不同窗口相减。APB58只取停止后稳定末帧周期。当前71未实测；此前67约10fps，15fps尚未达到。

## 已知限制

曝光降低的实际色彩、亮部细节、暗部噪声/层次需上板确认；不能恢复已饱和历史帧，也未排除光学散射。RAW拒收取消、偶发停流及自主失联没有因本改动宣称消除；画面正常也必须记录RAW。手势/恶灵特效及目标性能仍属于完整系统待办，本包只是下一轮受限监督测试放行。
'''
 (out/'README.md').write_text(readme,encoding='utf8')
 m=dict(pm)
 m.update(bitstream=str(out/bit.name),bitstream_sha256=sha(bit),bitstream_bytes=bit.stat().st_size,
  canonical=str(canonical),wns_ns=t['wns_ns'],whs_ns=t['whs_ns'],required_evidence=evidence,runtime_sha256=runtime,
  implementation_basis='frozen69 physical implementation plus official ROM-data overlay71',
  base_manifest=str(parent/'manifest.json'),base_manifest_sha256=sha(parent/'manifest.json'),
  native_parameter_overlay=str(candidate/'result.json'),effective_native_sha256=cr['candidate_native_sha256'],
  rom_overlay=str(overlay/'result.json'),rom_overlay_sha256=sha(overlay/'result.json'),
  inherited_board_pass=False,full_system_accepted=False,hardware_access=False,
  current_short_proofs=dict(parent69_unchanged_circuitry=str(parent/'manifest.json'),official_overlay=str(overlay/'result.json'),affected_pin_level=str(pins/'result.json'),host_restore_tests=str(host/'restore-preview-tests1.json')),
  pending=['supervised71 actual exposure/colour and reset comparison','same-epoch AI FPS and lost/repeated-frame accounting','RAW/startup and long-run stability','gesture/evil effects/target performance'],
  bundle_files={f.name:sha(f) for f in out.iterdir() if f.is_file()})
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')
 template=(parent/'program.py').read_text().replace(sha(parent/'manifest.json'),sha(out/'manifest.json')).replace(pm['bitstream_sha256'],sha(bit))
 (out/'program.py').write_text(template,encoding='utf8');bound(out/'manifest.json')
 print(json.dumps(dict(status=m['status'],bundle=str(out),manifest_sha256=sha(out/'manifest.json'),bitstream_sha256=sha(bit),required_evidence=len(evidence),hardware_access=False)),flush=True)


if __name__=='__main__':main()
