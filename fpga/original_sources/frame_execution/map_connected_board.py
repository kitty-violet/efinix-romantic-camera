"""Freeze one real board assembly and map all implemented components together.
Never invokes interface, P&R, bitstream or programming. Capacity is reported even
when overfull; warning/missing-module/comb-loop failures cannot count as PASS.
"""
import argparse,collections,json,re,shutil,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import HERE,ROOT,EF,sha,run
from pipeline_sources import rgb_pipeline_sources
BASE=ROOT/'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5'
NS={'e':'http://www.efinixinc.com/enf_proj'}
ET.register_namespace('efx',NS['e'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--boot-image',type=Path,required=True);p.add_argument('--overrides-dir',type=Path);p.add_argument('--sequential-opt',action='store_true');p.add_argument('--operator-sharing',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  # Preserve generated/PHY relative includes and memory files verbatim.
  shutil.copytree(BASE/'rtl',root/'rtl')
  # OCR image is a first-class input. Never silently pick an old SPI ROM from
  # a recursive glob (the former ROM addressed a non-existent SPI decoder).
  for f in BASE.glob('*.bin'):
   if not f.name.startswith('EfxSapphireSoc.v_toplevel_system_ramA_'):shutil.copy2(f,root/f.name)
  boot=a.boot_image.resolve();boot_meta=json.loads((boot/'boot.json').read_text())
  rom=(boot/'boot.bin').read_bytes();assert len(rom)==4096 and sha(boot/'boot.bin')==boot_meta['sha256']
  from board_integration.boot_image import make_program
  expected,_=make_program();assert rom==expected,'Boot image differs from current reviewed loader'
  boot_records=[]
  for byte in range(4):
   f=boot/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{byte}.bin'
   assert bytes(int(v,2) for v in f.read_text().split())==rom[byte::4]
   dst=root/f.name;shutil.copy2(f,dst)
   boot_records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst)))
  for n in ('ti60f225_oob.sdc','ti60f225_oob.peri.xml'):
   if (BASE/n).is_file():shutil.copy2(BASE/n,root/n)
  tree=ET.parse(BASE/'ti60f225_oob.xml');proj=tree.getroot();design=proj.find('e:design_info',NS)
  frozen=ROOT/'tmp/frame-execution/program-map2'
  manifest=json.loads((frozen/'inputs.json').read_text());result=json.loads((frozen/'result.json').read_text())
  assert sha(frozen/'inputs.json')==result['inputs_sha256']
  paths=[]
  for r in manifest['inputs']:
   f=Path(r['snapshot']);assert sha(f)==r['sha256'];paths.append(f)
  paths+=rgb_pipeline_sources()+[HERE/'cnn_video_pipeline.sv',HERE/'cnn_apb_control.sv',HERE/'cnn_production_frame.sv',
   HERE.parent/'video_scaler/framed_rgb_async_fifo.sv',
   HERE/'board_integration/ti60f225_oob_top.v',
   ROOT/'tmp/sapphire-memory-ab/area15-i3-attempt2/rtl/EfxSapphireSoc.v']
  if a.overrides_dir:
   extra=a.overrides_dir.resolve();assert extra.is_relative_to(HERE/'variants')
   paths=[extra/f.name if (extra/f.name).is_file() else f for f in paths]
  overrides={f.name:f for f in paths}
  omit={'cnn_quant_q17_top.sv','cnn_tile_descriptor_sequencer.sv','cnn_tile_dma_address_adapter.sv',
   'int8_dual_mac_lane.sv','int8_dual_mac_lane_dspacc.sv','cnn_mac_array_microbench.sv',
   'cnn_linebuffer_pressure.sv','cnn_tile_dma_row_executor.sv','cnn_tile_dma_cdc_bridge.sv',
   'cnn_real_axi_pressure_harness.sv','operator_core_pnr_top.sv','scaler_pnr_harness.sv',
   'rgb_scaler_display_pipeline.sv','gamma_correction.v','cnn_weight_bank64.sv'}
  records=[];used=set()
  old=list(design.findall('e:design_file',NS))
  for entry in old:design.remove(entry)
  for entry in old:
   rel=entry.get('name');oldsrc=(BASE/rel).resolve();name=oldsrc.name
   if name in omit or 'debug_edb' in name:continue
   src=overrides.get(name,oldsrc)
   if name in used:continue
   used.add(name)
   if src==oldsrc and rel.startswith('rtl/'):dst=root/rel
   elif name=='ti60f225_oob_top.v':dst=root/'rtl'/name;shutil.copy2(src,dst)
   else:dst=root/'assembly'/name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
   records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
   ET.SubElement(design,'{'+NS['e']+'}design_file',name=dst.relative_to(root).as_posix(),version='default',library='default')
  for name,src in overrides.items():
   if name in used:continue
   used.add(name);dst=root/'assembly'/name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
   records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
   ET.SubElement(design,'{'+NS['e']+'}design_file',name=dst.relative_to(root).as_posix(),version='default',library='default')
  # Include search follows the actual source directories. Headers are copied
  # beside consumers, in addition to being tracked as synthesis inputs.
  shutil.copy2(HERE.parent/'ddr_fabric/ddr_memory_map.vh',root/'ddr_memory_map.vh')
  syn=proj.find('e:synthesis',NS)
  for key,value in [('mode','speed'),('max_threads','2'),('pack-luts-to-comb4','2'),('seq_opt','1' if a.sequential_opt else '0'),('operator-sharing','1' if a.operator_sharing else '0')]:
   found=syn.find("e:param[@name='"+key+"']",NS)
   if found is None:found=ET.SubElement(syn,'{'+NS['e']+'}param',name=key,value_type='e_integer')
   found.set('value',value)
  # Efinity's project reader enforces XSD child order: all design_file
  # entries belong immediately after top_module, before top_vhdl_arch.
  entries=list(design.findall('e:design_file',NS))
  for entry in entries:design.remove(entry)
  for index,entry in enumerate(entries,1):design.insert(index,entry)
  ET.indent(tree,space='    ')
  xml=root/'ti60f225_oob.xml';tree.write(xml,encoding='utf-8',xml_declaration=False)
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,boot_inputs=boot_records,boot_sha256=boot_meta['sha256'],runner_sha256=sha(__file__),base=str(BASE),sequential_opt=a.sequential_opt,operator_sharing=a.operator_sharing,
   note='Actual connected RTL; board pins/PHY unchanged; AXI user clock 150 MHz uses generated ASYN_AXI_CLK=1, not yet STA qualified'),indent=2))
  (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 1500\nexit /b %errorlevel%\n')
  body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=1800)
  assert re.search(r'map\s*:\s*PASS',body)
  net=root/'outflow/ti60f225_oob.map.v';warn=root/'outflow/ti60f225_oob.warn.log';rpt=root/'outflow/ti60f225_oob.map.rpt'
  warnings=warn.read_text(errors='replace')
  # These generated CPU/PHY unused nets already exist in the frozen V23
  # evidence. Match the exact net AND parameterized module, never waive a
  # new undriven net by warning code alone.
  legacy=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
  pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
  old_undriven=set(re.findall(pattern,legacy,re.M))
  now_undriven=set(re.findall(pattern,warnings,re.M))
  unknown=sorted(now_undriven-old_undriven)
  assert not unknown and 'EFX-0654' not in warnings, 'New undriven net/comb loop: '+repr(unknown)
  dumped_counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
  for key in ('RAM10','DPRAM10','DSP24','DSP48'):assert counts[key]==dumped_counts[key]
  for r in records+boot_records:assert sha(r['snapshot'])==r['sha256']
  result=dict(status='CONNECTED_BOARD_MAP_MEASURED_NOT_RELEASE',resources=counts,
   dumped_net_primitives=dumped_counts,seconds=secs,
   inherited_generated_undriven_signatures=sorted(now_undriven),legacy_warning_sha256=sha(BASE/'outflow/ti60f225_oob.warn.log'),
   inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(rpt),warning_sha256=sha(warn),
   sequential_opt=a.sequential_opt,operator_sharing=a.operator_sharing,resource_fit_proven=False,physical_timing_pass=False,full_baseline=False,board_download_allowed=False,
   open=['full digital video/CPU boot integration validation','gesture/exit/demon/FPS baseline leaf modules','physical P&R/STA','trained weights','board calibration/bandwidth'])
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

