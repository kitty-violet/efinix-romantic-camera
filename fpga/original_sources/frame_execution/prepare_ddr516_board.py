"""Freeze the existing connected board plus the isolated 5.16 DDR candidate.

Only DDR integration/clock plan changes. All V4 pad assignments stay identical.
No bitstream or programming action is available in this script.
"""
import argparse,json,re,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,sha
NS={'e':'http://www.efinixinc.com/enf_proj'}
ET.register_namespace('efx',NS['e'])

def verified_write_read_rom(regression):
 """Admit only the ROM actually used by a complete strict turnaround run."""
 regression=regression.resolve()
 result=json.loads((regression/'result.json').read_text())
 assert result['status']=='DDR516_TURNAROUND_PASS','DDR turnaround candidate has not passed'
 assert result['inputs_sha256']==sha(regression/'inputs.json')
 assert result['log_sha256']==sha(regression/'simulation.log')
 meta=json.loads((regression/'inputs.json').read_text())
 assert meta['cases']==688 and result['passed_cases']==688
 assert not meta['calibration_forced'] and meta['rom_guard']['status']=='WR2RD_GUARD_STATIC_AUDIT_PASS'
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 assert sha(regression/'ddr3_controller.bin')==meta['rom_guard']['new_rom_sha256']
 assert sha(regression/'ddr3_controller.vh')==meta['rom_guard']['new_header_sha256']
 return regression,dict(regression=str(regression),result_sha256=sha(regression/'result.json'),
  inputs_sha256=result['inputs_sha256'],log_sha256=result['log_sha256'],
  rom_sha256=meta['rom_guard']['new_rom_sha256'],added_wait_ck=meta['rom_guard']['added_wait_ck'])

def verified_io_bridge(regression):
 """Bind the tested read/precharge-guard ROM and exact IO adapter.

 This ROM is unchanged relative to the admitted turnaround run, but is NOT
 the unmodified vendor ROM. Preserve this distinction in new receipts.
 """
 regression=regression.resolve()
 result=json.loads((regression/'result.json').read_text())
 assert result['status']=='DDR516_TURNAROUND_PASS','DDR IO-buffer candidate has not passed'
 assert result['inputs_sha256']==sha(regression/'inputs.json')
 assert result['log_sha256']==sha(regression/'simulation.log')
 meta=json.loads((regression/'inputs.json').read_text())
 assert meta['cases']==688 and result['passed_cases']==688
 assert not meta['calibration_forced'] and not meta['rom_candidate'] and meta['axi_bridge_candidate']
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 ip=ROOT/'tmp/frame-execution/ddr516-read-pre-guard1/ip/ddr516_reference'
 for name in ('ddr3_controller.bin','ddr3_controller.vh'):assert sha(regression/name)==sha(ip/name)
 return regression,dict(regression=str(regression),result_sha256=sha(regression/'result.json'),
  inputs_sha256=result['inputs_sha256'],log_sha256=result['log_sha256'],
  bridge_sha256=sha(regression/'ddr516_axi_address_bridge.sv'),
  rom_variant='project_read_pre_guard_6ck',vendor_original_rom=False,
  rom_sha256=sha(ip/'ddr3_controller.bin'),header_sha256=sha(ip/'ddr3_controller.vh'))

def adapt_top(text):
 assert text.count('ddr3_top                 u_ddr3_top')==1
 text=text.replace('ddr3_top                 u_ddr3_top','ddr516_v4_controller      u_ddr3_top')
 # Reuse PLL output 3 for the 200 MHz core. The old tx clock is now the
 # same 400 MHz clock as command/CK, so it needs no duplicate clock input.
 text,count=re.subn(r'input\s+tx_cal_clk\s*,','input ddr_core_clk200,',text);assert count==1
 text=text.replace('.tx_cal_clk             (tx_cal_clk          )','.tx_cal_clk             (sdram_clk           )')
 old='.core_clk               (core_clk            )'
 assert text.count(old)==1
 text=text.replace(old,'.core_clk               (ddr_core_clk200     )')
 # Never rely on implicit input-port padding: the Questa board run observed
 # AID/WID=z0 at the native controller for the old 4-to-8-bit connection.
 # Explicitly extend requests and retain all response bits up to the boundary.
 for channel in ('aw','ar'):
  old=f'.s_axi_{channel}id                     (s_axi_{channel}id        ),'
  assert text.count(old)==1,old
  text=text.replace(old,f".s_axi_{channel}id                     ({{{{(8-AXI_ID_WIDTH){{1'b0}}}},s_axi_{channel}id}}),")
 for channel,spacing in (('b','         '),('r','         ')):
  old=f'.s_axi_{channel}id                      (s_axi_{channel}id{spacing}),'
  assert text.count(old)==1,old
  text=text.replace(old,f'.s_axi_{channel}id                      (ddr516_{channel}id),')
 text=text.replace('ddr516_v4_controller      u_ddr3_top',
  '''wire [7:0] ddr516_bid,ddr516_rid;
assign s_axi_bid=ddr516_bid[AXI_ID_WIDTH-1:0];
assign s_axi_rid=ddr516_rid[AXI_ID_WIDTH-1:0];
ddr516_v4_controller      u_ddr3_top''')
 return text

def adapt_periphery(path,dest):
 tree=ET.parse(path);top=tree.getroot();uri=top.tag.split('}')[0]+'}'
 pins_before=[(x.get('name'),x.get('gpio_def')) for x in top.iter(uri+'comp_gpio')]
 pll=next(x for x in top.iter(uri+'pll') if x.get('name')=='DDR3_PLL')
 # Existing V4 crystal is 25 MHz, not the engboard's 50 MHz.
 assert pll.get('ref_clock_name')=='CLK_25M' and pll.get('ref_clock_freq')=='25.0000'
 pll.set('multiplier','2');pll.set('post_divider','2')
 pll.find(uri+'adv_prop').set('feedback_clock_name','ddr_feedback50')
 settings={0:('tx_cal_clk_90edge',6,'false','3'),1:('sdram_clk',6,'false','0'),
           2:('rx_cal_clk',6,'true','0'),3:('ddr_core_clk200',12,'false','0'),4:('ddr_feedback50',48,'false','0')}
 for x in pll.findall(uri+'comp_output_clock'):
  name,div,dynamic,phase=settings[int(x.get('number'))]
  for key,value in dict(name=name,out_divider=str(div),is_dyn_phase=dynamic,phase_setting=phase).items():x.set(key,value)
 for x in top.iter():
  if x.get('clock_name')=='tx_cal_clk':x.set('clock_name','sdram_clk')
 # Preserve the V4 differential clock pin and use the documented inverted
 # 400 MHz clock output. No fabricated data-path exception is introduced.
 ck=next(x for x in top.iter(uri+'comp_gpio') if x.get('name')=='ddr_ck')
 ck.set('mode','clkout');cfg=ck.find(uri+'output_config')
 cfg.set('name','');cfg.set('name_ddio_lo','');cfg.set('register_option','none')
 cfg.set('clock_name','sdram_clk');cfg.set('is_clock_inverted','true');cfg.set('ddio_type','none')
 # Reset is a registered DDR command in the official 5.16 interface.
 reset=next(x for x in top.iter(uri+'comp_gpio') if x.get('name')=='ddr_reset_n')
 reset.find(uri+'output_config').set('register_option','register')
 reset.find(uri+'output_config').set('clock_name','sdram_clk')
 assert pins_before==[(x.get('name'),x.get('gpio_def')) for x in top.iter(uri+'comp_gpio')]
 ET.indent(tree,space='    ');tree.write(dest,encoding='utf-8',xml_declaration=True)
 return dict(all_v4_gpio_assignments_preserved=True,gpio_count=len(pins_before),reference_mhz=25,
             memory_mhz=400,core_mhz=200,feedback_mhz=50,fpll_mhz=2400)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--axi-bridge',type=Path,help='Freeze an isolated public AXI adapter candidate')
 p.add_argument('--write-read-regression',type=Path,help='Use only a hash-bound complete 688-case strict DDR ROM regression')
 p.add_argument('--io-bridge-regression',type=Path,help='Use only a hash-bound complete 688-case strict shared IO-buffer regression')
 p.add_argument('--raw-resident-credit',action='store_true',help='Use tested RAW resident-burst admission in the current public sources');a=p.parse_args()
 rom_regression,rom_proof=verified_write_read_rom(a.write_read_regression) if a.write_read_regression else (None,None)
 assert not (a.write_read_regression and a.io_bridge_regression),'Do not combine independently tested DDR candidates'
 io_regression,io_proof=verified_io_bridge(a.io_bridge_regression) if a.io_bridge_regression else (None,None)
 tested_regression=io_regression or rom_regression
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 base=ROOT/'tmp/frame-execution/connected-board-boot2'
 meta=json.loads((base/'inputs.json').read_text());records=[]
 # Retain the frozen structure/includes but exclude old DDR RTL from synthesis.
 shutil.copytree(base/'rtl',root/'rtl');shutil.copytree(base/'assembly',root/'assembly')
 tree=ET.parse(base/'ti60f225_oob.xml');proj=tree.getroot();design=proj.find('e:design_info',NS)
 for x in list(design.findall('e:design_file',NS)):design.remove(x)
 paths=[]
 for row in meta['inputs']:
  src=Path(row['snapshot']);assert sha(src)==row['sha256'];rel=src.relative_to(base)
  if 'ddr3_controller' in rel.parts and src.suffix in ('.v','.sv'):continue
  dst=root/rel
  if a.raw_resident_credit and src.name in {'guarded_async_fifo.sv','raw32_frame_cdc.sv','camera_axi_write_path.sv'}:
   src=HERE.parent/'video_scaler'/src.name
   shutil.copy2(src,dst)
  if src.name=='ti60f225_oob_top.v':
   live=HERE/'board_integration/ti60f225_oob_top.v'
   compact=lambda x:re.sub(r'\s+','',x).replace('wirecamera_write_fault,camera_read_fault,camera_line_fault;','')
   assert compact(src.read_text())==compact(live.read_text())
   dst.write_text(adapt_top(live.read_text()))
  records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
  paths.append(dst)
 for src in [base/'ddr_memory_map.vh',*base.glob('*.bin')]:
  dst=root/src.name;shutil.copy2(src,dst)
  records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 ip=ROOT/'tmp/frame-execution/ddr516-read-pre-guard1/ip/ddr516_reference'
 bridge=a.axi_bridge.resolve() if a.axi_bridge else HERE/'ddr516_axi_address_bridge.sv'
 if io_regression:
  if a.axi_bridge:assert sha(bridge)==io_proof['bridge_sha256']
  bridge=io_regression/'ddr516_axi_address_bridge.sv'
 assert bridge.name=='ddr516_axi_address_bridge.sv'
 for src in [bridge,HERE/'ddr516_rx_retime.sv',HERE/'board_integration/ddr516_v4_controller.sv',
             ip/'ddr516_reference.sv',ip/'ddr516_reference_define.svh',ip/'ddr3_controller.vh',ip/'ddr3_controller.bin']:
  if rom_regression and src.name in {'ddr3_controller.vh','ddr3_controller.bin'}:src=rom_regression/src.name
  if tested_regression:
   # The vendor supplies different encryption envelopes for synthesis and
   # ModelSim. Bind the matching vendor view; never substitute simulation
   # encrypted HDL into the synthesis project or pretend both hashes match.
   tested_view=ip/'Testbench/modelsim'/src.name if src.name=='ddr516_reference.sv' else src
   assert sha(tested_view)==sha(tested_regression/src.name),f'DDR source differs from passing regression: {src.name}'
   if src.name=='ddr516_reference.sv':
    assert src.read_text().split('//pragma protect',1)[0]==tested_view.read_text().split('//pragma protect',1)[0]
    proof=io_proof if io_regression else rom_proof
    proof.update(controller_synthesis_sha256=sha(src),controller_simulation_sha256=sha(tested_view))
  dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  if src.suffix in ('.v','.sv'):paths.append(dst)
 for i,dst in enumerate(paths,1):
  design.insert(i,ET.Element('{'+NS['e']+'}design_file',name=dst.relative_to(root).as_posix(),version='default',library='default'))
 ET.indent(tree,space='    ');tree.write(root/'ti60f225_oob.xml',encoding='utf-8',xml_declaration=False)
 periphery=adapt_periphery(base/'ti60f225_oob.peri.xml',root/'ti60f225_oob.peri.xml')
 sdc=(base/'ti60f225_oob.sdc').read_text()
 sdc=sdc.replace('create_clock -period 2.500 -name tx_cal_clk [get_ports {tx_cal_clk}]',
                 'create_clock -period 5.000 -name ddr_core_clk200 [get_ports {ddr_core_clk200}]')
 sdc=sdc.replace('create_clock -period 10.000 -name clk_100m [get_ports {clk_100m}]',
                 'create_clock -period 20.000 -name ddr_feedback50 [get_ports {ddr_feedback50}]')
 sdc=re.sub(r'^set_clock_groups -exclusive.*\n','',sdc,flags=re.M)
 sdc=re.sub(r'\btx_cal_clk\b','sdram_clk',sdc)
 sdc=sdc.replace('create_clock -period 2.500 -name tx_cal_clk_90edge',
                 'create_clock -waveform {0.625 1.875} -period 2.500 -name tx_cal_clk_90edge')
 (root/'ti60f225_oob.sdc').write_text('# CANDIDATE: regenerate physical I/O constraints with Interface Designer before STA.\n'+sdc)
 for name in ('ti60f225_oob.xml','ti60f225_oob.peri.xml','ti60f225_oob.sdc'):
  records.append(dict(source='reviewed 5.16 integration transform',snapshot=str(root/name),sha256=sha(root/name)))
 result=dict(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),ddr516=True,
             ddr516_generation=str(ip.parent.parent),periphery=periphery,source_files=list(map(str,paths)),
             boot_sha256=meta['boot_sha256'],sequential_opt=meta['sequential_opt'],operator_sharing=meta['operator_sharing'],
             raw_resident_credit=a.raw_resident_credit,write_read_rom_proof=rom_proof,io_bridge_regression_proof=io_proof,
             physical_constraints_need_regeneration=True,board_release=False)
 (root/'inputs.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(status='DDR516_BOARD_PREPARED',root=str(root),periphery=periphery)),flush=True)
if __name__=='__main__':main()
