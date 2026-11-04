"""Bind the board simulation, integration RTL and unchanged V4 pad assignment."""
import argparse,json,re,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True);p.add_argument('--simulation',type=Path,required=True);a=p.parse_args()
 board=a.map.resolve();simulation=a.simulation.resolve()
 bm=json.loads((board/'inputs.json').read_text());sm=json.loads((simulation/'inputs.json').read_text())
 assert bm['ddr516'] and sm['ddr516'] and sm['base_inputs_sha256']==sha(board/'inputs.json')
 partition=None
 if sm.get('native'):
  from production_board_binding import validate
  partition=validate(simulation,sm)
 for meta in (bm,sm):
  for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 sims={Path(r['snapshot']).name:r['sha256'] for r in sm['inputs']};common=[];paired=[]
 for row in bm['inputs']:
  path=Path(row['snapshot']);name=path.name
  if path.suffix not in ('.v','.sv'):continue
  if name in ('EfxSapphireSoc.v','csi_rx_controller.sv','ddr516_reference.sv'):
   paired.append(name);continue
  assert name in sims and sims[name]==row['sha256'],name
  common.append(name)
 assert len(sm['csi_pairs'])==28 and set(paired)=={'EfxSapphireSoc.v','csi_rx_controller.sv','ddr516_reference.sv'}
 # Source equality alone previously admitted an erroneous implicit 4->8-bit
 # native ID connection. Require the explicit production boundary adaptation.
 compact=re.sub(r'\s+','',(board/'rtl/ti60f225_oob_top.v').read_text())
 for channel in ('aw','ar'):
  assert f".s_axi_{channel}id({{{{(8-AXI_ID_WIDTH){{1'b0}}}},s_axi_{channel}id}})" in compact
 for channel in ('b','r'):
  assert f'.s_axi_{channel}id(ddr516_{channel}id)' in compact
  assert f'assigns_axi_{channel}id=ddr516_{channel}id[AXI_ID_WIDTH-1:0];' in compact
 syn=(board/'ddr516_reference.sv').read_text();sim=(simulation/'ddr516_reference.sv').read_text()
 assert syn.split('//pragma protect',1)[0]==sim.split('//pragma protect',1)[0]
 root=ET.parse(board/'ti60f225_oob.peri.xml').getroot();ns=root.tag.split('}')[0]+'}'
 old=ET.parse(Path(bm['base'])/'ti60f225_oob.peri.xml').getroot()
 pins=lambda tree:{x.get('name'):x.get('gpio_def') for x in tree.iter(ns+'comp_gpio')}
 assert pins(root)==pins(old)
 pll=next(x for x in root.iter(ns+'pll') if x.get('name')=='DDR3_PLL')
 assert (pll.get('multiplier'),pll.get('pre_divider'),pll.get('post_divider'))==('2','1','2')
 clocks={int(x.get('number')):x.attrib for x in pll.findall(ns+'comp_output_clock')}
 expected={0:('tx_cal_clk_90edge','6','3','false'),1:('sdram_clk','6','0','false'),
           2:('rx_cal_clk','6','0','true'),3:('ddr_core_clk200','12','0','false'),4:('ddr_feedback50','48','0','false')}
 for n,values in expected.items():assert tuple(clocks[n][k] for k in ('name','out_divider','phase_setting','is_dyn_phase'))==values
 ck=next(x for x in root.iter(ns+'comp_gpio') if x.get('name')=='ddr_ck')
 assert ck.get('mode')=='clkout' and ck.find(ns+'output_config').get('is_clock_inverted')=='true'
 assert not re.search(r'^\s*set_clock_groups\s+-exclusive', (board/'ti60f225_oob.sdc').read_text(),re.M)
 assert not any('/ddr3_controller/' in str(x).replace('\\','/') for x in sm['source_files'])
 if sm.get('dsp_pipe_hierarchy_flattened'):
  proof=json.loads((simulation/'dsp_flat_proof.json').read_text())
  assert proof['status']=='DSP24_BOARD_PIPE_HIERARCHY_EQUIVALENCE_PASS'
  assert proof['flat_sha256']==sha(simulation/'efx_dsp24.v')
 data=dict(status='DDR516_BOARD_SOURCE_BINDING_PASS',common_verbatim_rtl=len(common),common_names=common,
  official_paired_views=paired,v4_pads_unchanged=len(pins(root)),clock_plan=clocks,
  map_inputs_sha256=sha(board/'inputs.json'),simulation_inputs_sha256=sha(simulation/'inputs.json'),
  dsp_simulation_pipe_flattened=sm.get('dsp_pipe_hierarchy_flattened',False),
  explicit_native_id_boundary=True,production_partition=partition,
  functional_frames_pass=False,resource_fit=False,physical_timing_pass=False,board_release=False)
 (simulation/'board_source_audit.json').write_text(json.dumps(data,indent=2));print(json.dumps({k:v for k,v in data.items() if k not in ('common_names','clock_plan')}))
if __name__=='__main__':main()
