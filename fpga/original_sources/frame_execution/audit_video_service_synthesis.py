"""Close every current synthesis warning against primitives or exact lineage."""
import argparse,collections,json,pickle,re
from pathlib import Path
from run_units import ROOT,sha
from probe_native_open_ports import protected_pins
from audit_native_roundtrip import tokens

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--boot',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();parent=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1'
 src=board/'native_source.v';warn=board/'outflow/ti60f225_oob.warn.log';oldwarn=parent/'outflow/ti60f225_oob.warn.log'
 text=src.read_text();body=warn.read_text();old=oldwarn.read_text();out=board/'synthesis-warning-audit.json';assert not out.exists()
 key,g,_,ps,_=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert key[:2]==[sha(src),sha(board/'outflow/ti60f225_oob.map.v')]
 proof=json.loads((board/'retained-map-audit.json').read_text());assert proof['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and proof['map_sha256']==key[1]
 boot=json.loads((a.boot/'result.json').read_text());assert boot['status']=='VIDEO_SERVICE_ACTUAL_MAP_BOOT_PASS' and boot['inputs'][str(board/'outflow/ti60f225_oob.map.v')]==key[1]
 def norm(line):
  line=re.sub(r'\s*\(E:[^\r\n]*:\d+\)','',line)
  return line.replace('protected_inst/','').strip()
 inherited={norm(l) for l in old.splitlines() if l.startswith('[')}
 cells={};current=None
 for i,line in enumerate(text.splitlines(),1):
  m=re.match(r'\s*(EFX_\w+)\s+(\S+)\s*\(',line)
  if m:current=(m[1],m[2].lstrip('\\'))
  if current:cells[i]=current
  if ';' in line:current=None
 counts=collections.Counter();ports=collections.Counter();new=[]
 protected_redundant={
 'inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/depacketizer_inst/genblk4.ecc_check_inst/corrected_pkt_header[25:24]',
 'inst_efx_csi2_rx/u_efx_csi2_rx/dphy_rx_inst/genblk2[0].rx_dword_align_inst/n536[3:2]',
 'inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/pixel_data_i[63:60]',
 'inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/pixel_data_i[7:0]',
 'u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/fifo_rd/xefx_fifo_ctl/async_clk.raddr_cntr_sync_g2b[5:0]',
 'u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/fifo_rd/xefx_fifo_ctl/async_clk.waddr_cntr_gry_sync_c[6:0]',
 }
 protected_redundant.update('u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/controller_main/bram_u2/u_efx_fifo_top/xefx_fifo_ctl/async_clk.'+n for n in ('waddr_cntr_gry_sync_c[6:0]','waddr_cntr_sync_g2b[5:0]','raddr_cntr_gry_sync_c[6:0]','raddr_cntr_sync_g2b[5:0]'))
 opens=re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",body)
 samples=tokens(protected_pins(text)['u_ddr3_top/samples'])
 assert len(opens)==10 and sorted(39-samples.index(n) for n in opens)==[12,13,14,15,32,33,36,37,38,39]
 assert all(len(re.findall(r'\b'+re.escape(n)+r'\b',text))==1 for n in opens)
 assert set(re.findall(r'Net (.*?) is dangling',body))==set(re.findall(r'Net (.*?) is dangling',old))
 owner=(board/'frame_pair_owner.sv').read_text();modes=(board/'video_service_modes.sv').read_text();health=(board/'video_service_health.sv').read_text();boottext=(board/'cnn_preview_boot.sv').read_text()
 for line in body.splitlines():
  if not line.startswith('['):continue
  m=re.search(r"port '([^']+)'.*\(((?:VERI|VDB)-\d+)\).*native_source.v:(\d+)\)",line)
  if m and int(m[3]) in cells:
   kind,n=cells[int(m[3])];pin=m[1];assert g[n][0]==kind and not g[n][1].get(pin),(n,pin)
   if kind=='EFX_LUT4':
    assert re.fullmatch('I[0-3]',pin);bit=int(pin[1]);mask=ps[n]['LUTMASK'][1]
    assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16)),(n,pin)
    reason='lut_unused_input_truth_table'
   else:
    assert (kind,pin) in {('EFX_ADD','O'),('EFX_ADD','CO'),('EFX_SRL8','Q7'),('EFX_DSP24','O'),('EFX_DSP24','CASCOUT'),('EFX_DSP48','CASCOUT')},(kind,pin)
    reason='unused_primitive_output'
   counts[reason]+=1;ports[kind+'.'+pin]+=1;continue
  if norm(line) in inherited:counts['unchanged_parent_warning']+=1;continue
  if 'Found ' in line and 'warnings in the post-synthesis netlist' in line:counts['tool_summary']+=1;continue
  if "using initial value of 'k'" in line:
   assert re.search(r'for\(integer k=0;k<64;k=k\+1\)operations\[k\]=0;',boottext);reason='constant_initialization_loop'
  elif "net 'operations' does not have a driver" in line:
   assert 'reg [13:0] operations[0:63];' in boottext and 'initial begin' in boottext
   reason='initialized_rom_proved_by_actual_map_boot'
  elif 'expression size 11 truncated to fit in target size 10' in line:
   assert 'video_service_modes.sv:' in line and 'HEIGHT=720' in modes and 'y<=y==HEIGHT-1 ? 0:y+1\'b1' in modes
   assert 'reg[10:0] x;reg[9:0] y;' in modes
   reason='bounded_720_row_counter_10_bits'
  elif line.startswith('[EFX-0200'):
   name=re.search(r'Removing redundant signal : (.*?)\. \(',line)[1]
   if name.removeprefix('protected_inst/') in protected_redundant:
    assert proof['protected_input_unchanged'] and proof['protected_boundary']['other_connections_equal']
    assert proof['protected_boundary']['ports']==1201 and proof['protected_boundary']['bits']==4128
    assert 'native_source.v:' in line
    # This is an optimizer's unused-signal notice inside identical encrypted
    # input IP, NOT proof of internal equivalence or of hardware behavior.
    reason='official_protected_internal_optimization_same_ip_and_external_ports'
   else:
    assert name in {'k[31:0]','b[31:0]','b_2[31:0]','b_3[31:0]','b_4[31:0]','b_5[31:0]','config_data[31:13]','config_data[8:4]','config_data[1:0]','commits[31:16]'},line
    if name.startswith('commits'):assert 'commits[15:0]' in health
    reason='static_loop_or_declared_unused_input_bits'
  elif line.startswith('[EFX-0657'):
   assert ("'frame' (48 bits)" in line and 'reg [15:0] frame[0:2]' in owner) or ("'rank' (6 bits)" in line and 'reg [1:0] rank[0:2]' in owner)
   reason='small_owner_array_mapped_to_logic'
  else:raise AssertionError(('unclassified_warning',line))
  counts[reason]+=1;new.append(dict(warning=line,reason=reason))
 evidence=[src,warn,oldwarn,board/'retained-map-audit.json',board/'frame_pair_owner.sv',board/'video_service_modes.sv',board/'video_service_health.sv',board/'cnn_preview_boot.sv',a.boot/'result.json',a.boot/'inputs.json',Path(__file__)]
 result=dict(status='ALL_CURRENT_SYNTHESIS_WARNINGS_CLASSIFIED_PASS',counts=dict(counts),primitive_ports=dict(ports),new_semantic_warnings=new,protected_open_bits=[12,13,14,15,32,33,36,37,38,39],map_sha256=key[1],evidence={str(f.resolve()):sha(f) for f in evidence},board_release=False,hardware_access=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],counts=result['counts'])))
if __name__=='__main__':main()
