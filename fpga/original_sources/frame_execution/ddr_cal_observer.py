"""Exact observation-only delta from the currently downloaded complete design."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
BASE=ROOT/'tmp/frame-execution/connected-board-first-fault1'
MODULE=HERE/'variants/board_ddr_observer/ddr_cal_observer.sv'

def transform_soc(text):
 old=" wire apb_hit=address[31:7]==25'(32'hf8100000>>7) && address[6:0]<=7'h6c;"
 new=" wire apb_hit=address[31:8]==24'(32'hf8100000>>8) && address[7:0]<=8'h94;"
 assert text.count(old)==1
 return text.replace(old,new)

def transform_apb(text):
 old=' input wire [31:0] bringup_pipeline_status,bringup_ddr_status,'
 assert text.count(old)==1
 text=text.replace(old,old+'\n input wire [31:0] ddr_observation_data,')
 old="   16'h006c:prdata=bringup_ddr_status;"
 assert text.count(old)==1
 keys=','.join(f"16'h{x:04x}" for x in range(0x70,0x98,4))
 return text.replace(old,old+'\n   '+keys+':prdata=ddr_observation_data;')

def transform_top(text):
 old=' .bringup_pipeline_status(bringup_pipeline_status),.bringup_ddr_status(bringup_ddr_sync),'
 assert text.count(old)==1
 text=text.replace(old,old+'\n .ddr_observation_data(ddr_observation_data),')
 old='cnn_apb_control control(.clk(process_clk_150),.reset(!w_arstn),'
 assert text.count(old)==1
 # Read address is the exact existing APB control address, recovered below.
 import re
 control=text[text.index(old):];control=control[:control.index(');')]
 addr=re.search(r'\.paddr\(([^)]+)\)',control)[1]
 instance='''wire [31:0] ddr_observation_data;
ddr_cal_observer ddr_observer(.clk(process_clk_150),.core_clk(ddr_core_clk200),
 .tdqss_clk(sdram_clk),.tac_clk(rx_cal_clk),.twd_clk(tx_cal_clk_90edge),
 .system_reset_n(w_arstn),.ddr_reset_n(ddr_reset_n),.ddr_cke(ddr_cke[0]),
 .shift_ena(pll_shift_ena),.shift_code(pll_shift),.shift_sel(pll_shift_sel),
 .read_address(@@ADDRESS@@),.read_data(ddr_observation_data));
'''.replace('@@ADDRESS@@',addr)
 return text.replace(old,instance+old)+'\n'+MODULE.read_text(encoding='utf-8')

def audit_delta(board):
 board=Path(board);meta=json.loads((board/'inputs.json').read_text());changed=[]
 for rec in meta['inputs']:
  p=Path(rec['snapshot']);assert p.is_relative_to(board) and sha(p)==rec['sha256']
  rel=p.relative_to(board)
  if sha(p)!=sha(BASE/rel):changed.append(rel.as_posix())
 assert sorted(changed)==['assembly/EfxSapphireSoc.v','assembly/cnn_apb_control.sv','rtl/ti60f225_oob_top.v'],changed
 assert (board/'assembly/EfxSapphireSoc.v').read_text()==transform_soc((BASE/'assembly/EfxSapphireSoc.v').read_text())
 assert (board/'assembly/cnn_apb_control.sv').read_text()==transform_apb((BASE/'assembly/cnn_apb_control.sv').read_text())
 assert (board/'rtl/ti60f225_oob_top.v').read_text()==transform_top((BASE/'rtl/ti60f225_oob_top.v').read_text())
 for rel in ('ti60f225_oob.xml','ti60f225_oob.sdc','ti60f225_oob.peri.xml','assembly/jtag_bringup_snapshot.sv','ddr516_v4_controller.sv'):
  assert sha(board/rel)==sha(BASE/rel),rel
 return dict(status='DDR_PUBLIC_PORT_OBSERVATION_ONLY_DELTA',changed=changed,
             source_functional_logic_unchanged_except_diagnostic_decode=True,only_new_apb_read_addresses='0070..0094',
             pins_clocks_sdc_unchanged=True,clock_mhz=150,full_video_a_chain_preserved=True,board_release=False)
