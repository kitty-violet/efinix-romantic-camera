"""Read-only first-adapter-fault and DDR clock/reset probe on the proven CPU.

All differences are reversible, explicitly generated, and checked against the
frozen complete-transition2 source. Existing functional statements stay intact.
"""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from complete_control_transition import audit_delta as audit_parent
BASE=ROOT/'tmp/frame-execution/connected-board-complete-transition2'
CLOCK=HERE/'variants/board_fault_probe/bringup_clock_probe.sv'
PROBE_PORT='    output wire [127:0] adapter_diagnostic,\n'

def one(text,before,after):
 assert text.count(before)==1,(before,text.count(before))
 return text.replace(before,after)

def adapter(text):
 text=one(text,'    input wire clk, input wire rst_n,',PROBE_PORT+'    input wire clk, input wire rst_n,')
 observe=r'''
    // Independent observer: retain the PRECEDING cycle when either real
    // sticky error FF first asserts. Do not infer a cause from the OR alone.
    wire [31:0] probe_live={s_axi_wready,m_axi_arready,m_axi_awready,
        (wremaining==1),s_axi_bready,s_axi_rready,(m_axi_bid!=0),(m_axi_rid!=0),
        m_axi_bresp,m_axi_rresp,s_axi_wstrb,s_axi_wlast,m_axi_rlast,
        m_axi_bvalid,m_axi_rvalid,s_axi_wvalid,s_axi_awvalid,s_axi_arvalid,
        rst_n,werror,rerror,wstate,rstate};
    reg [31:0] probe_previous=0,probe_address=0;
    reg [23:0] probe_ticks=0;
    reg [95:0] probe_first=0;
    always @(posedge clk)begin
        if(!rst_n)begin
            probe_previous<=0;probe_address<=0;probe_ticks<=0;probe_first<=0;
        end else begin
            probe_ticks<=probe_ticks+1'b1;
            probe_previous<=probe_live;
            probe_address<=s_axi_arvalid ? s_axi_araddr :
                s_axi_awvalid ? s_axi_awaddr : rstate!=R_IDLE ? raddr : waddr;
            if(!probe_first[95] && (rerror || werror))
                probe_first<={1'b1,5'd0,werror,rerror,probe_ticks,probe_address,probe_previous};
        end
    end
    assign adapter_diagnostic={probe_live,probe_first};
'''
 return one(text,'endmodule',observe+'\nendmodule')

def transform(rel,text):
 if rel=='assembly/sapphire_axi32_to_axi128.sv':return adapter(text)
 if rel=='assembly/ti60_ddr_four_master_fabric.sv':
  text=one(text,'    input wire clk, input wire rst_n,',PROBE_PORT+'    input wire clk, input wire rst_n,')
  return one(text,') cpu_width_adapter (',') cpu_width_adapter (\n        .adapter_diagnostic(adapter_diagnostic),')
 if rel in ('assembly/cnn_rgb_frame_pipeline.sv','assembly/cnn_video_pipeline.sv'):
  # Add a port to the real module, then propagate only the diagnostic output.
  marker=') (' if ') (' in text[:2500] else ')('
  text=one(text,marker,marker+'\n'+PROBE_PORT)
  needle=' .read_busy(),.write_busy(),.protocol_error(fabric_error)' if 'rgb_frame' in rel else '.clk(process_clk),.reset(!process_rst_n),.capture_abort'
  return one(text,needle,'.adapter_diagnostic(adapter_diagnostic),\n '+needle)
 if rel=='assembly/EfxSapphireSoc.v':
  text=text.replace('[383:0] diagnostic_payload','[703:0] diagnostic_payload').replace('[383:0] observation_payload','[703:0] observation_payload')
  text=one(text,'  input wire [31:0] diagnostic_board_status,','  input wire [319:0] diagnostic_extra,\n  input wire [31:0] diagnostic_board_status,')
  return one(text,'assign observation_payload={ddr_lock_drops','assign observation_payload={diagnostic_extra,ddr_lock_drops')
 if rel=='assembly/jtag_bringup_snapshot.sv':
  return text.replace('[383:0]','[703:0]').replace('[447:0]','[767:0]').replace('[447:1]','[767:1]').replace("8'd3","8'd4").replace("32'h4f465033","32'h4f465034")
 if rel=='rtl/ti60f225_oob_top.v':
  block='''wire [127:0] adapter_diagnostic;
wire [191:0] ddr_clock_diagnostic;
bringup_clock_probe clock_probe(.clk(process_clk_150),.core_clk(ddr_core_clk200),
 .tdqss_clk(sdram_clk),.tac_clk(rx_cal_clk),.twd_clk(tx_cal_clk_90edge),
 .system_reset_n(w_arstn),.ddr_reset_n(ddr_reset_n),.ddr_cke(ddr_cke[0]),
 .probe(ddr_clock_diagnostic));
'''
  text=one(text,'cnn_fixed_control_soc cpu(',block+'cnn_fixed_control_soc cpu(')
  text=one(text,' .diagnostic_board_status(bringup_ddr_sync),',' .diagnostic_extra({ddr_clock_diagnostic,adapter_diagnostic}),\n .diagnostic_board_status(bringup_ddr_sync),')
  return one(text,') video_processing_pipeline (',') video_processing_pipeline (\n .adapter_diagnostic(adapter_diagnostic),')
 if rel=='ti60f225_oob.xml':
  before='        <efx:top_module name="ti60f225_oob_top" />\n'
  return one(text,before,before+'        <efx:design_file name="assembly/bringup_clock_probe.sv" version="default" library="default" />\n')
 return text

CHANGED={'assembly/sapphire_axi32_to_axi128.sv','assembly/ti60_ddr_four_master_fabric.sv',
 'assembly/cnn_rgb_frame_pipeline.sv','assembly/cnn_video_pipeline.sv','assembly/EfxSapphireSoc.v',
 'assembly/jtag_bringup_snapshot.sv','rtl/ti60f225_oob_top.v','ti60f225_oob.xml'}
def audit_delta(board):
 audit_parent(BASE)
 m=json.loads((board/'inputs.json').read_text())
 assert m['parent']==str(BASE) and m['parent_inputs_sha256']==sha(BASE/'inputs.json')
 assert m['fault_probe_transform_sha256']==sha(__file__)
 parent=json.loads((BASE/'inputs.json').read_text());expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}
 seen=set();changed=set()
 for row in m['inputs']:
  f=Path(row['snapshot']);assert f.is_relative_to(board) and sha(f)==row['sha256']
  rel=f.relative_to(board).as_posix();seen.add(rel)
  if rel=='assembly/bringup_clock_probe.sv':assert sha(f)==sha(CLOCK);continue
  if rel in CHANGED:
   assert f.read_text()==transform(rel,(BASE/rel).read_text());changed.add(rel)
  else:assert sha(f)==sha(BASE/rel),rel
 assert seen==expected|{'assembly/bringup_clock_probe.sv'} and changed==CHANGED
 return dict(status='FAULT_PROBE_ONLY_FULL_BOARD_DELTA',changed=sorted(changed),
  pins_clocks_constraints_ddr_unchanged=True,existing_functional_statements_unchanged=True,
  payload_bits=704,header_bits=64,serial_bits_with_wrapper=770,
  full_video_a_chain_preserved=True,hardware_access=False,board_release=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
 board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp') and not board.exists()
 audit_parent(BASE);m=json.loads((BASE/'inputs.json').read_text());board.mkdir()
 for row in m['inputs']:
  src=Path(row['snapshot']);assert sha(src)==row['sha256'];rel=src.relative_to(BASE).as_posix()
  dst=board/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if rel in CHANGED:dst.write_text(transform(rel,src.read_text()),encoding='utf8')
  else:shutil.copy2(src,dst)
  row.update(snapshot=str(dst),sha256=sha(dst))
 dst=board/'assembly/bringup_clock_probe.sv';shutil.copy2(CLOCK,dst)
 m['inputs'].append(dict(source=str(CLOCK),snapshot=str(dst),sha256=sha(dst)))
 m.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),
  fault_probe_transform_sha256=sha(__file__),warning_reference_base=str(BASE),
  hardware_access=False,board_download_allowed=False)
 (board/'inputs.json').write_text(json.dumps(m,indent=2))
 (board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2))
 print('PREPARED_FAULT_PROBE_ONLY_NOT_RELEASED',board)
if __name__=='__main__':main()
