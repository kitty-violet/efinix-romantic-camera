"""Short actual CSI -> boundary -> RAW writer diagnostic, behavioral AXI sink.

No replacement inside the camera path. This isolates the ingress from DDR/CPU;
it is a diagnostic, never a whole-board measurement or PHY test.
"""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
 base=ROOT/'tmp/frame-execution/board-ddr516-frame6'
 names=['camera_axi_write_path.sv','axi_frame_write_dma.sv','raw32_frame_cdc.sv','raw32_frame_packer.sv','guarded_async_fifo.sv','video_frame_boundary_detector.sv','DC_FIFO.v']
 sources=[next(base.rglob(n)) for n in names]
 sources += [base/'csi_rx_controller_sim.sv',HERE/'board_integration/camera_raw10_fixture.sv',EF/'ipm/ip/efx_csi2/testbench/soft_dphy/rx/efx_csi2_tx_modelsim.sv',*sorted((EF/'sim_models/verilog').glob('*.v'))]
 records=[]
 for src in sources:
  dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 tb='''`timescale 1ns/1ps
module tb;
reg clk=0,bc=0,pc=0,ac=0,rst=0;
always #5 clk=~clk;always #7.142857 bc=~bc;always #10 pc=~pc;always #3.333333 ac=~ac;
wire cp,cn;wire [3:0]dp,dn,he;wire [7:0]d0,d1,d2,d3;
wire [63:0]data;wire valid,hs,vs;wire [31:0]debug;
camera_raw10_fixture #(.LINES(20),.FRAMES(2),.FRAME_TICKS(35000)) cam(.clk(clk),.byte_clk(bc),.pixel_clk(pc),.reset_n(rst),.start(1'b1),
 .lp_cp(cp),.lp_cn(cn),.lp_dp(dp),.lp_dn(dn),.d0(d0),.d1(d1),.d2(d2),.d3(d3),.hs_enable(he));
'''
 header=(base/'csi_rx_controller_sim.sv').read_text().split('module csi_rx_controller',1)[1].split(');',1)[0]
 connections={'reset_n':'rst','reset_byte_HS_n':'rst','reset_pixel_n':'rst','clk':'clk','clk_byte_HS':'bc','clk_pixel':'pc','axi_clk':'clk','axi_reset_n':'rst',
 'Rx_LP_CLK_P':'cp','Rx_LP_CLK_N':'cn','Rx_LP_D_P':'dp','Rx_LP_D_N':'dn',**{f'Rx_HS_D_{i}':f'd{i}' for i in range(4)},
 'fifo_rd_empty':"4'd0",'pixel_data':'data','pixel_data_valid':'valid','hsync_vc0':'hs','vsync_vc0':'vs','mipi_debug_out':'debug'}
 ports=['.'+name+'('+connections.get(name,"'0" if direction=='input' else '')+')' for direction,width,name in re.findall(r'\b(input|output)\s+(\[[^]]+\])?\s*(\w+)',header)]
 tb+='csi_rx_controller rx('+','.join(ports)+');\n'
 tb+='''wire fs,fe,pv,be;
video_frame_boundary_detector boundary(.clk(pc),.rst_n(rst),.vsync(vs),.data_enable(valid&hs),.frame_start(fs),.frame_end(fe),.pixel_valid(pv),.boundary_error(be));
reg m_axi_awready=0,m_axi_wready=0,m_axi_bvalid=0;
reg [3:0]m_axi_bid=0;reg [1:0]m_axi_bresp=0;
'''
 writer=(base/'assembly/camera_axi_write_path.sv').read_text().split(') (',1)[1].split(');',1)[0]
 mapping={'reset':'!rst','camera_clk':'pc','camera_frame_start':'fs','camera_frame_end':'fe','camera_pixel_valid':'pv',
 'camera_pixel_word':'{data[39:32],data[29:22],data[19:12],data[9:2]}','axi_clk':'ac','frame_base_addr':"32'h06000000"}
 parameters={'AXI_ADDR_WIDTH':'32','AXI_DATA_WIDTH':'128','AXI_ID_WIDTH':'4','FRAME_ID_WIDTH':'16','FIFO_DEPTH':'256'}
 ports=[]
 for direction,width,name in re.findall(r'\b(input|output)\s+(?:wire|reg)\s+(\[[^]]+\]\s*)?(\w+)',writer):
  if direction=='output':
   for k,v in parameters.items():width=width.replace(k,v)
   tb+='wire '+width+name+';\n'
  ports.append('.'+name+'('+mapping.get(name,name)+')')
 tb+='camera_axi_write_path #(.WORDS_PER_FRAME(9600),.FIFO_DEPTH(256)) dut('+','.join(ports)+');\n'
 tb+='''integer aws=0,ws=0,bs=0,left=0;reg active=0;reg [15:0]lfsr=16'hbc39;
always @(posedge ac)if(rst)begin
 lfsr<={lfsr[14:0],lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
 m_axi_awready<=!active&&!m_axi_bvalid&&lfsr[0];m_axi_wready<=active&&lfsr[1];
 if(m_axi_awvalid&&m_axi_awready)begin
  aws=aws+1;left=m_axi_awlen+1;active<=1;
  if(aws<4)$display("RAW_AW t=%0t addr=%h len=%0d",$time,m_axi_awaddr,left);
 end
 if(m_axi_wvalid&&m_axi_wready)begin
  if(m_axi_wdata!=={16{8'h80}} || m_axi_wstrb!==16'hffff || m_axi_wlast!==(left==1))$fatal(1,"RAW W data/last");
  ws=ws+1;left=left-1;if(m_axi_wlast)begin active<=0;m_axi_bvalid<=1;end
 end
 if(m_axi_bvalid&&m_axi_bready)begin bs=bs+1;m_axi_bvalid<=0;end
 if(be||dma_config_error||dma_protocol_error||frame_length_error||packer_overflow_error||packer_protocol_error||fifo_protocol_error||stream_protocol_error||frame_failed)begin
  $display("RAW_FAULT be=%b cfg=%b dma=%b length=%b overflow=%b pack=%b fifo=%b stream=%b fail=%b aws=%0d ws=%0d bs=%0d state=%0d depth=%0d words=%0d",
   be,dma_config_error,dma_protocol_error,frame_length_error,packer_overflow_error,packer_protocol_error,fifo_protocol_error,stream_protocol_error,frame_failed,aws,ws,bs,dut.writer.state,dut.axi_fifo_count,dut.captured_words);
  $fatal(1,"CSI RAW ingress error");
 end
 if(frame_done)begin
  if(ws!=2400 || bs!=19)$fatal(1,"count ws=%0d bs=%0d",ws,bs);
  $display("PASS CSI_RAW_20_LINES words=9600 writes=%0d bursts=%0d debug=%h",ws,bs,debug);$finish;
 end
end
initial begin #200;rst=1;#2000000;$fatal(1,"timeout aws=%0d ws=%0d bs=%0d words=%0d depth=%0d",aws,ws,bs,dut.captured_words,dut.axi_fifo_count);end
endmodule
'''
 (root/'tb.sv').write_text(tb);records.append(dict(source=str(Path(__file__)),snapshot=str(root/'tb.sv'),sha256=sha(root/'tb.sv')))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2))
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
 for name,cmd in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv',*[r['snapshot'] for r in records]]),('simulation',[sim/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])]:
  with (root/(name+'.log')).open('wb')as f:subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
 body=(root/'simulation.log').read_text();passed='PASS CSI_RAW_20_LINES' in body and '** Fatal' not in body
 result=dict(status='PASS' if passed else 'FAIL',seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),whole_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert passed
if __name__=='__main__':main()
