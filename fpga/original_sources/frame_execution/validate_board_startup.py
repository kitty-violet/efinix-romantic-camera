"""Actual board startup with ideal digital clock/I/O envelope and Micron model.

No forced calibration flags, no preloaded external memory, no CPU substitution.
This prerequisite is separate from a complete camera-to-screen measurement.
"""
import argparse,json,os,re,shutil,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_full_board_simulation import prepare

def pin_trace():
 # Observe public I/O and the open Micron model only. No protected PHY probes,
 # timing overrides, signal forces, or changes to commands/calibration.
 return r'''
initial begin
 #700000;
 $dumpfile("ddr_pin_trace.vcd");
 $dumpvars(0,sdram_clk,tx_cal_clk,tx_cal_clk_90edge,rx_cal_clk,phase,
  ddr_odt,ddr_addr,ddr_ba,ddr_cs_n,ddr_ras_n,ddr_cas_n,ddr_we_n,
  ddr_dqs_oe,ddr_dqs_out_hi,ddr_dqs_out_lo,ddr_dq_in_hi,ddr_dq_in_lo,
  io_model.ck,io_model.dqs,io_model.dq,io_model.odt,io_model.addr,
  io_model.ba,io_model.cs_n,io_model.ras_n,io_model.cas_n,io_model.we_n,
  io_model.memory.write_levelization,io_model.memory.odt_state,
  io_model.memory.odt_pipeline,io_model.memory.odt_cntr);
end
real odt_fall_ns=0;
always @(negedge io_model.odt)odt_fall_ns=$realtime;
always @(posedge io_model.ck)begin
 if(io_model.rst_n && io_model.cke && !io_model.cs_n && {io_model.ras_n,io_model.cas_n,io_model.we_n}==3'b000)begin
  $display("DDR_MRS t_ns=%0.3f mr=%0d addr=%h odt=%b odt_state=%b odt_pipe=%h since_odt_fall_ns=%0.3f",
   $realtime,io_model.ba,io_model.addr,io_model.odt,io_model.memory.odt_state,io_model.memory.odt_pipeline,$realtime-odt_fall_ns);
  $fflush();
 end
 if($realtime>830000 && !io_model.cs_n && {io_model.ras_n,io_model.cas_n}==2'b10)begin
  $display("DDR_COMMAND t_ns=%0.3f we_n=%b odt=%b odt_state=%b addr=%h",
   $realtime,io_model.we_n,io_model.odt,io_model.memory.odt_state,io_model.addr);$fflush();
 end
end
always @(io_model.odt or io_model.memory.odt_state)if($realtime>700000)begin
 $display("DDR_ODT t_ns=%0.3f pin=%b effective=%b write_level=%b",
  $realtime,io_model.odt,io_model.memory.odt_state,io_model.memory.write_levelization);$fflush();
end
integer leveling_pulses=0;
always @(posedge io_model.dqs[0])if(io_model.memory.write_levelization && leveling_pulses<80)begin
 leveling_pulses=leveling_pulses+1;
 $display("DDR_LEVEL_DQS t_ns=%0.3f phase=%0d ck=%b dq=%h input_hi=%h input_lo=%h",
  $realtime,phase[3],io_model.ck,io_model.dq,ddr_dq_in_hi,ddr_dq_in_lo);$fflush();
end
'''

def official_pll_tb(tb):
 # Exact DDR3_PLL parameters/ports exported by Efinity from V4 peri.xml.
 # Keep the independent 100 MHz core clock as in the board's MIPI_TX_PLL.
 for name,idx in {'tx_cal_clk_90edge':0,'rx_cal_clk':2,'tx_cal_clk':3}.items():
  tb=tb.replace(f'assign #(phase[{idx}]*0.15625) {name}=sdram_clk;', '')
 tb=tb.replace('reg  sdram_clk=0;', 'wire sdram_clk;')
 tb=tb.replace('always #1.250000000 sdram_clk=~sdram_clk;', '')
 assert 'sdram_clk=~sdram_clk' not in tb and 'reg  sdram_clk' not in tb
 code='''
reg reference_25=0;always #20 reference_25=~reference_25;
wire feedback_100,ddr_pll_locked;
EFX_PLL_V3 #(.N(1),.M(4),.O(1),.CLKOUT0_DIV(8),.CLKOUT1_DIV(8),
 .CLKOUT2_DIV(8),.CLKOUT3_DIV(8),.CLKOUT4_DIV(32),.FEEDBACK_CLK("CLK4"),
 .FEEDBACK_MODE("CORE"),.REFCLK_FREQ(25.0)) official_pll(
 .CLKIN({reference_25,3'bzzz}),.CLKSEL(2'b11),.RSTN(1'b1),.FBK(feedback_100),
 .CLKOUT0(tx_cal_clk_90edge),.CLKOUT1(sdram_clk),.CLKOUT2(rx_cal_clk),
 .CLKOUT3(tx_cal_clk),.CLKOUT4(feedback_100),.LOCKED(ddr_pll_locked),
 .SHIFT(pll_shift),.SHIFT_SEL(pll_shift_sel),.SHIFT_ENA(pll_shift_ena),.DESKEWED());
'''
 return tb.rsplit('endmodule',1)[0]+code+'\nendmodule\n'

def make_ddr_tb(root):
 # Isolated diagnosis uses the identical DDR RTL/pin/device models, not a
 # substitute for the still-required complete-board frame measurement.
 source=(root/'rtl/ddr3_controller/ddr3_top.v').read_text()
 header=source.split('// Clock and reset ports',1)[1].split(');',1)[0]
 header=re.sub(r'//[^\n]*','',header)
 ports=re.findall(r'\b(input|output)\s+(\[[^]]+\])?\s*(\w+)\s*[,\n]',header+'\n')
 assert len(ports)==86,len(ports)
 widths={m.group(1):m.group(2) for m in re.finditer(r'`define\s+(\w+)\s+(\d+)',(root/'rtl/ddr3_controller/ddr3_parameter.vh').read_text())}
 widths['RANK_RATIO']='1'
 decl=['localparam '+','.join(k+'='+v for k,v in widths.items())+';','reg [2:0] phase[0:4];']
 dynamic={'tx_cal_clk_90edge':0,'rx_cal_clk':2,'tx_cal_clk':3}
 clocks={'axi_clk':150,'core_clk':100,'sdram_clk':400}
 for direction,width,name in ports:
  wire=direction=='output' or name in dynamic or name.startswith(('ddr_dq_in','ddr_dqs_in'))
  decl.append(('wire ' if wire else 'reg ')+(width or '')+' '+name+(';' if wire else '=0;'))
  if name in clocks:decl.append(f'always #{500/clocks[name]:.9f} {name}=~{name};')
 for name,idx in dynamic.items():decl.append(f'assign #(phase[{idx}]*0.15625) {name}=sdram_clk;')
 full=make_tb(root/'rtl/ti60f225_oob_top.v')
 io=full.split('board_ddr_io_model io_model(',1)[1].split(');',1)[0]
 return '''`timescale 1ns/1ps
module tb_board_startup;
'''+ '\n'.join(decl)+'\nddr3_top dut('+','.join(f'.{n}({n})' for _,_,n in ports)+');\nboard_ddr_io_model io_model('+io+''');
initial begin for(integer k=0;k<5;k=k+1)phase[k]=0;
 #200;rstn=1;
end
always @(posedge pll_shift_ena)begin
 for(integer k=0;k<5;k=k+1)if(pll_shift_sel[k])phase[k]<=pll_shift;
 $display("PLL_SHIFT t=%0t sel=%b value=%0d",$time,pll_shift_sel,pll_shift);$fflush();
end
always @(init_cur_state)begin
 $display("DDR_STATE t=%0t state=%h dqcheck=%h dqscheck=%h",$time,init_cur_state,wrlvl_dq_check,rd_level_dqs_check);$fflush();
end
integer cycles=0;reg cal_seen=0;
always @(posedge core_clk)begin
 cycles<=cycles+1;
 if(cycles%10000==0)begin $display("BOARD_START cycles=%0d cal=%b state=%h debug=%h",cycles,cal_done,init_cur_state,ddr_debug_port);$fflush();end
 if(cal_done===1'b1 && !cal_seen)begin cal_seen=1;$display("DDR_CALIBRATION_FLAG cycles=%0d t=%0t",cycles,$time);$fflush();end
 if(cycles>150000)$fatal(1,"DDR calibration unit timeout");
end
function [127:0] pattern(input integer index);
 for(integer b=0;b<16;b=b+1)pattern[8*b+:8]=(index*17+b*13)^8'h6d;
endfunction
integer pin_events=0;
always @(posedge io_model.dqs[0])if(cal_seen && io_model.dqs_en[0] && pin_events<12)begin
 pin_events=pin_events+1;
 $display("DDR_PIN t=%0.3f ck=%b tx_phase=%0d dq_phase=%0d last_ck_ps=%0d dqs_high=%b dqs_low=%b",$realtime,io_model.ck,phase[3],phase[0],io_model.memory.tm_ck_pos,ddr_dqs_out_hi,ddr_dqs_out_lo);$fflush();
end
initial begin
 wait(cal_done===1'b1);repeat(1000)@(posedge axi_clk);
 @(negedge axi_clk);s_axi_awaddr=32'h10000;s_axi_awlen=15;s_axi_awsize=4;s_axi_awburst=1;s_axi_awid=3;s_axi_awvalid=1;
 do @(posedge axi_clk);while(!s_axi_awready);
 @(negedge axi_clk);s_axi_awvalid=0;
 for(integer k=0;k<16;k=k+1)begin
  s_axi_wvalid=1;s_axi_wdata=pattern(k);s_axi_wstrb='1;s_axi_wlast=(k==15);
  do @(posedge axi_clk);while(!s_axi_wready);
  @(negedge axi_clk);
 end
 s_axi_wvalid=0;s_axi_wlast=0;s_axi_bready=1;
 do @(posedge axi_clk);while(!s_axi_bvalid);
 if(s_axi_bresp!==0 || s_axi_bid!==3)$fatal(1,"AXI write response");
 @(negedge axi_clk);s_axi_bready=0;
 s_axi_araddr=32'h10000;s_axi_arlen=15;s_axi_arsize=4;s_axi_arburst=1;s_axi_arid=5;s_axi_arvalid=1;
 do @(posedge axi_clk);while(!s_axi_arready);
 @(negedge axi_clk);s_axi_arvalid=0;s_axi_rready=1;
 for(integer k=0;k<16;k=k+1)begin
  do @(posedge axi_clk);while(!s_axi_rvalid);
  if(s_axi_rdata!==pattern(k) || s_axi_rresp!==0 || s_axi_rid!==5 || s_axi_rlast!==(k==15))
   $fatal(1,"AXI readback beat=%0d got=%h expected=%h resp=%h id=%h last=%b",k,s_axi_rdata,pattern(k),s_axi_rresp,s_axi_rid,s_axi_rlast);
  @(negedge axi_clk);
 end
 s_axi_rready=0;$display("PASS DDR_UNIT_CALIBRATION cycles=%0d readback_beats=16 t=%0t",cycles,$time);$fflush();$finish;
end
endmodule
'''

def make_tb(top,ddr516=False):
 header=top.read_text().split(');',1)[0]
 ports=re.findall(r'\b(input|output|inout)\s+(?:wire\s+)?(\[[^]]+\])?\s*(\w+)',header)
 decl=['localparam CKE_WIDTH=1,ROW_WIDTH=14,BANK_WIDTH=3,CS_WIDTH=1,RANK_RATIO=1,DQS_WIDTH=2,DQ_WIDTH=16,DM_WIDTH=2,ODT_WIDTH=1;'];links=[]
 freq={'process_clk_150':150,'core_clk':100,'sdram_clk':400,'CLK_25M':25,'CLK_5M':5,'i_mipi_rx_pclk':50,'vid_clk_dvi2':74.25,'hdmi_tx_slow_clk':74.25,'i_cam_ck_CLKOUT':187.5}
 if ddr516:freq.pop('sdram_clk')
 dyn={'tx_cal_clk_90edge':0,'rx_cal_clk':2,'tx_cal_clk':3};iorx={'ddr_dq_in_hi','ddr_dq_in_lo','ddr_dqs_in_hi','ddr_dqs_in_lo'}
 for direction,width,name in ports:
  links.append(f'.{name}({name})')
  kind='wire' if direction=='output' or name in iorx or name in dyn or (ddr516 and name in ('sdram_clk','ddr_core_clk200','pll_locked')) else 'reg'
  high=bool(re.search(r'LP_[PN]_IN$|FIFO_EMPTY$|^io_cam_s[cd][al]_IN$',name))
  decl.append(f'{kind} {width} {name}'+("=1" if high else "=0" if kind=='reg' else '')+';')
  if name in freq:decl.append(f'always #{500/freq[name]:.9f} {name}=~{name};')
 if not ddr516:
  for name,idx in dyn.items():decl.append(f'assign #(phase[{idx}]*0.15625) {name}=sdram_clk;')
 else:
  decl.append('''wire tx_cal_clk=sdram_clk;
wire ddr_feedback50;
EFX_PLL_V3_FAST_PROTOCOL #(.N(1),.M(2),.O(2),.CLKOUT0_DIV(6),.CLKOUT1_DIV(6),
 .CLKOUT2_DIV(6),.CLKOUT3_DIV(12),.CLKOUT4_DIV(48),.CLKOUT0_PHASE_STEP(3),.CLKOUT2_DYNPHASE_EN(1),
 .FEEDBACK_CLK("CLK4"),.FEEDBACK_MODE("CORE"),.REFCLK_FREQ(25.0)) ddr_pll(
 .CLKIN({CLK_25M,3'bzzz}),.CLKSEL(2'd3),.RSTN(DDR3_PLL_RSTN),.FBK(ddr_feedback50),
 .SHIFT(pll_shift),.SHIFT_SEL(pll_shift_sel),.SHIFT_ENA(pll_shift_ena),
 .CLKOUT0(tx_cal_clk_90edge),.CLKOUT1(sdram_clk),.CLKOUT2(rx_cal_clk),.CLKOUT3(ddr_core_clk200),.CLKOUT4(ddr_feedback50),.LOCKED(pll_locked),.DESKEWED());''')
 ddr_header=(HERE/'board_integration/ddr_periphery_model.sv').read_text().split(');',1)[0]
 io_names=[]
 for m in re.finditer(r'\b(?:input|output)\s+(?:\[[^]]+\]\s*)?([^;()]+?)(?=,\s*(?:input|output)|\Z)',ddr_header.split('module board_ddr_io_model(',1)[1],re.S):
  io_names.extend(n.strip() for n in m.group(1).replace('\n','').split(','))
 assert len(io_names)==27 and len(set(io_names))==27,io_names
 tb='''`timescale 1ns/1ps
module tb_board_startup;
reg [2:0] phase[0:4];
'''+ '\n'.join(decl)+'\nti60f225_oob_top dut('+','.join(links)+');\nboard_ddr_io_model io_model('+','.join(f'.{n}({n})' for n in io_names)+''');
initial begin for(integer k=0;k<5;k=k+1)phase[k]=0;
 #100;i_arstn=1;#100;i_pll_locked=1;pll_locked=1;user_pll_locked=1;
end
always @(posedge pll_shift_ena)begin
 for(integer k=0;k<5;k=k+1)if(pll_shift_sel[k])phase[k]<=pll_shift;
 $display("PLL_SHIFT t=%0t sel=%b value=%0d",$time,pll_shift_sel,pll_shift);$fflush();
end
integer cycles=0;
always @(posedge core_clk)begin
 cycles<=cycles+1;
 if(cycles%10000==0)begin $display("BOARD_START cycles=%0d cal=%b cpu_enable=%b",cycles,dut.cal_done,dut.control_enable);$fflush();end
 if(dut.cal_done===1'b1)begin $display("PASS BOARD_CALIBRATION cycles=%0d t=%0t",cycles,$time);$fflush();$finish;end
 if(cycles>1000000)$fatal(1,"board calibration timeout");
end
endmodule
'''
 if ddr516:tb=tb.replace('i_pll_locked=1;pll_locked=1;user_pll_locked=1;','i_pll_locked=1;user_pll_locked=1;')
 return tb

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--boot-image',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--unit-ddr',action='store_true');p.add_argument('--trace-pins',action='store_true');p.add_argument('--legacy-fifo',type=Path);p.add_argument('--official-io',action='store_true');p.add_argument('--official-pll',action='store_true');a=p.parse_args()
 assert not a.trace_pins or a.unit_ddr,'Pin trace is restricted to the isolated same-source DDR fixture'
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 files,inc,manifest=prepare(root,a.base.resolve(),a.boot_image.resolve())
 if a.legacy_fifo:
  assert a.unit_ddr,'FIFO rollback is an isolated diagnostic, not a board design change'
  src=a.legacy_fifo.resolve()
  assert '// IP Version: 1.3' in src.read_text() and 'module efx_fifo_wrapper # (' in src.read_text()
  removed=[f for f in files if 'efx_fifo_v8.1' in f.parts]
  assert removed,'Expected the frozen production IP 8.1 FIFO'
  files=[f for f in files if f not in removed]
  dst=root/'diagnostic_legacy_fifo.v';shutil.copy2(src,dst);files.append(dst)
  manifest['inputs'].append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  manifest['fifo_diagnostic']=dict(version='1.3',source=str(src),excluded_from_compile=[str(f) for f in removed])
 package=EF/'ipm/ip/efx_ddr3_soft_controller/testbench'
 # The package's generic header selects sg25E. This fixture explicitly selects
 # the existing 2 Gb sg125 x16 reference; never use a conflicting default bin.
 (root/'ddr3_device_ID.vh').write_text('`define sg125\n')
 more=[package/'ddr3.v',package/'efx_iddr.v',package/'efx_oddr.v',HERE/'board_integration/ddr_periphery_model.sv',ROOT/'tmp/frame-execution/full-board-vendor-probe3/2048Mb_ddr3_parameters.vh']
 if a.official_io:
  assert a.unit_ddr
  more=[f for f in more if f.name!='ddr_periphery_model.sv']
  more += [HERE/'board_integration/ddr_periphery_official_model.sv',EF/'pt/sim_models/verilog/EFX_GPIO_V3.v']
 manifest['official_io']=a.official_io
 if a.official_pll:
  assert a.unit_ddr
  more.append(EF/'pt/sim_models/verilog/EFX_PLL_V3.v')
 manifest['official_pll']=a.official_pll
 for src in more:
  dst=root/src.name;shutil.copy2(src,dst);manifest['inputs'].append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  if src.suffix in ('.v','.sv'):files.append(dst)
 peri=a.base/'ti60f225_oob.peri.xml';manifest['periphery_sha256']=sha(peri)
 tree=ET.parse(peri).getroot()
 # Check the actual register/clock mapping before constructing ideal pins.
 for e in tree.iter():
  if e.tag.endswith('comp_gpio') and e.get('name','').startswith('ddr_'):
   for prop in e:
    attrs=prop.attrib
    if prop.tag.endswith('output_config'):
     if attrs['name']=='ddr_reset_n':
      assert attrs['register_option']=='none' and attrs['clock_name']=='';continue
     assert attrs['register_option']=='register' and attrs['is_clock_inverted']=='false',attrs
     assert attrs['clock_name'] in ('sdram_clk','tx_cal_clk','tx_cal_clk_90edge'),attrs
    if prop.tag.endswith('input_config'):assert attrs['clock_name']=='rx_cal_clk' and attrs['ddio_type']=='resync',attrs
 tb=make_ddr_tb(root) if a.unit_ddr else make_tb(root/'rtl/ti60f225_oob_top.v')
 if a.trace_pins:tb=tb.rsplit('endmodule',1)[0]+pin_trace()+'\nendmodule\n'
 if a.official_pll:tb=official_pll_tb(tb)
 (root/'tb_board_startup.sv').write_text(tb);files.append(root/'tb_board_startup.sv')
 manifest.update(scope='Full RTL startup/calibration only, ideal digital I/O; no electrical qualification',tb_sha256=sha(files[-1]),source_files=[str(f) for f in files])
 manifest['device_selection_header_sha256']=sha(root/'ddr3_device_ID.vh')
 manifest['unit_ddr']=a.unit_ddr
 manifest['pin_trace']=a.trace_pins
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2));sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 commands=[('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16',*[('+incdir+'+str(d)) for d in inc],*files])]
 if not a.prepare_only:commands.append(('simulation',[sim/'vsim.exe','-c','work.tb_board_startup','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0']))
 for stage,cmd in commands:
  start=time.monotonic()
  with (root/(stage+'.log')).open('wb') as log:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
  body=(root/(stage+'.log')).read_text(errors='replace')
  if r.returncode or re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',body):
   result=dict(status='FAIL',stage=stage,returncode=r.returncode,tail=body[-2500:],full_board_frame_ms=None);(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
 status='PREPARED_COMPILE_PASS' if a.prepare_only else 'BOARD_CALIBRATION_PASS'
 if not a.prepare_only:assert 'PASS BOARD_CALIBRATION' in body
 result=dict(status=status,seconds=time.monotonic()-start,full_board_frame_ms=None,inputs_sha256=sha(root/'inputs.json'))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
