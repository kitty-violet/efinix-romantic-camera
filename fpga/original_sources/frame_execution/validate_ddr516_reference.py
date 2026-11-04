"""Run the official IP 5.16 testbench with the generated 2Gb configuration.

Copies only required files. Uses the generated root ROM/header, since the vendor
Testbench folder contains fixed default geometry/timing headers. No PHY edits.
"""
import argparse,collections,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha

def official_gpio(tb):
 """Use documented registered command/OE and resynchronized DDR I/O paths.

 No delay fitting: parameters match the official Ti60 reference peri.xml and
 DDR3 Soft Controller UG v2.1 table 15. GPIO/PLL primitive sources are unchanged.
 """
 commands=('reset','cs','ras','cas','we','cke','odt')
 declarations='wire [15:0] fabric_addr;wire [2:0] fabric_ba;\n'
 declarations+='wire '+','.join('fabric_'+x for x in commands)+';\n'
 for name in (*commands,'addr','ba'):
  needle=f'.ddr_{name}({name}),'
  assert tb.count(needle)==1,needle
  tb=tb.replace(needle,f'.ddr_{name}(fabric_{name}),')
 section=declarations+'''
wire [25:0] fabric_cmd={fabric_addr,fabric_ba,fabric_reset,fabric_cs,fabric_ras,fabric_cas,fabric_we,fabric_cke,fabric_odt};
wire [25:0] pin_cmd;
assign {addr,ba,reset,cs,ras,cas,we,cke,odt}=pin_cmd;
for(genvar k=0;k<26;k=k+1)begin:g_cmd
 EFX_GPIO_V3 #(.MODE("OUTPUT"),.OUT_REG("REG"),.IO_STANDARD("1.5_V_SSTL"))
  pin(.O({3'bzzz,fabric_cmd[k]}),.OUTCLK(tdqss_clk),.IO(pin_cmd[k]));
end
for(genvar k=0;k<16;k=k+1)begin:g_dq
 wire [3:0] sampled;
 EFX_GPIO_V3 #(.MODE("INOUT"),.OUT_REG("DDIO_RESYNC"),.IN_REG("DDIO_RESYNC"),.OE_REG("REG"),.IO_STANDARD("1.5_V_SSTL"))
  pin(.O({2'bzz,o_dq_lo[k],o_dq_hi[k]}),.OE(o_dq_oe[k]),.OUTCLK(twd_clk),.INCLK(tac_clk),.IO(dq[k]),.I(sampled));
 assign {i_dq_lo[k],i_dq_hi[k]}=sampled[1:0];
end
for(genvar k=0;k<2;k=k+1)begin:g_dqs
 wire [3:0] sampled,sampled_n;
 EFX_GPIO_V3 #(.MODE("INOUT"),.OUT_REG("DDIO_RESYNC"),.IN_REG("DDIO_RESYNC"),.OE_REG("REG"),.IO_STANDARD("1.5_V_Differential_SSTL"))
  pin(.O({2'bzz,o_dqs_lo[k],o_dqs_hi[k]}),.OE(o_dqs_oe[k]),.OUTCLK(tdqss_clk),.INCLK(tac_clk),.IO(dqs[k]),.I(sampled));
 EFX_GPIO_V3 #(.MODE("INOUT"),.OUT_REG("DDIO_RESYNC"),.IN_REG("DDIO_RESYNC"),.OE_REG("REG"),.IO_STANDARD("1.5_V_Differential_SSTL"))
  pin_n(.O({2'bzz,o_dqs_n_lo[k],o_dqs_n_hi[k]}),.OE(o_dqs_n_oe[k]),.OUTCLK(tdqss_clk),.INCLK(tac_clk),.IO(dqs_n[k]),.I(sampled_n));
 assign {i_dqs_lo[k],i_dqs_hi[k]}=sampled[1:0];
 assign {i_dqs_n_lo[k],i_dqs_n_hi[k]}=sampled_n[1:0];
 EFX_GPIO_V3 #(.MODE("OUTPUT"),.OUT_REG("DDIO_RESYNC"),.IO_STANDARD("1.5_V_SSTL"))
  mask(.O({2'bzz,o_dm_lo[k],o_dm_hi[k]}),.OUTCLK(twd_clk),.IO(dm[k]));
end
'''
 start=tb.index('genvar n;');end=tb.index('example_top #(',start)
 return tb[:start]+section+'\n'+tb[end:]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--generated',type=Path,required=True)
 p.add_argument('--dynamic-pll',action='store_true');p.add_argument('--power-on-reset',action='store_true')
 p.add_argument('--official-gpio',action='store_true')
 p.add_argument('--fast-memory',action='store_true');p.add_argument('--observe-read-pipe',action='store_true')
 p.add_argument('--trace-data',action='store_true',help='Read-only diagnostics of public Micron memory tasks and controller I/O')
 p.add_argument('--trace-axi',action='store_true',help='Independent accepted-transfer data/ID/LAST scoreboard')
 p.add_argument('--rx-retime-stages',type=int,choices=[0,2],default=0,help='Isolated synthesizable input latency adapter; requires official GPIO')
 p.add_argument('--distinct-data',action='store_true',help='Use address-dependent byte-distinct normal traffic to expose halfword permutations')
 p.add_argument('--user-mhz',type=int,choices=[125,150],default=125)
 p.add_argument('--v4-dqs',action='store_true',help='Use the V4 differential pair: positive samples and their logical complement')
 p.add_argument('--compile-only',action='store_true')
 p.add_argument('--v4-wrapper',action='store_true',help='Exercise the actual V4 controller wrapper including separate AW/AR bridge and retiming')
 p.add_argument('--fast-pll',action='store_true');p.add_argument('--wall-seconds',type=int,default=300)
 p.add_argument('--pll-protocol-fix',action='store_true',help='Use audited public PLL simulation correction for simultaneous ENA/SEL deassertion')
 p.add_argument('--trace-commands',action='store_true');p.add_argument('--limit-us',type=int,default=2000);a=p.parse_args()
 assert not a.trace_data or a.observe_read_pipe,'Data diagnostics require the public read pipeline output'
 assert not a.rx_retime_stages or a.official_gpio
 assert not a.v4_dqs or a.official_gpio
 assert not a.v4_wrapper or (a.v4_dqs and a.rx_retime_stages==2 and a.observe_read_pipe)
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 ip=a.generated.resolve()/'ip/ddr516_reference';tbdir=ip/'Testbench';records=[]
 def freeze(src,name=None):
  dst=root/(name or src.name);shutil.copy2(src,dst)
  records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)));return dst
 files=[freeze(tbdir/name) for name in ('ddr3.v','efx_iddr.v','efx_oddr.v','efx_srl8.v','memory_checker_axi.v')]
 if a.distinct_data:
  checker=files[4].read_text()
  for counter in ('write_cnt','read_cnt'):
   old='{{WIDTH/32{~aaddr[7:0]}},{WIDTH/32{~'+counter+'[7:0]}},{WIDTH/32{aaddr[7:0]}},{WIDTH/32{'+counter+'[7:0]}}}'
   assert checker.count(old)==2,(counter,checker.count(old))
   checker=checker.replace(old,f'distinct_pattern(aaddr,{counter})')
  function='''function [WIDTH-1:0] distinct_pattern(input [31:0] address,input [8:0] count);
 for(integer lane=0;lane<WIDTH/8;lane=lane+1)
  distinct_pattern[lane*8+:8]=(((address>>4)*13+count*17+lane*29)^(lane<<3)^8'h5a);
endfunction
'''
  checker=checker.replace('localparam  ASIZE',function+'\nlocalparam  ASIZE',1)
  records[4]['original_sha256']=records[4]['sha256'];files[4].write_text(checker)
  records[4].update(sha256=sha(files[4]),transformation='byte-distinct address/count-dependent stimulus; unchanged checker state/handshakes')
 if a.fast_memory:
  from prepare_board_frame_run import fast_memory
  proof=json.loads((ROOT/'tmp/frame-execution/ddr-memory-backend1/result.json').read_text())
  assert proof['status']=='DDR_MODEL_LOOKUP_EQUIVALENCE_PASS' and proof['checks']==6102
  assert sha(files[0])==proof['original_sha256']
  memory=fast_memory(files[0].read_text())
  # A two-state associative key must never coerce an unknown address to zero.
  memory=memory.replace('            get_index = native_index.exists(addr);',
   '            if ($isunknown(addr)) $fatal(1,"DDR516 unknown sparse memory address");\n            get_index = native_index.exists(addr);')
  records[0]['original_sha256']=records[0]['sha256'];files[0].write_text(memory)
  records[0].update(sha256=sha(files[0]),transformation='6102-check validated lookup plus reject unknown address')
  freeze(ROOT/'tmp/frame-execution/ddr-memory-backend1/result.json','lookup_equivalence.json')
 if a.trace_data:
  memory=files[0].read_text()
  for task_name in ('write','read'):
   start=memory.index('    task memory_'+task_name+';');end=memory.index('    endtask',start)
   section=memory[start:end]
   # Observe data after the original task has finished; change no memory value,
   # command, latency, timing checker or protected controller signal.
   marker='        end\n'
   pos=section.rindex(marker)
   section=section[:pos]+f'''            trace_{task_name}_count=trace_{task_name}_count+1;
            if ($isunknown(data)) trace_{task_name}_unknown=trace_{task_name}_unknown+1;
            if(trace_{task_name}_count<=8)begin
                $display("DDR516_MEMORY_{task_name.upper()} time=%t bank=%d row=%h col=%h data=%h",$time,bank,row,col,data);$fflush();end
'''+section[pos:]
   if task_name=='read':
    pos=section.rindex(marker)
    section=section[:pos]+'''            if(rd_pipeline[0])begin
                trace_return_count=trace_return_count+1;
                if($isunknown(data))trace_return_unknown=trace_return_unknown+1;
                if(trace_return_count<=8)begin
                    $display("DDR516_MEMORY_RETURN time=%t bank=%d row=%h col=%h data=%h",$time,bank,row,col,data);$fflush();end
            end
'''+section[pos:]
   memory=memory[:start]+section+memory[end:]
  memory=memory.replace('    task memory_write;', '''    integer trace_write_count=0,trace_read_count=0,trace_write_unknown=0,trace_read_unknown=0;
    integer trace_return_count=0,trace_return_unknown=0;
    task memory_write;''',1)
  records[0].setdefault('original_sha256',records[0]['sha256']);files[0].write_text(memory)
  records[0].update(sha256=sha(files[0]),transformation=records[0].get('transformation','')+'; read-only public memory task diagnostics')
 files.append(freeze(tbdir/'modelsim/ddr516_reference.sv'))
 assert '// IP Version: 5.16' in files[-1].read_text()
 if a.observe_read_pipe:
  rtl=files[-1].read_text();prefix,protected=rtl.split('//pragma protect',1)
  assert prefix.count('    output cal_pass,')==1
  prefix=prefix.replace('    output cal_pass,','    output cal_pass,\n    output [3:0] diag_read_pipe,')
  prefix=prefix.replace('    .cal_pass ( cal_pass ),','    .cal_pass ( cal_pass ),\n    .read_back_en_pipe ( diag_read_pipe ),')
  if a.trace_data:
   prefix=prefix.replace('    output [3:0] diag_read_pipe,','    output [3:0] diag_read_pipe,\n    output [127:0] diag_native_data,\n    output diag_native_valid,\n    output diag_native_ack,')
   prefix=prefix.replace('    .read_back_en_pipe ( diag_read_pipe ),','    .read_back_en_pipe ( diag_read_pipe ),\n    .rd_data(diag_native_data),\n    .rd_valid(diag_native_valid),\n    .rd_ack(diag_native_ack),')
  changed=prefix+'//pragma protect'+protected
  assert changed.split('//pragma protect',1)[1]==protected
  records[-1]['original_sha256']=records[-1]['sha256'];files[-1].write_text(changed)
  records[-1].update(sha256=sha(files[-1]),transformation='export already-public read_back_en_pipe output; protected source identical')
 for name in ('ddr3_controller.vh','ddr3_controller.bin','ddr516_reference_define.svh'):freeze(ip/name)
 header=(root/'ddr3_controller.vh').read_text()
 for value in ('`define ROW 14','`define DRAM_WIDTH 16','`define MICRO_SEC 400','`define tRL 6','`define tWL 5'):assert value in header
 assert '`define RTL_SIM' not in header,'Do not compress reset timing under strict Micron checks'
 freeze(ROOT/'tmp/frame-execution/full-board-vendor-probe3/2048Mb_ddr3_parameters.vh')
 (root/'ddr3_device_ID.vh').write_text('`define sg125\n')
 example=(tbdir/'example_top.v').read_text()
 if a.observe_read_pipe:
  example=example.replace('wire [7:0] cal_fail_log;','wire [7:0] cal_fail_log;wire [3:0] diag_read_pipe;')
  example=example.replace('.cal_done(w_cal_done),','.cal_done(w_cal_done),\n .diag_read_pipe(diag_read_pipe),')
  example=example.rsplit('endmodule',1)[0]+'''always @(diag_read_pipe)begin
 $display("DDR516_READ_PIPE value=%0d",diag_read_pipe);$fflush();end
endmodule
'''
  if a.trace_data:
   example=example.replace('wire [3:0] diag_read_pipe;','wire [3:0] diag_read_pipe;wire [127:0] diag_native_data;wire diag_native_valid,diag_native_ack;')
   example=example.replace('.diag_read_pipe(diag_read_pipe),','.diag_read_pipe(diag_read_pipe),\n .diag_native_data(diag_native_data),.diag_native_valid(diag_native_valid),.diag_native_ack(diag_native_ack),')
   example=example.rsplit('endmodule',1)[0]+'''
integer diag_native_count=0;
always @(diag_read_pipe or posedge ddr_shift_ena)diag_native_count=0;
always @(posedge clk)if(diag_native_valid===1 && diag_native_count<8)begin
 diag_native_count=diag_native_count+1;
 $display("DDR516_NATIVE_RETURN time=%t pipe=%d data=%h ack=%b",$time,diag_read_pipe,diag_native_data,diag_native_ack);$fflush();
end
endmodule
'''
 assert '.axi_bresp(),' in example
 example=example.replace('.axi_bresp(),','.axi_bresp(m_bresp_0),')
 monitor='''
integer measured_writes=0,measured_reads=0,measured_aw=0,measured_ar=0;
always @(posedge clk)if(w_nrst && w_cal_done)begin
 if(m_avalid_0 && m_aready_0)begin
  if(m_atype_0)measured_aw=measured_aw+1;else measured_ar=measured_ar+1;
 end
 if(m_wvalid_0 && m_wready_0)measured_writes=measured_writes+1;
 if(m_bvalid_0 && m_bready_0 && m_bresp_0!==0)$fatal(1,"DDR516 BRESP");
 if(m_rvalid_0 && m_rready_0)begin
  measured_reads=measured_reads+1;
  if(m_rresp_0!==0)$fatal(1,"DDR516 RRESP");
 end
 if(done)begin
  if(measured_writes!=256 || measured_reads!=256 || measured_aw!=4 || measured_ar!=4)
   $fatal(1,"DDR516 incomplete transactions w=%0d r=%0d aw=%0d ar=%0d",measured_writes,measured_reads,measured_aw,measured_ar);
  $display("DDR516_TRAFFIC write_beats=%0d read_beats=%0d writes=%0d reads=%0d bytes_each_direction=%0d",measured_writes,measured_reads,measured_aw,measured_ar,measured_reads*16);$fflush();
 end
end
'''
 example=example.replace('memory_checker_axi #(',monitor+'\nmemory_checker_axi #(')
 if a.trace_axi:
  freeze(HERE/'ddr516_axi_diagnostics.svh')
  example=example.replace('memory_checker_axi #(','`include "ddr516_axi_diagnostics.svh"\nmemory_checker_axi #(')
 if a.v4_wrapper:
  files.extend([freeze(HERE/'ddr516_axi_address_bridge.sv'),freeze(HERE/'board_integration/ddr516_v4_controller.sv')])
  start=example.index('ddr516_reference inst_ddr3_axi');end=example.index(');',start)+2
  instance=example[start:end].replace('ddr516_reference inst_ddr3_axi','ddr516_v4_controller inst_ddr3_axi')
  mapping={'clk':'axi_clk','reset_n':'rstn','tac_clk':'rx_cal_clk','twd_clk':'tx_cal_clk_90edge','tdqss_clk':'tx_cal_clk',
   'reset':'ddr_reset_n','cs':'ddr_cs_n','ras':'ddr_ras_n','cas':'ddr_cas_n','we':'ddr_we_n','cke':'ddr_cke','addr':'ddr_addr','ba':'ddr_ba','odt':'ddr_odt',
   'shift':'pll_shift','shift_sel':'pll_shift_sel','shift_ena':'pll_shift_ena',
   'o_dm_hi':'ddr_dm_hi','o_dm_lo':'ddr_dm_lo','i_dq_hi':'ddr_dq_in_hi','i_dq_lo':'ddr_dq_in_lo',
   'o_dq_hi':'ddr_dq_out_hi','o_dq_lo':'ddr_dq_out_lo','o_dq_oe':'ddr_dq_oe',
   'i_dqs_hi':'ddr_dqs_in_hi','i_dqs_lo':'ddr_dqs_in_lo','o_dqs_hi':'ddr_dqs_out_hi','o_dqs_lo':'ddr_dqs_out_lo',
   'o_dqs_oe':'ddr_dqs_oe','o_dqs_n_oe':'ddr_dqs_oe_n'}
  for old,new in mapping.items():instance=re.sub(r'\.'+old+r'\s*\(','.'+new+'(',instance)
  for channel in ('wdata','wstrb','wlast','wvalid','wready','bid','bresp','bvalid','bready','rid','rdata','rresp','rlast','rvalid','rready'):
   instance=instance.replace('.axi_'+channel+'(','.s_axi_'+channel+'(')
  # The reference stimulus is combined-address. Split it into genuine AW/AR
  # inputs before the same bridge that will be present in the board wrapper.
  for suffix in ('id','addr','len','size','burst'):
   pattern=r'\.axi_a'+suffix+r'\(([^)]+)\),'
   instance,count=re.subn(pattern,lambda m:'.s_axi_aw'+suffix+'('+m[1]+'),.s_axi_ar'+suffix+'('+m[1]+'),',instance)
   assert count==1,(suffix,count)
  instance=instance.replace('.axi_alock(m_alock_0),','.s_axi_awlock(m_alock_0[0]),.s_axi_arlock(m_alock_0[0]),')
  instance=instance.replace('.axi_avalid(m_avalid_0),','.s_axi_awvalid(m_avalid_0 && m_atype_0),.s_axi_arvalid(m_avalid_0 && !m_atype_0),')
  instance=instance.replace('.axi_aready(m_aready_0),','.s_axi_awready(m_awready_0),.s_axi_arready(m_arready_0),')
  for port in ('axi_wid','axi_atype','i_dqs_n_hi','i_dqs_n_lo','o_dqs_n_hi','o_dqs_n_lo','cal_ena','diag_read_pipe','cal_fail_log','cal_pass'):
   instance=re.sub(r'\.'+port+r'\s*\([^)]*\),?', '',instance)
  instance=re.sub(r',\s*\);',');',instance)
  instance=instance.replace('.core_clk',".sdram_clk(ddr_tdqss_clk),.app_sr_req(1'b0),.app_ref_req(1'b0),.app_zq_req(1'b0),\n.s_axi_awcache(4'b0),.s_axi_arcache(4'b0),.s_axi_awprot(3'b0),.s_axi_arprot(3'b0),.s_axi_awqos(4'b0),.s_axi_arqos(4'b0),\n.core_clk",1)
  # Avoid a width mismatch on the existing 16-bit reference pin bus.
  instance=instance.replace('.ddr_addr(ddr_addr)', '.ddr_addr(ddr_addr[13:0])')
  instance+='''
assign ddr_addr[15:14]=0;
assign m_aready_0=m_atype_0 ? m_awready_0 : m_arready_0;
assign ddr_cal_pass=inst_ddr3_axi.calibration_pass;
assign cal_fail_log=inst_ddr3_axi.calibration_log;
assign diag_read_pipe=inst_ddr3_axi.controller.diag_read_pipe;
'''
  example=example[:start]+instance+example[end:]
 (root/'example_top.v').write_text(example);files.append(root/'example_top.v')
 tb=(tbdir/'efx_ddr3_soft_controller_tb.v').read_text()
 if a.user_mhz!=125:
  assert tb.count('#40 user_clk = ~user_clk;')==1
  tb=tb.replace('#40 user_clk = ~user_clk;',f'#{5000/a.user_mhz:.9f} user_clk = ~user_clk;')
  tb=tb.replace('`timescale 100ps/10ps','`timescale 100ps/1ps')
 tb=tb.replace('`define den4096Mb','`define den2048Mb')
 tb=re.sub(r'\s*\$dump(?:file|vars)\([^;]+;', '',tb)
 tb=tb.replace('example_top u_top','example_top #(.END_ADDR(32\'hC00)) u_top')
 if a.official_gpio:
  tb=official_gpio(tb)
  if a.v4_dqs:
   # V4/official Ti60 peri.xml each expose only the positive differential
   # receiver to fabric. The negative sample is its logical complement.
   tb=tb.replace('.O({2\'bzz,o_dqs_n_lo[k],o_dqs_n_hi[k]}),.OE(o_dqs_n_oe[k])',
                 '.O({2\'bzz,~o_dqs_lo[k],~o_dqs_hi[k]}),.OE(o_dqs_oe[k])')
   tb=tb.replace('assign {i_dqs_n_lo[k],i_dqs_n_hi[k]}=sampled_n[1:0];',
                 'assign {i_dqs_n_lo[k],i_dqs_n_hi[k]}=~sampled[1:0];')
  if a.rx_retime_stages:
   # Retiming is a candidate RTL change, not a testbench transport delay.
   # Only the GPIO sample assignments are redirected; IP inputs remain named.
   for name in ('dq','dqs','dqs_n'):
    rhs='~sampled' if name=='dqs_n' and a.v4_dqs else 'sampled'+('_n' if name=='dqs_n' else '')
    old=f'assign {{i_{name}_lo[k],i_{name}_hi[k]}}={rhs}[1:0];'
    new=old.replace(f'i_{name}_lo',f'pad_{name}_lo').replace(f'i_{name}_hi',f'pad_{name}_hi')
    assert tb.count(old)==1,old
    tb=tb.replace(old,new)
   adapter='''wire [15:0] pad_dq_hi,pad_dq_lo;
wire [1:0] pad_dqs_hi,pad_dqs_lo,pad_dqs_n_hi,pad_dqs_n_lo;
ddr516_rx_retime #(.STAGES(2)) rx_retime(.tac_clk(tac_clk),
 .samples_in({pad_dqs_n_hi,pad_dqs_n_lo,pad_dqs_hi,pad_dqs_lo,pad_dq_hi,pad_dq_lo}),
 .samples_out({i_dqs_n_hi,i_dqs_n_lo,i_dqs_hi,i_dqs_lo,i_dq_hi,i_dq_lo}));
'''
   tb=tb.replace('wire [15:0] fabric_addr;',adapter+'wire [15:0] fabric_addr;')
   files.append(freeze(HERE/'ddr516_rx_retime.sv'))
   if a.v4_wrapper:
    # The real wrapper contains this register stage. Do not add another one
    # in the testbench, which would change its physical receive latency.
    tb=tb.replace(adapter,'')
    for name in ('dq','dqs','dqs_n'):
     tb=tb.replace(f'pad_{name}_hi',f'i_{name}_hi').replace(f'pad_{name}_lo',f'i_{name}_lo')
  files.append(freeze(EF/'pt/sim_models/verilog/EFX_GPIO_V3.v'))
  freeze(ROOT/'tmp/frame-execution/ddr516-periphery1/ti60f225_ddr3_controller.peri.pt.v','gpio_reference.peri.pt.v')
 # MEM_BITS controls sparse simulation storage, not the physical DDR density.
 # The new IP calibration writes more locations than the model default holds.
 tb=tb.replace('ddr3 ddr3md','ddr3 #(.DEBUG(0),.MEM_BITS(14)) ddr3md')
 # Explicit fixture assumption: hold the memory reset low until the real
 # controller first asserts it. No shortening of its reset/CKE sequence.
 reset_fixture='wire reset;wire memory_reset;assign memory_reset=reset;'
 if a.power_on_reset:
  reset_fixture='''wire reset;wire memory_reset;reg power_on_hold=1;
assign memory_reset=power_on_hold ? 1'b0 : reset;
initial begin wait(reset===1'b0);power_on_hold=0;
 $display("DDR516_POWER_ON_RESET_RELEASE t_ns=%0.3f",$realtime/10.0);end'''
 tb=tb.replace('wire reset;','wire [2:0] shift;wire [4:0] shift_sel;wire shift_ena;\n'+reset_fixture)
 tb=tb.replace('.rst_n(reset),','.rst_n(memory_reset),')
 tb=tb.replace('.addr(addr),','.addr(addr[13:0]),')
 tb=tb.replace('wire pass;', 'wire cal_done,cal_pass;wire pass;')
 tb=tb.replace('.ddr_cal_done(),','.ddr_cal_done(cal_done),').replace('.ddr_cal_pass(),','.ddr_cal_pass(cal_pass),')
 for port in ('shift','shift_sel','shift_ena'):tb=tb.replace(f'.ddr_{port}(),',f'.ddr_{port}({port}),')
 if a.dynamic_pll:
  if a.fast_pll:
   proof_dir=ROOT/'tmp/frame-execution/ddr516-pll-equivalence1'
   proof=json.loads((proof_dir/'result.json').read_text())
   assert proof['status']=='FIXED_REFERENCE_PLL_EQUIVALENCE_PASS' and proof['checks']==58269
   assert sha(EF/'pt/sim_models/verilog/EFX_PLL_V3.v')==proof['official_sha256']
   assert sha(proof_dir/'EFX_PLL_V3_FAST.v')==proof['fast_sha256']
   if not a.pll_protocol_fix:files.append(freeze(proof_dir/'EFX_PLL_V3_FAST.v'))
   freeze(proof_dir/'result.json','pll_equivalence.json')
  elif not a.pll_protocol_fix:files.append(freeze(EF/'pt/sim_models/verilog/EFX_PLL_V3.v'))
  if a.pll_protocol_fix:
   protocol_dir=ROOT/'tmp/frame-execution/ddr516-pll-protocol1'
   proof=json.loads((protocol_dir/'result.json').read_text())
   assert proof['status']=='PLL_PROTOCOL_CORRECTION_PASS' and proof['original_missed']==7 and proof['phase_checks']==160
   assert sha(EF/'pt/sim_models/verilog/EFX_PLL_V3.v')==proof['official_sha256']
   fixed=protocol_dir/('EFX_PLL_V3_FAST_PROTOCOL.v' if a.fast_pll else 'EFX_PLL_V3_PROTOCOL.v')
   assert sha(fixed)==proof['fast_corrected_sha256' if a.fast_pll else 'corrected_sha256']
   files.append(freeze(fixed));freeze(protocol_dir/'result.json','pll_protocol_proof.json')
  freeze(EF/'ipm/ip/efx_ddr3_soft_controller/fpga/Ti60F225_engboard/ti60f225_ddr3_controller.peri.xml','clock_reference.peri.xml')
  for clock in ('core_clk','tdqss_clk','tac_clk','twd_clk','clk'):
   tb=tb.replace(f'reg {clock};',f'wire {clock};')
   tb,count=re.subn(r'initial begin\s*'+clock+r' =.*?forever begin.*?end\s*end','',tb,flags=re.S)
   assert count==1,(clock,count)
  tb=tb.replace(".pll_lock(1'b1),",'.pll_lock(ddr_pll_locked),')
  tb=tb.replace('wire reset;', '''// Official Ti60 reference PLL clock plan, with no board pin migration.
reg ref50=0;always #100 ref50=~ref50;
wire feedback50,ddr_pll_locked;
assign clk=tdqss_clk;
EFX_PLL_V3 #(.N(1),.M(1),.O(2),.CLKOUT0_DIV(6),.CLKOUT1_DIV(12),
 .CLKOUT2_DIV(6),.CLKOUT3_DIV(6),.CLKOUT4_DIV(48),
 .CLKOUT3_PHASE_STEP(3),.CLKOUT2_DYNPHASE_EN(1),
 .FEEDBACK_CLK("CLK4"),.FEEDBACK_MODE("CORE"),.REFCLK_FREQ(50.0)) ddr_pll(
 .CLKIN({ref50,3'bzzz}),.CLKSEL(2'd3),.RSTN(1'b1),.FBK(feedback50),
 .SHIFT(shift),.SHIFT_SEL(shift_sel),.SHIFT_ENA(shift_ena),
 .CLKOUT0(tdqss_clk),.CLKOUT1(core_clk),.CLKOUT2(tac_clk),
 .CLKOUT3(twd_clk),.CLKOUT4(feedback50),.LOCKED(ddr_pll_locked),.DESKEWED());
wire reset;''')
  model='EFX_PLL_V3'+('_FAST' if a.fast_pll else '')+('_PROTOCOL' if a.pll_protocol_fix else '')
  tb=tb.replace('EFX_PLL_V3 #(',model+' #(')
 else:assert not a.fast_pll and not a.pll_protocol_fix,'PLL corrections require the documented dynamic PLL clock plan'
 tb=tb.replace('if(pass)','if(pass && cal_pass===1\'b1)')
 tb=tb.replace('$display ("DDR3 Controller FAIL");','$fatal(1,"DDR3 Controller FAIL cal_pass=%b data_pass=%b",cal_pass,pass);')
 tb=tb.rsplit('endmodule',1)[0]+'''
initial begin #20000000;$fatal(1,"DDR516 2ms timeout cal_done=%b cal_pass=%b",cal_done,cal_pass);end
always @(posedge cal_done)begin $display("DDR516_CAL_DONE t_ns=%0.3f pass=%b",$realtime/10.0,cal_pass);$fflush();end
initial begin forever begin #1000000;$display("DDR516_PROGRESS t_ns=%0.3f cal_done=%b pass=%b",$realtime/10.0,cal_done,cal_pass);$fflush();end end
always @(reset or memory_reset or cke)if($realtime<10000)begin $display("DDR516_RESET t_ns=%0.3f reset=%b memory_reset=%b cke=%b",$realtime/10.0,reset,memory_reset,cke);$fflush();end
integer shift_count=0;
always @(posedge shift_ena)begin shift_count=shift_count+1;
 if(shift_count<=16)$display("DDR516_SHIFT t_ns=%0.3f value=%d select=%b",$realtime/10.0,shift,shift_sel);end
always @(u_top.cal_fail_log)if(u_top.cal_fail_log!==0)begin
 $display("DDR516_CAL_LOG t_ns=%0.3f log=%h",$realtime/10.0,u_top.cal_fail_log);$fflush();end
endmodule
'''
 tb=tb.replace('#20000000;',f'#{a.limit_us*10000};').replace('DDR516 2ms timeout',f'DDR516 {a.limit_us}us timeout')
 if a.trace_commands:
  tb=tb.rsplit('endmodule',1)[0]+'''
integer cmd_fd;integer seen_activate[0:7];realtime last_activate[0:7];
initial begin cmd_fd=$fopen("pin_commands.csv","w");
 $fdisplay(cmd_fd,"time_ns,ras_cas_we,bank,address");
 for(integer k=0;k<8;k=k+1)begin seen_activate[k]=0;last_activate[k]=0;end end
always @(negedge clk)if(memory_reset===1 && cke===1 && cs===0 && {ras,cas,we}!==3'b111)begin
 $fdisplay(cmd_fd,"%0.3f,%b,%d,%h",$realtime/10.0,{ras,cas,we},ba,addr);
 if({ras,cas,we}===3'b011)begin seen_activate[ba]=1;last_activate[ba]=$realtime;end
 if({ras,cas,we}===3'b010)begin
  for(integer k=0;k<8;k=k+1)if((addr[10] || ba==k) && seen_activate[k])begin
   if(($realtime-last_activate[k])/10.0 < 34.930)
    $display("DDR516_TRAS_PIN t_ns=%0.3f bank=%0d act_ns=%0.3f interval_ns=%0.3f",$realtime/10.0,k,last_activate[k]/10.0,($realtime-last_activate[k])/10.0);
   seen_activate[k]=0;
  end
 end
end
endmodule
'''
 if a.trace_data:
  tb=tb.rsplit('endmodule',1)[0]+'''
integer diag_capture=0;integer diag_reads=0;
always @(posedge shift_ena)diag_reads=0;
always @(negedge clk)if(memory_reset===1 && cke===1 && cs===0 && {ras,cas,we}===3'b101)begin
 diag_reads=diag_reads+1;
 if(diag_reads==1)diag_capture=64;
end
always @(posedge tac_clk)if(diag_capture>0)begin
 #0.1;diag_capture=diag_capture-1;
 $display("DDR516_IO_SAMPLE t_ns=%0.3f dq=%h dqs=%b hi=%h lo=%h dqs_hi=%b dqs_lo=%b oe=%h",$realtime/10.0,dq,dqs,i_dq_hi,i_dq_lo,i_dqs_hi,i_dqs_lo,o_dq_oe);$fflush();
end
always @(u_top.diag_read_pipe or posedge cal_done)begin
 $display("DDR516_MEMORY_COUNTS pipe=%d writes=%0d reads_including_rmw=%0d unknown_writes_including_half_bursts=%0d unknown_reads_including_rmw=%0d returned_bursts=%0d unknown_returned_bursts=%0d",u_top.diag_read_pipe,ddr3md.trace_write_count,ddr3md.trace_read_count,ddr3md.trace_write_unknown,ddr3md.trace_read_unknown,ddr3md.trace_return_count,ddr3md.trace_return_unknown);$fflush();
end
endmodule
'''
  if a.dynamic_pll:
   tb=tb.rsplit('endmodule',1)[0]+'''
always @(shift or shift_sel or shift_ena)begin
 $display("DDR516_PLL_REQUEST t_ns=%0.3f shift=%d sel=%b ena=%b",$realtime/10.0,shift,shift_sel,shift_ena);$fflush();end
always @(ddr_pll.CLKOUT2_shift)begin
 $display("DDR516_PLL_APPLIED t_ns=%0.3f delay_fs=%0.3f",$realtime/10.0,ddr_pll.CLKOUT2_shift);$fflush();end
endmodule
'''
 (root/'tb.sv').write_text(tb);files.append(root/'tb.sv')
 for f in (root/'example_top.v',root/'tb.sv',root/'ddr3_device_ID.vh'):
  records.append(dict(source='diagnostic testbench/explicit device selection',snapshot=str(f),sha256=sha(f)))
 meta=dict(inputs=records,generation_request_sha256=sha(a.generated/'generation_request.json'),device=dict(density_bits=2**31,width=16,row=14,col=10,bank=3,clock_mhz=400,cl=6,cwl=5),generated_header_used=True,reset_acceleration=False,board_release=False,runner_sha256=sha(__file__))
 meta.update(dynamic_pll=a.dynamic_pll,power_on_reset_assumption=a.power_on_reset,trace_commands=a.trace_commands,trace_data=a.trace_data,limit_us=a.limit_us,official_gpio=a.official_gpio,fast_memory=a.fast_memory,fast_pll=a.fast_pll,pll_protocol_fix=a.pll_protocol_fix,observe_read_pipe=a.observe_read_pipe,project_patch=json.loads((a.generated/'generation_request.json').read_text()).get('project_patch'))
 meta['trace_axi']=a.trace_axi
 meta['rx_retime_stages']=a.rx_retime_stages
 meta['distinct_data']=a.distinct_data
 meta['user_mhz']=a.user_mhz;meta['v4_dqs']=a.v4_dqs
 meta['v4_wrapper']=a.v4_wrapper
 (root/'inputs.json').write_text(json.dumps(meta,indent=2))
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 commands=[('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16','+incdir+'+str(root),*files]),
  ('simulation',[sim/'vsim.exe','-c','work.efx_ddr3_soft_controller_tb','-do','run.do'])]
 stages=[];start=time.monotonic();status='FAIL';body=''
 if a.compile_only:commands=commands[:2]
 for name,cmd in commands:
  t=time.monotonic()
  timed_out=False
  with (root/(name+'.log')).open('wb') as out:
   proc=subprocess.Popen(list(map(str,cmd)),cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
   (root/'active_process.json').write_text(json.dumps(dict(stage=name,pid=proc.pid,command=list(map(str,cmd)),started=time.time())))
   try:proc.wait(timeout=a.wall_seconds)
   except subprocess.TimeoutExpired:
    timed_out=True
    # Only terminate the still-live process tree created immediately above.
    with (root/'timeout_cleanup.log').open('wb') as cleanup:
     subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=cleanup,stderr=subprocess.STDOUT,timeout=15)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
     (root/'cleanup_required.json').write_text(json.dumps(dict(pid=proc.pid,command=list(map(str,cmd)),status='OWNED_PROCESS_STILL_RUNNING',board_release=False)))
  body=(root/(name+'.log')).read_text(errors='replace');stages.append(dict(stage=name,returncode=proc.returncode,seconds=time.monotonic()-t,wall_timeout=timed_out))
  if timed_out:break
  if proc.returncode or re.search(r'\*\* (?:Error|Fatal)',body):break
 else:
  if 'DDR3 Controller PASS' in body and 'DDR516_TRAFFIC' in body and not re.search(r'\b(?:ERROR|Error):|DDR3 Controller FAIL',body):status='DDR516_STRICT_REFERENCE_PASS'
 if a.trace_axi and not re.search(r'DDR516_SCOREBOARD mismatches=0 b=4\b',body):status='FAIL'
 if a.compile_only and len(stages)==2 and all(s['returncode']==0 for s in stages) and 'Errors: 0' in body:
  status='DDR516_REFERENCE_COMPILE_PASS'
 errors=[line for line in body.splitlines() if re.search(r'\b(?:ERROR|Error):|\*\* (?:Error|Fatal)',line)]
 for row in records:assert sha(row['snapshot'])==row['sha256']
 result=dict(status=status,seconds=time.monotonic()-start,stages=stages,error_count=len(errors),error_categories=dict(collections.Counter(re.sub(r'.*ERROR: ?','',x) for x in errors)),first_errors=errors[:12],markers=[line for line in body.splitlines() if 'DDR516_' in line or 'DDR3 Controller ' in line],inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/(stages[-1]['stage']+'.log')),board_release=False,full_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:result[k] for k in ('status','seconds','error_count','first_errors','inputs_sha256','log_sha256')}),flush=True)
 if status not in ('DDR516_STRICT_REFERENCE_PASS','DDR516_REFERENCE_COMPILE_PASS'):raise SystemExit(1)

if __name__=='__main__':main()
