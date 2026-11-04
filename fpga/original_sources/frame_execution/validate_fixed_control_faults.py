"""Bounded four-state direct JTAG tests of the new control bus owner/error paths."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run
from validate_fixed_control_soc import public_jtag,DIRECT_TASKS

BODY=r'''
 integer mode=0,ticks=0,ac=0,wc=0,rc=0,bc=0,rb=0,checks=0,holds=0;
 integer lost_aw=0,lost_w=0,lost_ar=0,rlasts=0;
 reg have_aw=0,have_w=0,have_ar=0,first_r=1;integer age=0;
 reg [31:0] memory=0,held_aw=0,held_ar=0,held_w=0;reg hold_aw=0,hold_ar=0,hold_w=0;
 wire sys_reset=io_systemReset;
 always @(negedge clk)begin
  axiA_awready=!reset && !sys_reset && !have_aw && (ticks%17==1);
  axiA_wready=!reset && !sys_reset && !have_w && (ticks%13==2);
  axiA_arready=!reset && !sys_reset && !have_ar && (ticks%11==3);
  io_apbSlave_0_PREADY=!reset && ticks%3==0;
  io_apbSlave_0_PSLVERROR=mode==7;
 end
 always @(posedge clk)begin
  ticks<=ticks+1;
  if(reset || sys_reset)begin
   if(have_aw)lost_aw<=lost_aw+1;if(have_w)lost_w<=lost_w+1;if(have_ar)lost_ar<=lost_ar+1;
   have_aw<=0;have_w<=0;have_ar<=0;axiA_bvalid<=0;axiA_rvalid<=0;
   axiA_rresp<=0;axiA_rid<=0;axiA_rlast<=1;axiA_bresp<=0;axiA_bid<=0;
   first_r<=1;age<=0;hold_aw<=0;hold_ar<=0;hold_w<=0;
  end else begin
   if(hold_aw && (!axiA_awvalid || axiA_awaddr!==held_aw))$fatal(1,"held AW changed");
   if(hold_ar && (!axiA_arvalid || axiA_araddr!==held_ar))$fatal(1,"held AR changed");
   if(hold_w && (!axiA_wvalid || axiA_wdata!==held_w))$fatal(1,"held W changed");
   hold_aw<=axiA_awvalid&&!axiA_awready;held_aw<=axiA_awaddr;
   hold_ar<=axiA_arvalid&&!axiA_arready;held_ar<=axiA_araddr;
   hold_w<=axiA_wvalid&&!axiA_wready;held_w<=axiA_wdata;
   if(hold_aw||hold_ar||hold_w)holds<=holds+1;
   if(axiA_awvalid&&axiA_awready)begin
    if(have_aw || axiA_awaddr!=32'he1200000 || axiA_awlen!=0 || axiA_awsize!=2)$fatal(1,"AW bounds");
    have_aw<=1;ac<=ac+1;
   end
   if(axiA_wvalid&&axiA_wready)begin
    if(have_w || axiA_wstrb!=15 || !axiA_wlast)$fatal(1,"W bounds");
    have_w<=1;memory<=axiA_wdata;wc<=wc+1;
   end
   if(axiA_arvalid&&axiA_arready)begin
    if(have_ar || axiA_araddr!=32'he1200000 || axiA_arlen!=0 || axiA_arsize!=2)$fatal(1,"AR bounds");
    have_ar<=1;rc<=rc+1;first_r<=1;
   end
   if(have_ar || (have_aw&&have_w))age<=age+1;else age<=0;
   if(have_aw&&have_w&&!axiA_bvalid && age>(mode==8 ? 10000:9))begin
    axiA_bvalid<=1;axiA_bresp<=mode==1 ? 2:0;axiA_bid<=mode==2 ? 7:0;
   end
   if(axiA_bvalid&&axiA_bready)begin have_aw<=0;have_w<=0;axiA_bvalid<=0;bc<=bc+1;end
   if(have_ar&&!axiA_rvalid && age>(mode==6 ? 10000:7))begin
    axiA_rvalid<=1;axiA_rdata<=memory;axiA_rresp<=mode==3 ? 2:0;
    axiA_rid<=mode==4 ? 5:0;axiA_rlast<=mode==5 && first_r ? 0:1;
   end
   if(axiA_rvalid&&axiA_rready)begin
    axiA_rvalid<=0;rb<=rb+1;
    if(axiA_rlast)begin have_ar<=0;rlasts<=rlasts+1;end else begin first_r<=0;age<=0;end
   end
  end
 end
 `include "jtag_direct_tasks.svh"
 task reboot;
  begin
   @(negedge clk);reset=1;repeat(10)@(negedge clk);reset=0;
   repeat(100)@(negedge clk);jexpect(32'hf9000f04,1);
  end
 endtask
 task send_only(input bit wr);
  reg [74:0] command;
  begin
   command={2'd2,wr,32'h125487ab,32'he1200000,8'd0};jscan_start(0);
   for(integer k=0;k<75;k=k+1)begin jtdi=command[k];jclock();end
   jscan_end();
  end
 endtask
 initial begin
  reg [31:0] got;bit error;integer before_reads;
  for(mode=0;mode<=8;mode=mode+1)begin
   // Initialization must complete without an APB fault; inject afterwards.
   begin integer save_mode;save_mode=mode;mode=0;reboot();mode=save_mode;end
   before_reads=rb;
   if(mode==0)begin
    jwrite(32'he1200000,32'h814672a3);jexpect(32'he1200000,32'h814672a3);
    if(dut.fatal_bus_error)$fatal(1,"healthy path faulted");
   end else if(mode==7)begin
    jtransaction(0,32'hf8100004,0,2,got,error);if(!error)$fatal(1,"APB error lost");
   end else begin
    jtransaction(mode==1||mode==2||mode==8,32'he1200000,32'h923812ab,2,got,error);
    if(mode==6||mode==8)begin
     if(!dut.fatal_bus_error)$fatal(1,"bus timeout failed to revoke permit");
    end else if(!error)$fatal(1,"AXI error lost mode=%0d",mode);
    if(mode==5 && rb-before_reads!=2)$fatal(1,"missing RLAST not drained");
   end
   if(mode!=0 && !dut.fatal_bus_error)$fatal(1,"sticky fault absent mode=%0d",mode);
   if(boot_run_allowed)$fatal(1,"unverified permit");
   checks=checks+1;$display("CASE CONTROL_FAULT mode=%0d",mode);
  end
  // Partial transactions may only be discarded by the common reset.
  mode=0;reboot();send_only(0);wait(dut.state==8);reboot();
  jwrite(32'he1200000,32'h12846572);jexpect(32'he1200000,32'h12846572);checks=checks+1;
  reboot();send_only(1);wait(dut.state==5 || dut.state==6);reboot();
  jwrite(32'he1200000,32'h32647891);jexpect(32'he1200000,32'h32647891);checks=checks+1;
  if(ac!=bc+lost_aw || wc!=bc+lost_w || rc!=rlasts+lost_ar || holds==0)
   $fatal(1,"coverage aw=%0d w=%0d b=%0d ar=%0d lasts=%0d reset_aw/w/ar=%0d/%0d/%0d",ac,wc,bc,rc,rlasts,lost_aw,lost_w,lost_ar);
  $display("PASS FIXED_CONTROL_FAULTS cases=%0d held_cycles=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d",checks,holds,ac,wc,bc,rc,rb);$finish;
 end
 initial begin #10000000;$fatal(1,"control fault test timeout state=%0d mode=%0d",dut.state,mode);end
endmodule
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
 assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir();records=[]
 try:
  v=HERE/'variants/fixed_boot_rom'
  for n in ('cnn_fixed_control_soc.sv','cnn_boot_sequencer.sv'):
   f=v/n;d=r/n;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
  vendor=ROOT/'tmp/frame-execution/connected-board-fault-boundary1/assembly/EfxSapphireSoc.v'
  (r/'official_jtag.sv').write_text(public_jtag(vendor))
  header=(v/'cnn_fixed_control_soc.sv').read_text().split(');',1)[0]
  decl=['`timescale 1ns/1ps','module tb_control_faults;','reg clk=0,reset=1;always #3.333333 clk=~clk;',
   'reg jtck=0,jtdi=0,jsel=0,jcapture=0,jshift=0,jupdate=0,jreset=0;wire jtdo;'];links=[]
  aliases=dict(io_systemClk='clk',io_asyncReset='reset',jtagCtrl_tck='jtck',jtagCtrl_tdi='jtdi',jtagCtrl_enable='jsel',jtagCtrl_capture='jcapture',jtagCtrl_shift='jshift',jtagCtrl_update='jupdate',jtagCtrl_reset='jreset',jtagCtrl_tdo='jtdo')
  for direction,width,name in re.findall(r'\b(input|output)\s+(?:wire|reg)\s*(\[[^]]+\])?\s*(\w+)',header):
   target=aliases.get(name,name);links.append(f'.{name}({target})')
   if name not in aliases:decl.append(f'{"reg" if direction=="input" else "wire"} {width} {name}'+('=0' if direction=='input' else '')+';')
  decl.append('cnn_fixed_control_soc dut('+','.join(links)+');')
  (r/'tb_control_faults.sv').write_text('\n'.join(decl)+'\n'+BODY)
  (r/'jtag_direct_tasks.svh').write_text((HERE/'board_integration/jtag_memory_tasks.svh').read_text().split('task jdebug(')[0]+DIRECT_TASKS)
  generated={n:sha(r/n) for n in ('official_jtag.sv','tb_control_faults.sv','jtag_direct_tasks.svh')}
  (r/'inputs.json').write_text(json.dumps(dict(inputs=records,generated=generated,runner_sha256=sha(__file__),vendor_sha256=sha(vendor)),indent=2))
  run([MS/'vlib.exe','work'],r,'vlib.log',timeout=30)
  run([MS/'vlog.exe','-sv',*[x['snapshot'] for x in records],r/'official_jtag.sv',r/'tb_control_faults.sv'],r,'compile.log',timeout=30)
  scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch));t=time.monotonic()
  with (r/'simulation.log').open('w') as log:
   proc=subprocess.run([str(MS/'vsim.exe'),'-c','work.tb_control_faults','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
  body=(r/'simulation.log').read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(?:FAIL|\*\* (?:Error|Fatal))',body)
  lines=re.findall(r'^# (PASS FIXED_CONTROL_FAULTS.*)$',body,re.M);assert len(lines)==1
  for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
  result=dict(status='FIXED_CONTROL_FAULTS_PASS',lines=lines,seconds=time.monotonic()-t,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),model='external JTAG actual; AXI32/APB slaves are fault injection models',board_release=False)
  (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
