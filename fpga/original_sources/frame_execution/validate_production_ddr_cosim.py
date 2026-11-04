"""Bounded production-CNN/four-master/real-DDR partition regression.

Retains both the original four-state CNN and its native partition for edge
comparisons. Other masters are short protocol stimuli, not camera/display
models. Micron memory_write only supplies deterministic initial contents;
all observed transfers travel through the actual fabric, controller and pads.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from probe_board_dpi_cost import BIN, SIM, run
from run_units import ROOT, sha


def ansi_ports(text):
    header=text[:text.index('\n);')]
    result=[]
    for m in re.finditer(r'\b(input|output)\s+(?:wire\s+)?(\[\d+:\d+\]\s*)?([^;()]+?)(?=\binput\b|\boutput\b|$)',header):
        width=m[2].strip() if m[2] else ''
        for name in m[3].strip().rstrip(',').split(','):
            name=name.strip()
            assert re.fullmatch(r'\w+',name),(name,m[0])
            result.append((m[1],width,name))
    return result


def make_tb(base,fixture,root):
    text=(fixture/'tb.sv').read_text().replace(fixture.as_posix(),root.as_posix())
    # Keep exactly the already-passing edge comparison and coefficient loader.
    start=text.index('task serve(');end=text.index('reg [127:0] compute_rom',start)
    text=text[:start]+text[end:]
    start=text.index('// Endpoint state');end=text.index('initial begin\n $readmemh',start)
    text=text[:start]+text[end:]
    external=['arready','awready','wready','rvalid','rlast','bvalid','rdata','rresp','bresp']
    for name in external:
        text=re.sub(r'reg (\[[^\]]+\]) '+name+r'=0;',r'wire \1 '+name+';',text)
    text=text.replace('start_valid=0;arready=0;rvalid=0;rlast=0;rresp=0;','start_valid=0;')
    # No CNN job exists during calibration. Hold its testbench clock in reset
    # while the real controller continues on the independent 150 MHz bus.
    text=text.replace('always #3.333 clk=~clk;',
        'reg bus_clk=0;always #3.333 bus_clk=~bus_clk;always @(bus_clk)clk=cal_done ? bus_clk : 0;')
    text=text.replace('reset_all();compute_crc32=', 'reset_all();wait(cal_done===1);@(negedge clk);traffic_start=1;compute_crc32=')
    text=text.replace('while(compute_completed<13 || dma_completed<25)', 'while(compute_completed<13 || dma_completed<25 || !(&traffic_done))')
    text=text.replace('PASS PRODUCTION_DPI_ACTIVE_PREFIX','PASS PRODUCTION_DPI_VENDOR_DDR')
    text=text.replace('initial begin #1000000;', 'initial begin #12000000;')
    text=text.replace('$finish;', '$display("DDR_CONCURRENT_COVERAGE traffic_done=%b checked_read_beats=%0d overlap=%0d",traffic_done,checked_read_beats,overlap);\n $finish;')
    text=text[:text.rindex('endmodule')]
    text=text.replace('module tb; timeunit 1ns;timeprecision 1ps;',
        'module tb; timeunit 1ns;timeprecision 1ps;\nwire cal_done;reg traffic_start=0;reg [2:0] traffic_done=0;integer checked_read_beats=0,overlap=0;')
    fabric=ansi_ports((base/'assembly/ti60_ddr_four_master_fabric.sv').read_text())
    for direction,width,name in fabric:
        if name in ('clk','rst_n'):continue
        kind='reg' if direction=='input' and name.startswith('cpu_') else 'wire'
        text+=f'{kind} {width} {name}'+('=0' if kind=='reg' else '')+';\n'
    text+='ti60_ddr_four_master_fabric fabric('+','.join(f'.{n}({"!reset && cal_done" if n=="rst_n" else "bus_clk" if n=="clk" else n})' for _,_,n in fabric)+');\n'
    for direction,width,name in fabric:
        if not name.startswith('s_axi_'):continue
        signal=name[6:]
        total=int(re.search(r'\d+',width)[0])+1;bits=total//3
        if direction=='input':
            for lane in (0,2):text+=f'reg [{bits-1}:0] p{lane}_{signal}=0;\n'
            cnn=dict(arid="4'd1",awid="4'd1",arsize="3'd4",awsize="3'd4",arburst="2'd1",awburst="2'd1").get(signal,signal)
            text+=f'assign {name}={{p2_{signal},{cnn},p0_{signal}}};\n'
        else:
            for lane in (0,2):text+=f'wire [{bits-1}:0] p{lane}_{signal}={name}[{lane*bits}+:{bits}];\n'
            if signal not in ('rid','bid'):text+=f'assign {signal}={name}[{bits}+:{bits}];\n'
    # Match the full board's PLL plan, pad model and explicit 4-to-8 ID wiring.
    ctl=ansi_ports((base/'ddr516_v4_controller.sv').read_text())
    special=dict(axi_clk='bus_clk',core_clk='ddr_core_clk200',rstn='controller_reset_n && pll_locked',
                 app_sr_req="1'b0",app_ref_req="1'b0",app_zq_req="1'b0")
    for direction,width,name in ctl:
        if name in special or name.startswith('s_axi_') or name=='cal_done':continue
        text+=f'wire {width} {name};\n'
    text+='wire ddr_core_clk200,pll_locked,feedback50;reg ref25=0,controller_reset_n=0;\n'
    text+='always #20 ref25=~ref25;initial begin #100;controller_reset_n=1;end\n'
    text+='''EFX_PLL_V3_FAST_PROTOCOL #(.N(1),.M(2),.O(2),.CLKOUT0_DIV(6),.CLKOUT1_DIV(6),
 .CLKOUT2_DIV(6),.CLKOUT3_DIV(12),.CLKOUT4_DIV(48),.CLKOUT0_PHASE_STEP(3),.CLKOUT2_DYNPHASE_EN(1),
 .FEEDBACK_CLK("CLK4"),.FEEDBACK_MODE("CORE"),.REFCLK_FREQ(25.0)) ddr_pll(
 .CLKIN({ref25,3'bzzz}),.CLKSEL(2'd3),.RSTN(1'b1),.FBK(feedback50),
 .SHIFT(pll_shift),.SHIFT_SEL(pll_shift_sel),.SHIFT_ENA(pll_shift_ena),
 .CLKOUT0(tx_cal_clk_90edge),.CLKOUT1(sdram_clk),.CLKOUT2(rx_cal_clk),
 .CLKOUT3(ddr_core_clk200),.CLKOUT4(feedback50),.LOCKED(pll_locked),.DESKEWED());
assign tx_cal_clk=sdram_clk;
wire [7:0] controller_rid,controller_bid;
assign m_axi_rid=controller_rid[3:0];assign m_axi_bid=controller_bid[3:0];
'''
    connections=[]
    for direction,width,name in ctl:
        value=special.get(name,name)
        if name.startswith('s_axi_'):
            signal=name[6:];value='m_axi_'+signal
            if signal in ('arid','awid'):value="{4'b0,"+value+'}'
            elif signal in ('rid','bid'):value='controller_'+signal
            elif signal in ('arcache','awcache','arqos','awqos','arprot','awprot','arlock','awlock'):value="'0"
        connections.append(f'.{name}({value})')
    text+='ddr516_v4_controller controller('+','.join(connections)+');\n'
    pins=ansi_ports((base/'ddr516_periphery_model.sv').read_text())
    text+='board_ddr_io_model io_model('+','.join(f'.{n}({n})' for _,_,n in pins)+');\n'
    text+='''
// Initial content is supplied only once, before the address reaches the PHY.
// Writes are NEVER applied by this API; real DDR W/DQ traffic performs them.
bit initialized[bit[31:0]];
task initial_word(input [31:0] address);reg[31:0] physical;begin
 if(!initialized.exists(address))begin
  physical=address;
  io_model.memory.memory_write(physical[13:11],physical[27:14],physical[10:1],memread(address));
  initialized[address]=1;
 end
end endtask
reg[31:0] check_raddr=0,check_waddr=0;integer check_rleft=0,check_wleft=0;
always @(posedge clk)if(cal_done && !reset)begin
 if(protocol_error)$fatal(1,"four-master protocol error");
 if(arvalid && !arready)ar_hold++;
 if(rvalid && !rready)r_hold++;
 if(arvalid && arready)ar_count++;
 if(rvalid && rready)r_count++;
 if(awvalid && awready)aw_count++;
 if(wvalid && wready)w_count++;
 if(bvalid && bready)b_count++;
 if(busy && (|{p0_arvalid,p0_awvalid,p2_arvalid,p2_awvalid,cpu_axi_arvalid,cpu_axi_awvalid}))overlap++;
 if(m_axi_arvalid && m_axi_arready)begin
  if(check_rleft)$fatal(1,"overlapping scoreboard reads");
  check_raddr=m_axi_araddr;check_rleft=int'(m_axi_arlen)+1;
  for(integer i=0;i<check_rleft;i++)initial_word(check_raddr+i*16);
 end
 if(m_axi_awvalid && m_axi_awready)begin
  if(check_wleft)$fatal(1,"overlapping scoreboard writes");
  check_waddr=m_axi_awaddr;check_wleft=int'(m_axi_awlen)+1;
  for(integer i=0;i<check_wleft;i++)initial_word(check_waddr+i*16);
 end
 if(m_axi_wvalid && m_axi_wready)begin
  if(check_wleft==0 || m_axi_wlast!==(check_wleft==1))$fatal(1,"W shape");
  next_word=memread(check_waddr);
  for(integer i=0;i<16;i++)if(m_axi_wstrb[i])next_word[i*8+:8]=m_axi_wdata[i*8+:8];
  memory[check_waddr]=next_word;check_waddr+=16;check_wleft--;
 end
 if(m_axi_rvalid && m_axi_rready)begin
  if(check_rleft==0 || m_axi_rlast!==(check_rleft==1) || m_axi_rresp!==0 || m_axi_rdata!==memread(check_raddr))
   $fatal(1,"real DDR read mismatch addr=%h got=%h expected=%h left=%0d",check_raddr,m_axi_rdata,memread(check_raddr),check_rleft);
  checked_read_beats++;check_raddr+=16;check_rleft--;
 end
 if(m_axi_bvalid && m_axi_bready && m_axi_bresp!==0)$fatal(1,"DDR B error");
end
initial begin wait(cal_done===1);$display("VENDOR_DDR_CAL_DONE sim_ns=%0.3f",$realtime);$fflush();end
initial forever begin #1000000;
 $display("VENDOR_DDR_PROGRESS sim_ns=%0.3f calibrated=%b compute=%0d dma=%0d",$realtime,cal_done,compute_completed,dma_completed);$fflush();
end
'''
    # Real read/write transfers by the two other wide clients and the CPU
    # narrow adapter. Read ready deliberately pauses; W supply is intermittent.
    for lane,baseaddr,width,port,done in [(0,0x06000000,128,'p0_',0),(2,0x06400000,128,'p2_',1),(3,0xe1001000,32,'cpu_axi_',2)]:
        lines=f'''initial begin
 wait(traffic_start);@(negedge clk);
 {port}awid={lane};{port}arid={lane};{port}awsize={4 if width==128 else 2};{port}arsize={4 if width==128 else 2};
 {port}awburst=1;{port}arburst=1;{port}awlen=7;{port}arlen=7;
 for(integer j=0;j<8;j++)begin
  {port}awaddr=32'h{baseaddr:08x}+j*4096;{port}awvalid=1;
  do @(posedge clk);while(!{port}awready);@(negedge clk);{port}awvalid=0;
  for(integer k=0;k<8;k++)begin
   repeat((k+j)%3)@(negedge clk);
   {port}wdata={width}'(32'habcd0000+j*8+k);{port}wstrb='1;{port}wlast=k==7;{port}wvalid=1;
   do @(posedge clk);while(!{port}wready);@(negedge clk);{port}wvalid=0;
  end
  {port}bready=1;do @(posedge clk);while(!{port}bvalid);
  if({port}bresp!==0 || {port}bid!=={lane})$fatal(1,"client {lane} B route");
  @(negedge clk);{port}bready=0;
  {port}araddr=32'h{baseaddr:08x}+j*4096;{port}arvalid=1;
  do @(posedge clk);while(!{port}arready);@(negedge clk);{port}arvalid=0;
  for(integer k=0;k<8;k++)begin
   repeat((j+k)%4)@(negedge clk);{port}rready=1;
   do @(posedge clk);while(!{port}rvalid);
   if({port}rdata!=={width}'(32'habcd0000+j*8+k) || {port}rresp!==0 || {port}rid!=={lane} || {port}rlast!==(k==7))$fatal(1,"client {lane} R data/route");
   @(negedge clk);{port}rready=0;
  end
 end
 traffic_done[{done}]=1;
end
'''
        text+=lines
    text+='endmodule\n'
    return text


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=int,default=3600);a=p.parse_args()
    assert 60<=a.timeout_seconds<=3600
    root=a.root.resolve();base=a.board.resolve();native=a.native.resolve();fixture=a.fixture.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    bm=json.loads((base/'inputs.json').read_text());nm=json.loads((native/'inputs.json').read_text())
    fm=json.loads((fixture/'result.json').read_text())
    assert fm['status']=='PRODUCTION_ACTIVE_DPI_EXTERNAL_EQUIVALENCE_PASS'
    assert fm['testbench_sha256']==sha(fixture/'tb.sv')
    assert nm['board_inputs_sha256']==sha(base/'inputs.json') and fm['native_inputs_sha256']==sha(native/'inputs.json')
    for row in bm['inputs']+nm['inputs']:assert sha(row['snapshot'])==row['sha256']
    assert sha(native/'production.dll')==nm['dll_sha256']
    for f in fixture.glob('*.hex'):shutil.copy2(f,root/f.name)
    # The encrypted controller opens this path relative to the simulator CWD.
    rom=base/'ddr3_controller.bin';assert rom.is_file()
    shutil.copy2(rom,root/rom.name)
    shutil.copy2(fixture/'reference.sv',root/'reference.sv')
    (root/'tb.sv').write_text(make_tb(base,fixture,root))
    names={'ti60_ddr_four_master_fabric.sv','sapphire_axi32_to_axi128.sv','axi_three_master_burst_arbiter.sv',
           'ddr516_axi_address_bridge.sv','ddr516_v4_controller.sv','ddr516_rx_retime.sv','ddr516_reference.sv',
           'ddr3.v','efx_iddr.v','efx_oddr.v','efx_srl8.v','EFX_GPIO_V3.v','EFX_PLL_V3_FAST_PROTOCOL.v','ddr516_periphery_model.sv'}
    boardfiles=[Path(f) for f in bm['source_files'] if Path(f).name in names]
    assert names=={f.name for f in boardfiles},names-{f.name for f in boardfiles}
    files=[Path(r['snapshot']) for r in nm['inputs'] if Path(r['snapshot']).name!='cnn_production_frame.sv']+boardfiles
    files=list(dict.fromkeys(files))+[root/'reference.sv',native/'cnn_production_frame_dpi.sv',root/'tb.sv']
    manifest=dict(controller_rom_sha256=sha(rom),board_inputs_sha256=sha(base/'inputs.json'),native_inputs_sha256=sha(native/'inputs.json'),
                  fixture_inputs_sha256=sha(fixture/'result.json'),files=[dict(path=str(f),sha256=sha(f)) for f in files],
                  timeout_seconds=a.timeout_seconds,runner_sha256=sha(__file__),full_frame=False,real_vendor_ddr=True,four_state_edge_compare=True)
    (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16',
         *['+incdir+'+str(d) for d in [native/'source',base,base/'assembly',*[Path(x) for x in bm['include_dirs']]]],*files],root,'compile',env,90)
    started=time.monotonic()
    try:
        seconds=run([SIM/'vsim.exe','-c','-warning','14408,16154','-voptargs=-warning 14408 -O5',
                     '-sv_lib',native/'production','work.tb','-do','run.do'],root,'simulation',env,a.timeout_seconds)
    except (AssertionError,subprocess.TimeoutExpired) as exc:
        log=(root/'simulation.log').read_text(errors='replace')
        result=dict(status='PRODUCTION_DDR_HOST_TIMEOUT' if isinstance(exc,subprocess.TimeoutExpired) else 'PRODUCTION_DDR_SIMULATION_FAILED',
                    error=str(exc),wall_seconds=time.monotonic()-started,full_frame=False,
                    inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
                    progress=re.findall(r'VENDOR_DDR_(?:PROGRESS|CAL_DONE).*',log))
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        raise
    log=(root/'simulation.log').read_text(errors='replace')
    assert not re.search(r'\*\* (Fatal|Error)| ERROR: |Failed to open readmem',log)
    line=re.search(r'PASS PRODUCTION_DPI_VENDOR_DDR .*',log);assert line
    coverage=re.search(r'DDR_CONCURRENT_COVERAGE traffic_done=111 checked_read_beats=(\d+) overlap=(\d+)',log)
    assert coverage and int(coverage[1])>1000 and int(coverage[2])>0
    result=dict(status='PRODUCTION_DPI_REAL_DDR_FOUR_MASTER_EQUIVALENCE_PASS',wall_seconds=seconds,line=line[0],coverage=coverage[0],
                inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),native_inputs_sha256=sha(native/'inputs.json'),
                real_calibration=True,full_frame=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
