"""Observe the frozen board's real Sapphire boot, without supplying firmware.

This is a startup prerequisite probe, NOT a full-frame or DDR test. The vendor
ModelSim CPU model executes the unchanged OCR image; SPI inputs equal the board
ties. Actual cnn_apb_control is connected. No testbench APB/weight injection.
"""
import argparse,json,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,sha

MS=Path('E:/altera/15.0/modelsim_ase/win32aloem')
MODEL=ROOT/'tmp/efinity-smoke-2026.1/sapphire-lite-ipgen-i1-o4/ip/sapphire_lite_i1_o4/source/hardware/netlist/source_modelsim/EfxSapphireSoc.v'

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cycles',type=int,default=1000000);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 board=ROOT/'tmp/frame-execution/connected-board-sharing1'
 actual=(board/'assembly/EfxSapphireSoc.v').read_text();model=MODEL.read_text()
 attr='(* syn_ramstyle = "registers" *) '
 assert actual.split('module EfxCPUSp1')[0].replace(attr,'')==model.split('module EfxCPUSp1')[0].replace(attr,'')
 sources=[MODEL,board/'assembly/cnn_apb_control.sv'];records=[]
 for f in [*sources,*board.glob('*.bin')]:
  dst=root/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
 # Reuse only the official-CPU port wiring from the existing diagnostic bench.
 old=(ROOT/'hw/efinity/prototypes/sapphire_memory/tb_cpu_diagnostic.sv').read_text()
 inst=old[old.index('    EfxSapphireSoc dut ('):old.index('    initial begin')]
 inst=inst.replace('pready=rng[4]','pready')
 inst=inst.replace('.io_apbSlave_0_PRDATA(32\'h0)', '.io_apbSlave_0_PRDATA(prdata)').replace('.io_apbSlave_0_PSLVERROR(1\'b0)','.io_apbSlave_0_PSLVERROR(pslverror)')
 for n in range(4):inst=inst.replace(f'.system_spi_0_io_data_{n}_read(1\'b0)',f'.system_spi_0_io_data_{n}_read(1\'b1)')
 control=(board/'assembly/cnn_apb_control.sv').read_text().split(');',1)[0]
 ports=re.findall(r'\b(input|output)\s+(?:wire|reg)\s*(?:\[[^]]+\])?\s*([^;\n]+)',control)
 # Parse the compact ANSI header one direction group at a time.
 header=control[control.index('(')+1:]
 groups=re.split(r'\b(input|output)\b',header)[1:];links=[]
 named={'clk':'clk','reset':'reset','paddr':'paddr','psel':'psel','penable':'penable','pwrite':'pwrite','pwdata':'pwdata','pready':'pready','prdata':'prdata','pslverror':'pslverror','enable':'enabled','load_valid':'loading_command','begin_valid':'begin_command','compute_crc32':'ccrc','dma_crc32':'dcrc'}
 for i in range(0,len(groups),2):
  direction,body=groups[i:i+2];body=re.sub(r'\b(wire|reg)\b|\[[^]]+\]','',body)
  for n in body.replace('\n','').split(','):
   n=n.strip()
   if n:
    target=named.get(n,"'0" if direction=="input" else "")
    links.append(f'.{n}({target})')
 tb='''`timescale 1ns/1ps
module tb_board_boot;
reg clk=0,reset=1; always #3.333333 clk=~clk;
wire sys_reset; wire awvalid,wvalid,wlast,bready,arvalid,rready;
wire [31:0] awaddr,wdata,araddr;wire [7:0] awid,awlen,arid,arlen;wire [2:0] awsize,arsize;wire [3:0] wstrb;
wire awready=0,wready=0,arready=0,bvalid=0,rvalid=0;wire [7:0] bid=0,rid=0;wire [31:0] rdata=0;
wire [15:0] paddr;wire psel,penable,pwrite,pready,pslverror,enabled,loading_command,begin_command;
wire [31:0] pwdata,prdata,ccrc,dcrc;
 integer cycles=0,apb_count=0,ocr_bytes=0,ocr_ff_bytes=0,spi_commands=0,external_requests=0,db_count=0;
wire dbfire=dut.system_cores_0_logic_cpu_dBus_cmd_valid&&dut.system_cores_0_dBus_cmd_ready;
wire [31:0] da=dut.system_cores_0_logic_cpu_dBus_cmd_payload_address;
wire [31:0] dd=dut.system_cores_0_logic_cpu_dBus_cmd_payload_data;
wire dw=dut.system_cores_0_logic_cpu_dBus_cmd_payload_wr;
wire [31:0] pc=dut.system_cores_0_logic_cpu_iBus_cmd_payload_address;
'''+inst+'\ncnn_apb_control control('+','.join(links)+');\n'+'''
initial begin repeat(20)@(negedge clk);reset=0;end
always @(posedge clk) begin
 cycles<=cycles+1;
 if(!reset&&!sys_reset)begin
  if(dbfire)begin
   db_count<=db_count+1;
   if(db_count<32)$display("BOOT_DB cycle=%0d addr=%h write=%b data=%h",cycles,da,dw,dd);
  end
  if(psel&&penable&&pready)begin apb_count<=apb_count+1;$display("BOOT_APB addr=%h write=%b data=%h",paddr,pwrite,pwdata);end
  if(dbfire&&dw&&da>=32'hf9000000&&da<32'hf9000c00)begin
   ocr_bytes<=ocr_bytes+1;if(dd[7:0]==8'hff)ocr_ff_bytes<=ocr_ff_bytes+1;
   if(da==32'hf9000000||da==32'hf9000bff)$display("BOOT_OCR cycle=%0d addr=%h data=%h",cycles,da,dd);
  end
  if(dbfire&&dw&&da==32'hf8014000)begin
   spi_commands<=spi_commands+1;if(spi_commands<16)$display("BOOT_SPI cycle=%0d command=%h",cycles,dd);
  end
  if(awvalid||arvalid)external_requests<=external_requests+1;
  if(cycles%100000==0)$display("BOOT_PROGRESS cycles=%0d pc=%h apb=%0d ocr=%0d",cycles,pc,apb_count,ocr_bytes);
 end
 if(cycles==CYCLE_LIMIT)begin
  $display("BOOT_OBSERVED cycles=%0d pc=%h apb=%0d enable=%b load=%b ccrc=%h dcrc=%h ocr_writes=%0d ff_writes=%0d spi_commands=%0d external_request_cycles=%0d",cycles,pc,apb_count,enabled,loading_command,ccrc,dcrc,ocr_bytes,ocr_ff_bytes,spi_commands,external_requests);
  $display("BOOT_TERMINAL_DB valid=%b ready=%b addr=%h write=%b transactions=%0d",dut.system_cores_0_logic_cpu_dBus_cmd_valid,dut.system_cores_0_dBus_cmd_ready,da,dw,db_count);
  $finish;
 end
end
endmodule
'''.replace('CYCLE_LIMIT',str(a.cycles))
 (root/'tb_board_boot.sv').write_text(tb)
 (root/'inputs.json').write_text(json.dumps(dict(board_inputs_sha256=sha(board/'inputs.json'),inputs=records,public_cpu_rtl_matches=True,scope='Actual unchanged ROM boot prerequisite, not full-board performance'),indent=2))
 subprocess.run([str(MS/'vlib.exe'),'work'],cwd=root,check=True,capture_output=True)
 cmd=[str(MS/'vlog.exe'),'-sv',str(root/MODEL.name),str(root/'cnn_apb_control.sv'),str(root/'tb_board_boot.sv')]
 c=subprocess.run(cmd,cwd=root,capture_output=True);(root/'compile.log').write_bytes(c.stdout+c.stderr);assert c.returncode==0,'See compile.log'
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 t=time.monotonic()
 with (root/'simulation.log').open('wb') as log:
  c=subprocess.run([str(MS/'vsim.exe'),'-c','-novopt','work.tb_board_boot','-do','do run.do'],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 body=(root/'simulation.log').read_text(errors='replace');rows=re.findall(r'BOOT_OBSERVED[^\r\n]+',body)
 result=dict(returncode=c.returncode,observation=rows,seconds=time.monotonic()-t,log_sha256=sha(root/'simulation.log'),full_frame_pass=False,production_rtl_modified=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 assert c.returncode==0 and len(rows)==1 and '** Error' not in body

if __name__=='__main__':main()
