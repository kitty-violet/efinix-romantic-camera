"""Short actual Sapphire CPU probe: unsupported CSR diagnosis and real CLINT time.

Only a diagnostic OCR image is supplied. No hierarchical writes/CPU substitute.
"""
import argparse,json,os,re,shutil,struct,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from observe_board_boot import MODEL
sys.path.insert(0,str(HERE/'board_integration'))
from boot_image import RV32,OCR

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sim-bin',type=Path,required=True);p.add_argument('--license-file',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 asm=RV32();asm.li(31,0xf8100000);asm.address(5,'trap');asm.emit((0x305<<20)|(5<<15)|(1<<12)|0x73,'csrw mtvec,x5')
 expected=[]
 for i,csr in enumerate((0xc00,0xb00)):
  asm.li(5,csr);asm.sw(5,31,0);asm.li(10,0);expected.append(OCR+len(asm.code)*4)
  asm.emit((csr<<20)|(2<<12)|(10<<7)|0x73,f'csrr x10,{csr:x}');asm.sw(10,31,4)
 asm.li(5,0xf8b0bff8);asm.li(6,3);asm.sw(6,31,0);asm.lw(10,5);asm.sw(10,31,16)
 for _ in range(64):asm.imm(0,0,0)
 asm.lw(10,5);asm.sw(10,31,20);asm.li(5,0x600df00d);asm.sw(5,31,24)
 asm.label('terminal');asm.jump('terminal');asm.label('trap')
 for csr,rd,off in ((0x342,12,8),(0x341,13,12)):
  asm.emit((csr<<20)|(2<<12)|(rd<<7)|0x73,f'csrr x{rd},{csr:x}');asm.sw(rd,31,off)
 asm.imm(13,13,4);asm.emit((0x341<<20)|(13<<15)|(1<<12)|0x73,'csrw mepc,x13');asm.emit(0x30200073,'mret')
 words=asm.finish();rom=bytearray(4096)
 for i,w in enumerate(words):struct.pack_into('<I',rom,4*i,w)
 for b in range(4):(root/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{b}.bin').write_text(''.join(f'{v:08b}\n' for v in rom[b::4]))
 (root/'boot.lst').write_text('\n'.join(f'{OCR+i*4:08x} {w:08x} {asm.text[i]}' for i,w in enumerate(words)))
 shutil.copy2(MODEL,root/MODEL.name)
 old=(HERE.parent/'sapphire_memory/tb_cpu_diagnostic.sv').read_text();inst=old[old.index('    EfxSapphireSoc dut ('):old.index('    initial begin')]
 tb='''`timescale 1ns/1ps
module tb_timer;
reg clk=0,reset=1;always #3.333333 clk=~clk;
wire sys_reset,awvalid,wvalid,wlast,bready,arvalid,rready;
wire [31:0] awaddr,wdata,araddr;wire [7:0] awid,awlen,arid,arlen;
wire [2:0] awsize,arsize;wire [3:0] wstrb;
wire awready=0,wready=0,arready=0,bvalid=0,rvalid=0;wire [7:0] bid=0,rid=0;wire [31:0] rdata=0;
wire [15:0] paddr;wire psel,penable,pwrite;wire [31:0] pwdata;wire pready=1;
integer cycles=0;
'''+inst+'''
initial begin repeat(20)@(negedge clk);reset=0;end
always @(posedge clk)begin
 cycles<=cycles+1;
 if(cycles>30000)$fatal(1,"timer probe timeout");
 if(!reset&&!sys_reset&&psel&&penable&&pready)begin
  if(!pwrite)$fatal(1,"unexpected APB read");
  $display("TIMER_PROBE addr=%h data=%h cycle=%0d",paddr,pwdata,cycles);$fflush();
  if(paddr==24&&pwdata==32'h600df00d)$finish;
 end
end
endmodule
'''
 (root/'tb_timer.sv').write_text(tb)
 env=os.environ.copy();env['SALT_LICENSE_SERVER']=str(a.license_file.resolve())
 def run(cmd,name):
  start=time.monotonic()
  with (root/name).open('wb') as log:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
  assert r.returncode==0,name
  return (root/name).read_text(errors='replace'),time.monotonic()-start
 run([a.sim_bin/'vlib.exe','work'],'vlib.log')
 run([a.sim_bin/'vlog.exe','-sv',MODEL.name,'tb_timer.sv'],'compile.log')
 body,seconds=run([a.sim_bin/'vsim.exe','-c','work.tb_timer','-do','onerror {quit -code 1}; run -all; quit -code 0'],'simulation.log')
 rows=[(int(x,16),int(y,16),int(z)) for x,y,z in re.findall(r'TIMER_PROBE addr=(\w+) data=(\w+) cycle=(\d+)',body)]
 causes=[v for k,v,c in rows if k==8];pcs=[v for k,v,c in rows if k==12]
 ts=[(v,c) for k,v,c in rows if k in (16,20)]
 assert causes==[2,2] and pcs==expected,(causes,pcs,expected)
 assert len(ts)==2 and ts[1][0]-ts[0][0]>64 and rows[-1][0:2]==(24,0x600df00d),rows
 result=dict(status='CPU_TIMER_DIAG_PASS',csr_traps=[dict(csr=hex(csr),mcause=code,mepc=hex(pc)) for csr,code,pc in zip((0xc00,0xb00),causes,pcs)],clint_address='0xf8b0bff8',timer_samples=ts,seconds=seconds,cpu_sha256=sha(MODEL),tb_sha256=sha(root/'tb_timer.sv'),log_sha256=sha(root/'simulation.log'),full_boot=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
