"""Observe real CPU posted APB writes versus OCR mailbox completion."""
import argparse,json,os,re,shutil,struct,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from validate_real_boot import make_tb
sys.path.insert(0,str(HERE/'board_integration'))
from boot_image import RV32,OCR
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--barrier',action='store_true');a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 ref=ROOT/'tmp/frame-execution/real-boot-questa2';records=[]
 for src in ref.iterdir():
  if src.suffix in ('.v','.sv','.svh','.vh') and src.name!='tb_real_boot.sv':
   shutil.copy2(src,root/src.name);records.append(dict(source=str(src),sha256=sha(src)))
 asm=RV32();asm.li(31,0xf8100000);asm.li(30,0xf9000f00);asm.li(16,0xf8b0bff8);asm.lw(24,16)
 asm.imm(5,0,3);asm.sw(5,31,8)
 if a.barrier:asm.lw(6,31,8)
 asm.imm(5,0,4);asm.sw(5,30,4);asm.lw(6,31,8);asm.sw(6,30,0x28);asm.label('wait');asm.jump('wait')
 rom=bytearray(4096)
 for i,w in enumerate(asm.finish()):struct.pack_into('<I',rom,4*i,w)
 for b in range(4):(root/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{b}.bin').write_text(''.join(f'{v:08b}\n' for v in rom[b::4]))
 tb=make_tb();tb=tb[:tb.index('initial begin\n if($value$plusargs')]+'''
initial begin repeat(20)@(negedge clk);reset=0;
 wait(phase==4);$display("ORDER_MAILBOX enable=%b cycle=%0d",enable,cycles);
 repeat(100)@(negedge clk);
 if(enable!==1)$fatal(1,"enable never reached");
 $display("PASS ORDER_DIAG enable=%b cycle=%0d",enable,cycles);$finish;
end
always @(posedge clk)begin
 if(psel&&penable&&pready)$display("ORDER_APB cycle=%0d addr=%h wr=%b data=%h",cycles,paddr,pwrite,pwdata);
 if(cpu_store&&cpu_da==32'hf9000f28)$display("ORDER_READBACK data=%h cycle=%0d",cpu_dd,cycles);
end
endmodule
'''
 (root/'tb_real_boot.sv').write_text(tb);sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 for name,cmd in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps',*[f.name for f in root.iterdir() if f.suffix in ('.sv','.v')]]),('simulation',[sim/'vsim.exe','-c','work.tb_real_boot','-do','onerror {quit -code 1}; run -all; quit -code 0'])]:
  with (root/(name+'.log')).open('wb') as log:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
  assert r.returncode==0,name
 body=(root/'simulation.log').read_text(errors='replace');assert 'PASS ORDER_DIAG' in body
 assert ('ORDER_MAILBOX enable=1' if a.barrier else 'ORDER_MAILBOX enable=0') in body
 result=dict(observations=re.findall(r'^# ((?:ORDER_|PASS ORDER)[^\n]+)',body,re.M),inputs=records,log_sha256=sha(root/'simulation.log'))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['observations']))
if __name__=='__main__':main()
