"""Fixed 100us native RTL execution in Questa, no vendor peripheral load."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from run_units import ROOT,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
 wrapper=a.native.resolve()/'cnn_video_pipeline_dpi.sv';head=wrapper.read_text().split(') (',1)[1].split(');',1)[0]
 ports=[(d,int(w)+1 if w else 1,n)for d,w,n in re.findall(r'\b(input|output) logic (?:\[(\d+):0\]\s+)?\s*(\w+)',head)]
 tb='`timescale 1ns/1ps\nmodule tb;\n'
 for d,w,n in ports:tb+=f'{"reg" if d=="input" else "wire"} [{w-1}:0] {n}'+('=0;\n'if d=='input'else';\n')
 tb+='''always #3.333 process_clk=~process_clk;
always #6.734 display_clk=~display_clk;
cnn_video_pipeline #(.CPU_DDR_BASE(32'h04000000)) dut(.*);
initial begin reset=1;#200;reset=0;#99800;$display("PASS NATIVE_BRIDGE_IDLE_100US");$finish;end
endmodule
'''
 (root/'tb.sv').write_text(tb);sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env.update(SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat',PATH='E:/fpga/tools/verilator/ucrt64/bin;'+env['PATH'])
 for stage,cmd in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv',wrapper,root/'tb.sv']),('simulation',[sim/'vsim.exe','-c','-sv_lib',a.native.resolve()/'obj/libboardcnn','work.tb','-do','onbreak {quit -code 1};run -all;quit -code 0'])]:
  start=time.monotonic()
  with(root/(stage+'.log')).open('wb')as f:subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 elapsed=time.monotonic()-start;assert 'PASS NATIVE_BRIDGE_IDLE_100US' in(root/'simulation.log').read_text()
 result=dict(status='NATIVE_BRIDGE_IDLE_100US_PASS',wall_seconds=elapsed,native=str(a.native.resolve()),log_sha256=sha(root/'simulation.log'))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
