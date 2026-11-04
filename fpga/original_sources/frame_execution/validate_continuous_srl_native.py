"""Compare the actual four expanded native slices against their original cells."""
import json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree_safe import stop_launched_tree
from continuous_srl_register_native import OUT
root=ROOT/'tmp/frame-execution/continuous-srl-register-short1';assert not root.exists()
proof=json.loads((OUT/'result.json').read_text());assert proof['status']=='FOUR_SLOW_SRL_EXACT_GRAPH_CANDIDATE_PENDING_ACTUAL_SLICE_TEST'
assert sha(OUT/'native_source.v')==proof['source_sha256']
for p,h in proof['inputs'].items():assert sha(p)==h,p
root.mkdir()
for n in ('candidate_slice.sv','reference_slice.sv'):shutil.copy2(OUT/n,root/n)
for kind in ('efx_ff','efx_lut4','efx_srl8'):shutil.copy2(EF/'sim_models/verilog'/(kind+'.v'),root/(kind+'.v'))
tb=r'''
module tb;
 reg D=0,CLK=0,CE=0;reg [2:0] A=0;
 wire [3:0] expected,actual;
 reference_slice reference_dut(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(expected));
 candidate_slice actual_dut(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(actual));
 integer checks=0,p,j,k;
 task check_all;
 begin for(k=0;k<8;k=k+1)begin A=k;#1;
  if(expected!==actual)$fatal(1,"NATIVE_SRL_MISMATCH pattern=%d address=%d expected=%b actual=%b",p,A,expected,actual);
  checks=checks+1;
 end end endtask
 task step(input logic enable,input logic value);
 begin CLK=0;CE=enable;D=value;#1;CLK=1;#1;check_all;CLK=0;#1;end endtask
 initial begin
  #2;check_all;
  for(p=0;p<256;p=p+1)begin
   for(j=7;j>=0;j=j-1)step(1'b1,(p>>j)&1);
   step(1'b0,1'b0);step(1'b0,1'b1);
  end
  step(1'b1,1'bx);for(j=0;j<10;j=j+1)step(1'b1,j&1);
  step(1'b1,1'bz);for(j=0;j<10;j=j+1)step(1'b1,j&1);
  step(1'bx,1'b1);step(1'bz,1'b0);
  $display("PASS ACTUAL_NATIVE_SRL_EXPANSION checks=%0d cells=4 states256=1 addresses8=1 init00_ff=1 unknown_data=1",checks);
  $finish;
 end
 initial begin #1000000;$fatal(1,"NATIVE_SRL_SIM_TIMEOUT");end
endmodule
'''
(root/'tb.sv').write_text(tb)
files=[Path(__file__),OUT/'result.json',OUT/'native_source.v',OUT/'replacement_contracts.json',HERE/'windows_process_tree_safe.py',*root.glob('*.v'),*root.glob('*.sv')]
inputs=dict(files={str(p):sha(p) for p in files},known_address_required=True,actual_parent_map=proof['inputs'],all_four_Q7_unconnected=True,latency_cycles_added=0,hardware_access=False)
(root/'inputs.json').write_text(json.dumps(inputs,indent=2))
q=Path('E:/quatra/questa_fse/win64');env=dict(os.environ);env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';stages=[]
try:
 for name,cmd in [('vlib',[q/'vlib.exe','work']),('compile',[q/'vlog.exe','-sv','-timescale','1ns/1ps',*root.glob('*.v'),*root.glob('*.sv')]),('simulate',[q/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])]:
  start=time.monotonic()
  with (root/(name+'.log')).open('x') as log:
   child=subprocess.Popen(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:rc=child.wait(timeout=90)
   except subprocess.TimeoutExpired:(root/'timeout.json').write_text(json.dumps(stop_launched_tree(child)));raise
  body=(root/(name+'.log')).read_text();assert rc==0 and not re.search(r'\*\* (Error|Fatal)',body),(name,body[-2000:])
  stages.append(dict(stage=name,seconds=time.monotonic()-start))
 assert 'PASS ACTUAL_NATIVE_SRL_EXPANSION' in body
 for p,h in inputs['files'].items():assert sha(p)==h,p
 result=dict(status='ACTUAL_FOUR_NATIVE_SRL_EXPANSIONS_OFFICIAL_MODEL_SHORT_PASS',inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulate.log'),stages=stages,markers=re.findall(r'^# (PASS.*)$',body,re.M),known_address_required=True,hardware_access=False,physical_pass=False,board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
