"""Official-model bounded contingency test, with no board-map modification."""
import json,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree_safe import stop_launched_tree
root=ROOT/'tmp/frame-execution/srl8-register-alternative2';assert not root.exists();root.mkdir()
from native_local_drivers import raw_cells
from probe_full_native_split_map import params
mapped=ROOT/'tmp/frame-execution/connected-board-continuous-control1/outflow/ti60f225_oob.map.v'
mtext=mapped.read_text();cells=raw_cells(mtext);mp=params(mtext)
for cell in ['video_processing_pipeline/pipeline/cnn/core/dma/FEATURE_BIT[30].payload','video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/QUEUE[0].BIT[3].storage','video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/QUEUE[0].BIT[76].storage']:
 assert cells[cell][0]=='EFX_SRL8' and 'Q7' not in cells[cell][1]
 assert mp[cell]['CLK_POLARITY']==(1,1) and mp[cell]['CE_POLARITY']==(1,1) and mp[cell]['INIT'] in ((8,0),(8,255))
del mtext,cells,mp
source=HERE/'variants/continuous_run/srl8_register_alternative2.sv'
shutil.copy2(source,root/source.name)
for kind in ['efx_ff','efx_lut4','efx_srl8']:shutil.copy2(EF/'sim_models/verilog'/(kind+'.v'),root/(kind+'.v'))
tb=r'''
module tb;
 reg D=0,CLK=0,CE=0;reg [2:0] A=0;
 wire [2:0] oq,nq,ot,nt;
 EFX_SRL8 #(.INIT(8'h00)) o0(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(oq[0]),.Q7(ot[0]));
 EFX_SRL8 #(.INIT(8'hff)) o1(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(oq[1]),.Q7(ot[1]));
 EFX_SRL8 #(.INIT(8'ha5)) o2(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(oq[2]),.Q7(ot[2]));
 srl8_register_alternative2 #(.INIT(8'h00)) n0(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(nq[0]));
 srl8_register_alternative2 #(.INIT(8'hff)) n1(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(nq[1]));
 srl8_register_alternative2 #(.INIT(8'ha5)) n2(.D(D),.CLK(CLK),.CE(CE),.A(A),.Q(nq[2]));
 integer checks=0,p,j,k;
 task check_all;
 begin
  for(k=0;k<8;k=k+1)begin A=k;#1;
   if(oq!==nq)$fatal(1,"SRL_EQ mismatch pattern=%d addr=%d old=%b/%b new=%b/%b",p,A,oq,ot,nq,nt);
   checks=checks+1;
  end
 end endtask
 task step(input logic enable,input logic value);
 begin CLK=0;CE=enable;D=value;#1;CLK=1;#1;check_all;CLK=0;#1;end
 endtask
 initial begin
  #2;check_all;
  for(p=0;p<256;p=p+1)begin
   for(j=7;j>=0;j=j-1)step(1'b1,(p>>j)&1);
   step(1'b0,1'b0);step(1'b0,1'b1);
  end
  step(1'b1,1'bx);
  for(j=0;j<10;j=j+1)step(1'b1,j&1);
  step(1'b1,1'bz);
  for(j=0;j<10;j=j+1)step(1'b1,j&1);
  step(1'bx,1'b1);step(1'bz,1'b0);
  $display("PASS SRL_REGISTER_ALTERNATIVE checks=%0d init00_ff_a5=1 all_states=256 all_addresses=8 ce_hold=1 unknown_data=1",checks);
  $finish;
 end
 initial begin #1000000;$fatal(1,"HOST_BOUNDED_TEST_SIM_TIMEOUT");end
endmodule
'''
(root/'tb.sv').write_text(tb)
files=[Path(__file__),HERE/'validate_srl8_register_alternative.py',mapped,source,HERE/'windows_process_tree_safe.py',*root.glob('*.v'),*root.glob('*.sv')]
evidence={str(p):sha(p) for p in files};(root/'inputs.json').write_text(json.dumps(dict(evidence=evidence,current_board_unchanged=True,scope='Only Q output, source Q7 unconnected verified on all three current-map candidates. Identity LUT normalizes Z data to X without changing binary behavior. Only rising-edge active-high CE cases used by the measured three candidates; address is fully known. No assertion of X-address equality, routed speed improvement or board release.'),indent=2))
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
 assert 'PASS SRL_REGISTER_ALTERNATIVE' in body
 for p,h in evidence.items():assert sha(p)==h,p
 result=dict(status='OFFICIAL_SRL_REGISTER_ALTERNATIVE_KNOWN_ADDRESS_SHORT_PASS',stages=stages,inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulate.log'),markers=re.findall(r'^# (PASS.*)$',body,re.M),current_board_changed=False,physical_implementation_pass=False,board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
