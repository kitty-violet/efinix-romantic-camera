"""Exercise all 1024 budget encodings on the actual admission/saturation cone."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from continuous_control_board import BOARD
from continuous_control_native import net,connections,VARIANT
from audit_continuous_vectors import expand_graph
from native_local_drivers import CELL,render_token
from camera_joint_native import OUT
from control_command_repair import once
from validate_continuous_run_control import ADMISSION_TB
from windows_process_tree import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 src=BOARD/'outflow/ti60f225_oob.map.v';text=src.read_text();g=expand_graph(text);dr,ld=connections(g)
 try:
  cuts={net('process_clk_150'):'clk',net('w_arstn'):'!reset',net('fixed_boot_run_allowed'):'permit',net('cnn_start_request'):'start_request'}
  cuts.update({net('preview_frame_budget',i):f'requested_frames[{i}]' for i in range(10)})
  outputs=[net('video_processing_pipeline/compute_start')]+[net('preview_admitted',i) for i in range(10)]+[net('cnn_admitted',i) for i in range(5)]
  owned={dr[net('preview_admitted',i)] for i in range(10)}
  assert all(g[n][0]=='EFX_FF' for n in owned)
  selected={dr[t] for t in outputs};todo=list(selected)
  while todo:
   n=todo.pop();kind,ps=g[n]
   for pin,ts in ps.items():
    if pin in OUT[kind]:continue
    for t in ts:
     if t[0]=='literal' or t in cuts:continue
     d=dr.get(t);assert d is not None,(n,pin,t)
     assert g[d][0] in ('EFX_LUT4','EFX_ADD') or d in owned,(n,pin,t,d)
     if d not in selected:selected.add(d);todo.append(d)
  cells='\n'.join(m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in selected)
  parameters='\n'.join(m[0] for m in re.finditer(r'^\s*defparam\s+(\S+?)\s*\.\w+\s*=[^;]*;',text,re.M) if m[1].lstrip('\\') in selected)
  declarations='\n'.join(re.findall(r'^\s*wire\b[^;]*;',text,re.M))
  assigns='\n'.join(f'assign {render_token(t)}={v};' for t,v in cuts.items())
  assigns+='\nassign start_accept=\\video_processing_pipeline/compute_start ;\nassign admitted=preview_admitted;\nassign diagnostic=cnn_admitted;'
  (root/'mapped.sv').write_text('module mapped_admission(input wire clk,reset,permit,start_request,input wire [9:0] requested_frames,output wire start_accept,output wire [9:0] admitted,output wire [4:0] diagnostic);\n'+declarations+'\n'+assigns+'\n'+cells+'\n'+parameters+'\nendmodule\n')
  shutil.copy2(VARIANT/'cnn_frame_admission.sv',root/'cnn_frame_admission.sv')
  t=once(ADMISSION_TB,' cnn_frame_admission dut(.*);',""" wire map_accept;wire [9:0] map_admitted;wire [4:0] diagnostic;
 cnn_frame_admission dut(.*);
 mapped_admission mapped(.clk(clk),.reset(reset),.permit(permit),.start_request(start_request),.requested_frames(requested_frames),.start_accept(map_accept),.admitted(map_admitted),.diagnostic(diagnostic));
 integer checks=0;
 always @(negedge clk)begin #2;
  if({start_accept,admitted}!=={map_accept,map_admitted})
   $fatal(1,"ACTUAL_ADMISSION budget=%d rtl=%d map=%d",requested_frames,admitted,map_admitted);
  if(diagnostic!==(admitted>31?5'd31:admitted[4:0]))$fatal(1,"SATURATED_DIAGNOSTIC");
  checks=checks+1;
 end
""")
  t=once(t,'$finish;',' $display("PASS MAPPED_ADMISSION comparisons=%0d all_1024=1 saturated31=1",checks);$finish;')
  (root/'tb.sv').write_text(t)
  for kind in ('EFX_FF','EFX_LUT4','EFX_ADD'):shutil.copy2(EF/'sim_models/verilog'/(kind.lower()+'.v'),root/(kind.lower()+'.v'))
  files={str(f):sha(f) for f in [Path(__file__),HERE/'audit_continuous_vectors.py',HERE/'validate_continuous_run_control.py',src,*root.glob('*.sv'),*root.glob('*.v')]}
  (root/'inputs.json').write_text(json.dumps(dict(files=files,map_sha256=sha(src),cells=sorted(selected),cuts={render_token(t):v for t,v in cuts.items()}),indent=2))
  q=Path('E:/quatra/questa_fse/win64');env=dict(os.environ);env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';stages=[]
  for stage,cmd in [('vlib',[q/'vlib.exe','work']),('compile',[q/'vlog.exe','-sv','-timescale','1ns/1ps',*root.glob('*.v'),*root.glob('*.sv')]),('simulate',[q/'vsim.exe','-c','work.tb_preview_admission','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])]:
   begin=time.monotonic()
   with (root/(stage+'.log')).open('x') as log:
    child=subprocess.Popen(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:rc=child.wait(timeout=90)
    except subprocess.TimeoutExpired:(root/'timeout.json').write_text(json.dumps(stop_launched_tree(child)));raise
   log=(root/(stage+'.log')).read_text();assert rc==0 and not re.search(r'\*\* (Error|Fatal)',log),(stage,log[-2000:])
   stages.append(dict(stage=stage,seconds=time.monotonic()-begin))
  assert 'PASS MAPPED_ADMISSION' in log
  for f,h in files.items():assert sha(f)==h,f
  result=dict(status='ACTUAL_MAP_ADMISSION_ALL_BUDGETS_AND_SATURATION_PASS',map_sha256=sha(src),inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulate.log'),stages=stages,markers=re.findall(r'^# (PASS.*)$',log,re.M),hardware_access=False,board_release=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
