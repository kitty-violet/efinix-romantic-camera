"""Official four-state numeric/transaction gates for registered boundaries.

RQ also compares every cycle against the frozen production implementation;
its oracle still checks INT32/Q17 values, backpressure and eight reset points.
"""
import concurrent.futures,json,os,re,shutil,sys,time
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha

RUN=ROOT/'tmp/frame-execution/split-registered-edges-shorts1'
BASE=ROOT/'tmp/frame-execution/connected-board-split-outputs1/assembly'
VAR=HERE/'variants/fixed_split_registered_edges'
MS=Path('E:/quatra/questa_fse/win64')

def rq_test():
 text=(HERE/'variants/fixed_dsp_round_feedback_test/tb_fixed_split_rq.sv').read_text()
 text=text.replace('module tb_fixed_split_rq;','module tb_fixed_split_rq #(parameter SHARED=1);')
 text=text.replace(' reg [287:0] feedback_value=0;',' wire [287:0] feedback_value,ref_feedback_value;\n wire feedback_post_selected,ref_feedback_post_selected;')
 old=' fixed_rq32_mixed dut(.*);';assert text.count(old)==1
 text=text.replace(old,''' fixed_rq32_mixed #(.REGISTER_SCALE_CLASS(1),.SHARED_FEEDBACK(SHARED)) dut(.*);
 wire [2:0] ref_in_ready,ref_out_valid,ref_out_bad;
 wire [767:0] ref_out_data;wire [26:0] ref_out_x;wire ref_busy,ref_error;
 fixed_rq32_mixed_reference #(.REGISTER_SCALE_CLASS(1),.SHARED_FEEDBACK(SHARED)) reference(
  .in_ready(ref_in_ready),.out_valid(ref_out_valid),.out_bad(ref_out_bad),
  .out_data(ref_out_data),.out_x(ref_out_x),.busy(ref_busy),.error(ref_error),
  .feedback_value(ref_feedback_value),.feedback_post_selected(ref_feedback_post_selected),.*);
 for(genvar f=0;f<32;f=f+1)begin
  assign feedback_value[f*9+:9]=feedback_post_selected ? post_value[f*9+:9]:{align_value[f*8+7],align_value[f*8+:8]};
  assign ref_feedback_value[f*9+:9]=ref_feedback_post_selected ? post_value[f*9+:9]:{align_value[f*8+7],align_value[f*8+:8]};
 end
 integer equivalent_cycles=0;
 always @(posedge clk)if(!reset && !cancel)begin
  if({in_ready,out_valid,busy,error}!=={ref_in_ready,ref_out_valid,ref_busy,ref_error})
   $fatal(1,"RQ_REGISTER_BOUNDARY_CONTROL_MISMATCH");
  for(integer f=0;f<3;f=f+1)if(out_valid[f] &&
   {out_data[f*256+:256],out_bad[f],out_x[f*9+:9]}!=={ref_out_data[f*256+:256],ref_out_bad[f],ref_out_x[f*9+:9]})
   $fatal(1,"RQ_REGISTER_BOUNDARY_DATA_MISMATCH");
  equivalent_cycles=equivalent_cycles+1;
 end''')
 for token in ('   in_valid=7;out_ready=0;','   payload(0,450);in_valid=1;'):
  assert text.count(token)==1;text=text.replace(token,'   @(negedge clk);\n'+token)
 text=text.replace('  $display("PASS FIXED_SPLIT_RQ',
  '  if(equivalent_cycles<10000)$fatal(1,"MISSING_EQUIVALENCE_COVERAGE");\n  $display("PASS RQ_BOUNDARY_EQUIVALENCE cycles=%0d shared=%0d",equivalent_cycles,SHARED);\n  $display("PASS FIXED_SPLIT_RQ')
 return text

def worker(kind):
 folder=RUN/kind;folder.mkdir();records=[]
 def snap(src,name=None,text=None):
  dest=folder/(name or src.name)
  if text is None:shutil.copy2(src,dest)
  else:dest.write_text(text,encoding='utf8')
  records.append(dict(source=str(src),snapshot=str(dest),source_sha256=sha(src),sha256=sha(dest)))
 if kind=='rq':
  snap(VAR/'fixed_rq32_mixed.sv')
  src=BASE/'fixed_rq32_mixed.sv'
  ref=src.read_text().replace('fixed_rq32_mixed','fixed_rq32_mixed_reference').replace('fixed_rq_predecoded_pipeline','fixed_rq_predecoded_reference')
  snap(src,'rq_reference.sv',ref)
  snap(HERE/'variants/fixed_dsp_round_feedback_test/tb_fixed_split_rq.sv',text=rq_test())
  models=['efx_dsp48.v','efx_srl8.v'];top='tb_fixed_split_rq';marker='PASS FIXED_SPLIT_RQ'
  cases=[['-gSHARED=0'],['-gSHARED=1']]
 else:
  snap(VAR/'fixed_source_columns6.sv');snap(BASE/'row_pair_arena6.sv')
  snap(HERE/'variants/fixed_source_elastic_test/tb_fixed_source_columns6.sv')
  models=['efx_dpram10.v'];top='tb_fixed_source_columns6';marker='PASS FIXED_SOURCE_COLUMNS6';cases=[[]]
 for name in models:snap(EF/'sim_models/verilog'/name)
 (folder/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 scratch=folder/'temp';scratch.mkdir()
 # Each task has its own library; the supervisor bounds the whole process tree.
 run([MS/'vlib.exe','work'],folder,'vlib.log',timeout=20)
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],folder,'compile.log',timeout=30)
 assert 'Errors: 0, Warnings: 0' in body,body[-2000:]
 results=[]
 for i,args in enumerate(cases):
  body,seconds=run([MS/'vsim.exe','-c','work.'+top,*args,'-do','onerror {quit -code 1}; run -all; quit -code 0'],folder,f'simulation{i}.log',timeout=120)
  assert marker in body and 'Errors: 0, Warnings: 0' in body,body[-2500:]
  if kind=='rq':assert 'PASS RQ_BOUNDARY_EQUIVALENCE' in body
  results.append(dict(case=args,seconds=seconds,lines=re.findall(r'^# (PASS .*)$',body,re.M),log_sha256=sha(folder/f'simulation{i}.log')))
 for r in records:assert sha(r['source'])==r['source_sha256'] and sha(r['snapshot'])==r['sha256']
 result=dict(status='REGISTERED_BOUNDARY_SHORT_PASS',kind=kind,results=results)
 (folder/'result.json').write_text(json.dumps(result,indent=2));return result

def main():
 assert not RUN.exists();RUN.mkdir()
 os.environ['MGLS_LICENSE_FILE']='E:/quatra/LR-195771_License.dat'
 os.environ['LM_LICENSE_FILE']='E:/quatra/LR-195771_License.dat'
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(worker,kind) for kind in ('rq','source')]
  results=[f.result() for f in futures]
 result=dict(status='RQ_CYCLE_EQUIVALENT_AND_SOURCE_NUMERIC_SHORTS_PASS',results=results,
  resource_fit=False,pnr=False,board_release=False,hardware_access=False)
 (RUN/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  if RUN.exists():(RUN/'failure.json').write_text(json.dumps(dict(error=repr(exc)),indent=2))
  raise
