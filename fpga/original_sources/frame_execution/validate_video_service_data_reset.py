"""Test data-only reset removal against numerical and residual protocol oracles."""
import argparse,json,os,re,shutil,time
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
from map_video_service_safe import run

def once(s,a,b):
 assert s.count(a)==1,a
 return s.replace(a,b)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 old=ROOT/'tmp/frame-execution/video-service-rq-fast-short1'
 proof=json.loads((old/'result.json').read_text());assert proof['status']=='GUARDED_FAST_RQ_INT32_NUMERIC_QUEUE_CANCEL_PASS'
 for f,h in proof['inputs'].items():assert sha(f)==h
 base=ROOT/'tmp/frame-execution/connected-board-video-service-probe31-ready-branch'
 rq=(old/'fixed_rq32_mixed.sv').read_text()
 rq=once(rq,'out_valid<=0;result<=0;multiplier_range_error<=0;', 'out_valid<=0;multiplier_range_error<=0;')
 line='    result<=sat2 ? ((neg2 && relu2) ? 8\'sd0:neg2 ? -8\'sd128:8\'sd127):rounded(product2[36:0],relu2);'
 rq=once(rq,line,'')
 end=rq.rfind('endmodule')
 rq=rq[:end]+' always @(posedge clk)if(v2)begin\n'+line+'\n end\n'+rq[end:]
 (out/'fixed_rq32_mixed.sv').write_text(rq)
 shutil.copy2(old/'tb.sv',out/'tb_rq.sv')
 skip_source=base/'assembly/fixed_skip_vectors4.sv';skip=skip_source.read_text()
 skip=once(skip,'skip_held<=0;skip_data<=0;read_x<=0;', 'skip_held<=0;read_x<=0;')
 skip=once(skip,'skip_held<=1;skip_data<=skip;skip_x<=read_x;', 'skip_held<=1;skip_x<=read_x;')
 skip=once(skip,'endmodule','always @(posedge clk)if(capture_skip)skip_data<=skip;\nendmodule')
 (out/'fixed_skip_vectors4.sv').write_text(skip)
 tb=HERE/'variants/fixed_skip_credit_test/tb_skip_credit.sv';test=tb.read_text()
 test=once(test,'.OFFSET_PREAPPLIED(1)) dut(','.OFFSET_PREAPPLIED(1),.REGISTERED_FEEDBACK_SELECT(1)) dut(')
 test=once(test,'.feedback_value(feedback_value),','.feedback_value(feedback_value),.feedback_post_selected(post_ready),')
 (out/'tb_skip_credit.sv').write_text(test)
 for n in ('efx_srl8.v','efx_dsp48.v','efx_lut4.v'):shutil.copy2(EF/'sim_models/verilog'/n,out/n)
 inputs={str(f):sha(f) for f in [old/'result.json',old/'fixed_rq32_mixed.sv',old/'tb.sv',skip_source,tb,Path(__file__),*out.iterdir()]}
 (out/'inputs.json').write_text(json.dumps(inputs,indent=2))
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 t=time.monotonic();result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  run([ms/'vlib.exe','work'],out,'vlib.log',30)
  log,_=run([ms/'vlog.exe','-sv',*out.glob('*.v'),*out.glob('*.sv')],out,'compile.log',60)
  assert 'Errors: 0, Warnings: 0' in log
  lines=[]
  for top,marker in [('tb_fixed_split_rq','PASS GUARDED_FAST_RQ'),('tb_skip_credit','PASS SKIP_CREDIT')]:
   log,_=run([ms/'vsim.exe','-c','work.'+top,'-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],out,top+'.log',150)
   matches=re.findall('^# ('+marker+r'.*)$',log,re.M);assert len(matches)==1 and 'Errors: 0, Warnings: 0' in log;lines+=matches
  for f,h in inputs.items():assert sha(f)==h
  result.update(status='DATA_ONLY_RESET_PROTOCOL_AND_NUMERIC_SHORT_PASS_NOT_NATIVE_PROOF',lines=lines)
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-t,inputs_sha256=sha(out/'inputs.json'))
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
