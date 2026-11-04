"""Measure sustained source throughput with real official RAM primitives."""
import json,os,re,shutil
from run_units import ROOT,HERE,EF,run,sha
from pathlib import Path
MS=Path('E:/quatra/questa_fse/win64')
RUN=ROOT/'tmp/frame-execution/registered-source-rate2'

def main():
 assert not RUN.exists();RUN.mkdir()
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 src=HERE/'variants/fixed_source_elastic_test/tb_fixed_source_columns6.sv'
 text=src.read_text().replace('module tb_fixed_source_columns6;','module tb_registered_source_rate;')
 text=text.replace('row_pair_arena6 ram(.clk',"row_pair_arena6 ram(.aux_write_collision(6'd0),.clk")
 text=text.replace('job<173','job<1')
 before='   ncols=cmd_pw ? cmd_pixels:cmd_stride2 ? 2*cmd_pixels+1:cmd_pixels+2;'
 assert text.count(before)==1
 text=text.replace(before,'''   cmd_pw=1;cmd_stem=0;cmd_stride2=0;cmd_nn=0;
   cmd_pixels=320;cmd_rows=4;cmd_groups=4;cmd_group=0;
   cmd_output_x=0;cmd_source_width=1280;cmd_source_origin=0;cmd_storage_phase=0;
'''+before)
 text=text.replace("gate=6'($random(seed))|6'($random(seed));out_ready=randn(4)!=0;",'gate=63;out_ready=1;')
 anchor=' function integer randn(input integer n);'
 assert text.count(anchor)==1
 text=text.replace(anchor,''' integer cycles=0,first_cycle=-1,last_cycle=-1,command_cycle=-1;
 always @(posedge clk)begin
  cycles=cycles+1;
  if(cmd_valid && cmd_ready)command_cycle=cycles;
  if(verify && out_valid && out_ready)begin
   if(first_cycle<0)first_cycle=cycles;
   if(last_cycle>=0 && cycles!=last_cycle+1)$fatal(1,"SOURCE_SUSTAINED_RATE_GAP");
   last_cycle=cycles;
  end
 end
'''+anchor)
 index=text.index('  // Revoke held real RAM responses')
 text=text[:index]+'''  if(jobs!=1 || packets!=1280 || checked!=61440 || last_cycle-first_cycle!=1279)
   $fatal(1,"SOURCE_RATE_COVERAGE");
  $display("PASS REGISTERED_SOURCE_RATE packets=%0d checked_bytes=%0d first_after_command=%0d span=%0d initiation_interval=1",packets,checked,first_cycle-command_cycle,last_cycle-first_cycle);
  $finish;
 end
 initial begin #20000000;$fatal(1,"SOURCE_RATE_TIMEOUT");end
endmodule
'''
 tb=RUN/'tb_registered_source_rate.sv';tb.write_text(text)
 files=[HERE/'variants/fixed_split_registered_edges/fixed_source_columns6.sv',
  ROOT/'tmp/frame-execution/connected-board-split-outputs1/assembly/row_pair_arena6.sv',EF/'sim_models/verilog/efx_dpram10.v']
 records=[]
 for f in files:
  dst=RUN/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
 (RUN/'inputs.json').write_text(json.dumps(dict(inputs=records,tb_sha256=sha(tb),original_tb_sha256=sha(src),runner_sha256=sha(__file__)),indent=2))
 run([MS/'vlib.exe','work'],RUN,'vlib.log',timeout=20)
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records],tb],RUN,'compile.log',timeout=30)
 assert 'Errors: 0, Warnings: 0' in body
 body,seconds=run([MS/'vsim.exe','-c','work.tb_registered_source_rate','-do','run -all; quit -code 0'],RUN,'simulation.log',timeout=90)
 assert 'Errors: 0, Warnings: 0' in body
 lines=re.findall(r'^# (PASS REGISTERED_SOURCE_RATE.*)$',body,re.M);assert len(lines)==1
 for r in records:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
 result=dict(status='REGISTERED_SOURCE_CONTINUOUS_RATE_PASS',lines=lines,seconds=seconds,hardware_access=False,
  scope='1280 C8 column tokens without RAM/MAC stalls, not full CNN FPS or board throughput.')
 (RUN/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
