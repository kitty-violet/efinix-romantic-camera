"""Reuse the passed strict V4 fixture; replace only its external AXI stimulus."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--compile-only',action='store_true');p.add_argument('--bridge',type=Path)
 p.add_argument('--concurrent',action='store_true');a=p.parse_args()
 stimulus=HERE/('tb_ddr516_concurrent.sv' if a.concurrent else 'tb_ddr516_traffic_matrix.sv')
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
 base=ROOT/'tmp/frame-execution/ddr516-v4-wrapper1'
 previous=json.loads((base/'result.json').read_text());meta=json.loads((base/'inputs.json').read_text())
 assert previous['status']=='DDR516_STRICT_REFERENCE_PASS' and previous['inputs_sha256']==sha(base/'inputs.json')
 records=[]
 for row in meta['inputs']:
  src=Path(row['snapshot']);assert sha(src)==row['sha256'];dst=root/src.name
  assert not dst.exists(),src
  shutil.copy2(src,dst)
  if src.name=='memory_checker_axi.v':
   dst.write_text(stimulus.read_text())
  if src.name=='ddr516_axi_address_bridge.sv' and a.bridge:
   shutil.copy2(a.bridge.resolve(),dst)
  if src.name=='example_top.v':
   text=dst.read_text();assert text.count('`include "ddr516_axi_diagnostics.svh"')==1
   text=text.replace('`include "ddr516_axi_diagnostics.svh"','')
   text=text.replace('measured_writes!=256 || measured_reads!=256',
    'measured_writes!=768 || measured_reads!=896' if a.concurrent else 'measured_writes!=640 || measured_reads!=640')
   text=text.replace('measured_aw!=4 || measured_ar!=4',
    'measured_aw!=6 || measured_ar!=7' if a.concurrent else 'measured_aw!=7 || measured_ar!=7')
   text=text.replace('bytes_each_direction=%0d",measured_writes,measured_reads,measured_aw,measured_ar,measured_reads*16)',
    'write_bytes=%0d read_bytes=%0d",measured_writes,measured_reads,measured_aw,measured_ar,measured_writes*16,measured_reads*16)')
   monitor='''
integer native_write_left=0;
always @(posedge clk)if(w_nrst && w_cal_done)begin
 if(genblk1.inst_ddr3_axi.avalid && genblk1.inst_ddr3_axi.aready && genblk1.inst_ddr3_axi.atype)
  native_write_left=integer'(genblk1.inst_ddr3_axi.alen)+1;
 if(genblk1.inst_ddr3_axi.wvalid && genblk1.inst_ddr3_axi.wready)begin
  if(native_write_left<=0)$fatal(1,"MATRIX native write without address");
  native_write_left=native_write_left-1;
 end
 if(genblk1.inst_ddr3_axi.bvalid && genblk1.inst_ddr3_axi.bready)begin
  $display("MATRIX_NATIVE_B remaining=%0d time=%t",native_write_left,$time);$fflush();
  if(native_write_left!=0)$fatal(1,"MATRIX premature native B response");
 end
end
'''
   assert text.count('endmodule')==1
   text=text.replace('endmodule',monitor+'\nendmodule')
   dst.write_text(text)
  records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
 data=dict(inputs=records,base=str(base),runner_sha256=sha(__file__),stimulus_sha256=sha(stimulus),concurrent=a.concurrent,
           cases=[dict(beats=b,gap=g,address=ad,idle_before_ns=idle) for b,g,ad,idle in
            [(64,0,0,0),(64,0,0x6000000,0),(128,0,0x6000000,0),(64,11,0x6000000,0),
             (128,11,0x6000000,0),(64,0,0x6000000,200000),(128,11,0x6000000,200000)]],
           physical_pass=False,full_board=False,calibration_forced=False,
           bridge_source=str(a.bridge.resolve()) if a.bridge else None)
 if a.concurrent:data['cases']=['prime/read128','read128 periodic backpressure',
   'overlap continuous W/R','overlap sparse W/continuous R',
   'overlap continuous W/backpressured R','overlap sparse W/backpressured R',
   'overlap line-like W/continuous R']
 (root/'inputs.json').write_text(json.dumps(data,indent=2))
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 # Compile just the proven fixture modules and the new external stimulus.
 sources=[root/n for n in ['ddr3.v','efx_iddr.v','efx_oddr.v','efx_srl8.v','memory_checker_axi.v','ddr516_reference.sv',
  'ddr516_axi_address_bridge.sv','ddr516_v4_controller.sv','example_top.v','ddr516_rx_retime.sv','EFX_GPIO_V3.v','EFX_PLL_V3_FAST_PROTOCOL.v','tb.sv']]
 assert all(f.exists() for f in sources)
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 commands=[('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16','+incdir+'+str(root),*sources])]
 if not a.compile_only:commands.append(('simulation',[sim/'vsim.exe','-c','work.efx_ddr3_soft_controller_tb','-do','run.do']))
 start=time.monotonic();stages=[];body=''
 for name,cmd in commands:
  with (root/(name+'.log')).open('wb') as out:
   proc=subprocess.Popen(list(map(str,cmd)),cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
   (root/'active_process.json').write_text(json.dumps(dict(pid=proc.pid,command=list(map(str,cmd)),stage=name,started=time.time())))
   ret=proc.wait()
  body=(root/(name+'.log')).read_text(errors='replace');stages.append(dict(stage=name,returncode=ret))
  if ret or re.search(r'\*\* (?:Fatal|Error)|\bERROR:',body):break
 passed=(len(stages)==len(commands) and all(s['returncode']==0 for s in stages) and not re.search(r'\*\* (?:Fatal|Error)|\bERROR:',body))
 marker='CONCURRENT_PASS' if a.concurrent else 'MATRIX_PASS'
 if not a.compile_only:passed=passed and 'DDR3 Controller PASS' in body and len(re.findall(r'^# '+marker,body,re.M))==7
 for row in records:assert sha(row['snapshot'])==row['sha256']
 result=dict(status=('DDR516_MATRIX_COMPILE_PASS' if a.compile_only else 'DDR516_TRAFFIC_MATRIX_PASS') if passed else 'DDR516_TRAFFIC_MATRIX_FAIL',
  seconds=time.monotonic()-start,stages=stages,markers=[l for l in body.splitlines() if 'MATRIX_' in l or 'CONCURRENT_' in l or '** Fatal' in l],
  inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/(stages[-1]['stage']+'.log')),full_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
