"""Check public sparse-memory bookkeeping against the unchanged Micron model.

No PHY result is claimed. The negative control must expose stale data after
the actual reset event; normal, X/Z address and bank-erase checks stay exact.
"""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,MS,sha
from prepare_board_frame_run import fast_memory

TB='''`timescale 1ns/1ps
module tb_lookup;
 parameter NEGATIVE=0;
 reg rst=0;wire [15:0] dq0,dq1;wire [1:0] dqs0,dqs1,dqsn0,dqsn1;
 wire [1:0] dm0=0,dm1=0;wire [1:0] td0,td1;
 __INSTANCES__
 reg [127:0] want,got,data;integer checks=0,seed=2671;
 task compare(input [2:0] bank,input [13:0] row,input [9:0] col);
 begin
  original.memory_read(bank,row,col,want);candidate.memory_read(bank,row,col,got);
  if(want!==got)$fatal(1,"LOOKUP_MISMATCH checks=%0d b=%h row=%h col=%h old=%h new=%h",checks,bank,row,col,want,got);
  if(original.memory_used!==candidate.memory_used)$fatal(1,"LOOKUP_USED mismatch");
  checks++;
 end endtask
 task write_both(input [2:0] bank,input [13:0] row,input [9:0] col,input [127:0] value);
 begin original.memory_write(bank,row,col,value);candidate.memory_write(bank,row,col,value);end
 endtask
 initial begin
  #200001;rst=1;#20;
  for(integer i=0;i<256;i++)begin
   data={$random(seed),$random(seed),$random(seed),$random(seed)};
   write_both(i%8,i/8,8*(i%128),data);compare(i%8,i/8,8*(i%128));
  end
  // Unknown addresses must not alias a known two-state cache key.
  if(!NEGATIVE)begin
   write_both(0,0,0,128'habc);write_both(0,14'bx,0,128'hdef);
   compare(0,0,0);compare(0,14'bx,0);compare(0,14'bz,0);
   original.erase_banks(8'h12);candidate.erase_banks(8'h12);
   for(integer i=0;i<256;i++)compare(i%8,i/8,8*(i%128));
  end
  // Exercise the real model reset path, including its NBA memory_used clear.
  rst=0;#200001;rst=1;#20;
  compare(0,0,0);
  write_both(7,99,16,128'h13579);compare(7,99,16);compare(0,0,0);
  write_both(0,0,0,128'hfedcba);compare(0,0,0);
  $display("PASS MICRON_LOOKUP checks=%0d",checks);$finish;
 end
endmodule
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sim-bin',type=Path,default=MS);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    vendor=EF/'ipm/ip/efx_ddr3_soft_controller/testbench/ddr3.v';source=vendor.read_text()
    frozen=ROOT/'tmp/frame-execution/ddr516-turnaround-sharedio1/ddr3.v'
    fixed=fast_memory(source)
    for name,body in [('original',source),('fixed',fixed),('old',frozen.read_text())]:
        assert len(re.findall(r'\bmodule ddr3\b',body))==1
        (root/(name+'.sv')).write_text(re.sub(r'\bmodule ddr3\b','module '+name,body,count=1))
    params=ROOT/'tmp/frame-execution/full-board-vendor-probe3/2048Mb_ddr3_parameters.vh'
    (root/params.name).write_bytes(params.read_bytes());(root/'ddr3_device_ID.vh').write_text('`define sg125\n')
    port='(.rst_n(rst),.ck(1\'b0),.ck_n(1\'b1),.cke(1\'b0),.cs_n(1\'b1),.ras_n(1\'b1),.cas_n(1\'b1),.we_n(1\'b1),.dm_tdqs(dm{n}),.tdqs_n(td{n}),.ba(3\'b0),.addr(14\'b0),.dq(dq{n}),.dqs(dqs{n}),.dqs_n(dqsn{n}),.odt(1\'b0));'
    for name,model in [('positive','fixed'),('negative','old')]:
        instances='original #(.DEBUG(0),.MEM_BITS(10)) original'+port.format(n=0)+'\n'+model+' #(.DEBUG(0),.MEM_BITS(10)) candidate'+port.format(n=1)
        (root/(name+'_tb.sv')).write_text(TB.replace('__INSTANCES__',instances).replace('module tb_lookup','module tb_'+name))
    inputs=dict(vendor_sha256=sha(vendor),fixed_sha256=__import__('hashlib').sha256(fixed.encode()).hexdigest(),
                negative_model_sha256=sha(frozen),runner_sha256=sha(__file__),files={f.name:sha(f) for f in root.iterdir()},simulator_bin=str(a.sim_bin.resolve()))
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';sim=a.sim_bin.resolve()
    def run(name,cmd):
        start=time.monotonic()
        with(root/(name+'.log')).open('wb')as log:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
        return r.returncode,(root/(name+'.log')).read_text(errors='replace'),time.monotonic()-start
    for name,cmd in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16','original.sv','fixed.sv','old.sv','positive_tb.sv','negative_tb.sv'])]:
        code,body,_=run(name,cmd);assert code==0 and '** Error' not in body,body[-2500:]
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    results={}
    for name,negative in [('positive',0),('negative',1)]:
        code,body,secs=run(name,[sim/'vsim.exe','-c',*(['-voptargs=-O5'] if 'questa' in str(sim).lower() else []),'work.tb_'+name,f'-gNEGATIVE={negative}','-do','run.do'])
        ok=('PASS MICRON_LOOKUP checks=' in body and '** Fatal' not in body and '** Error' not in body and code==0) if not negative else ('LOOKUP_MISMATCH checks=256' in body)
        assert ok,body[-2500:]
        results[name]=dict(seconds=secs,log_sha256=sha(root/(name+'.log')),expected_failure=bool(negative))
    result=dict(status='MICRON_LOOKUP_RESET_EQUIVALENCE_PASS',inputs_sha256=sha(root/'inputs.json'),results=results,full_board=False,phy_validation=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
