"""Actual 512 mapped data FFs, reset/capture/hold at their precise cut.

This supplements the complete RQ/skip numerical and protocol tests. It proves
valid-qualified data equality at the FF boundary, not whole-core equivalence.
"""
import argparse,json,os,pickle,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run

def main():
 p=argparse.ArgumentParser()
 for n in ('board','native','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.board.resolve();n=a.native.resolve();out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 receipt=n/'data-reset-edit.json';r=json.loads(receipt.read_text())
 proof=json.loads((b/'retained-map-audit.json').read_text())
 assert proof['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
 assert proof['source_sha256']==sha(n/'native_source.v')==sha(b/'native_source.v')
 key,src,g,pa,pb=pickle.loads((b/'retained-audit-parse.pkl').read_bytes())
 assert key[1]==proof['map_sha256']==sha(b/'outflow/ti60f225_oob.map.v')
 assert len(r['records'])==512
 records=sorted(r['records'],key=lambda x:('/skip/' in x['cell'],x['cell']))
 cells=[];cuts=[]
 for i,row in enumerate(records):
  name=row['cell'];kind,pins=g[name]
  assert kind=='EFX_FF' and pins['SR']==[('literal',"1'b1",None)]
  assert {k:list(v) for k,v in pb[name].items()}==row['parameters']
  assert pb[name]['SR_POLARITY']==(1,0) and pb[name]['SR_SYNC']==(1,1)
  cap='capture_skip' if '/skip/' in name else 'capture_rq'
  params=','.join('.'+k+'('+str(w)+"'d"+str(v)+')' for k,(w,v) in pb[name].items())
  for which,sr in [('old','!reset'),('new',"1'b1")]:
   cells.append(f'EFX_FF #({params}) {which}_{i}(.CLK(clk),.CE({cap}),.SR({sr}),.D(data[{i}]),.Q(q_{which}[{i}]));')
  cuts.append(dict(cell=name,ports=pins,parameters=pb[name],group=cap))
 out.mkdir();(out/'cuts.json').write_text(json.dumps(cuts,indent=2))
 tb='''module tb;
 reg clk=0,reset=1,capture_rq=0,capture_skip=0,release_skip=0;
 reg[511:0] data=0;wire[511:0] q_old,q_new;
 reg vr=0,vs=0;reg[31:0] random_state=32'h13572468;
 integer cycle,j,checks=0,resets=0,captures=0,holds=0;
 always #5 clk=~clk;
 always @(posedge clk)begin
  vr<=!reset && capture_rq;
  vs<=!reset && (capture_skip || (vs && !release_skip));
 end
 initial begin
  repeat(4)@(negedge clk);
  for(cycle=0;cycle<4096;cycle=cycle+1)begin
   reset=(cycle%97<3);
   capture_rq=(cycle%7!=0);
   capture_skip=!reset && (cycle%5==0);
   release_skip=(cycle%11==0);
   for(j=0;j<16;j=j+1)begin
    random_state={random_state[30:0],random_state[31]^random_state[21]^random_state[1]^random_state[0]};
    data[j*32+:32]=random_state;
   end
   if(cycle%67==0)data='x;
   @(posedge clk);#2;
   if(reset)resets=resets+1;
   if(capture_rq || capture_skip)captures=captures+1;
   if(vs && !capture_skip)holds=holds+1;
   for(j=0;j<512;j=j+1)if((j<256 && vr)||(j>=256 && vs))begin
    if(q_old[j]!==q_new[j])$fatal(1,"data mismatch cycle=%0d bit=%0d",cycle,j);
    checks=checks+1;
   end
   @(negedge clk);
  end
  if(checks<1000000 || resets<100 || holds<100)$fatal(1,"coverage");
  $display("PASS MAPPED_DATA_RESET cells=512 cycles=4096 checks=%0d resets=%0d captures=%0d holds=%0d",checks,resets,captures,holds);
  $finish;
 end
'''
 (out/'tb.sv').write_text(tb+'\n'.join(cells)+'\nendmodule\n')
 shutil.copy2(EF/'sim_models/verilog/efx_ff.v',out/'efx_ff.v')
 inputs={str(f):sha(f) for f in [Path(__file__),receipt,b/'retained-map-audit.json',b/'outflow/ti60f225_oob.map.v',*out.iterdir()]}
 (out/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic()
 result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
  run([ms/'vlib.exe','work'],out,'vlib.log',30)
  log,_=run([ms/'vlog.exe','-sv','efx_ff.v','tb.sv'],out,'compile.log',60)
  assert 'Errors: 0, Warnings: 0' in log
  log,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],out,'simulation.log',60)
  assert 'PASS MAPPED_DATA_RESET' in log and 'Errors: 0, Warnings: 0' in log
  for f,h in inputs.items():assert sha(f)==h
  result.update(status='ACTUAL_MAPPED_DATA_FF_QUALIFIED_RESET_PASS',map_sha256=key[1],
   scope='512 actual mapped FF primitives at D/CE/clock cuts; control-cone behavior covered separately by RQ/skip tests',
   lines=[x for x in log.splitlines() if 'PASS MAPPED_DATA_RESET' in x])
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-start,inputs_sha256=sha(out/'inputs.json'))
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
