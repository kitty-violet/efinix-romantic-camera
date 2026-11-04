"""Independent vendor-model four-state test of the exact cache LUT edit."""
import argparse,json,os,re,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
from native_local_drivers import CELL,graph
from probe_full_native_split_map import params

def main():
 p=argparse.ArgumentParser()
 for n in ('native','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();n=a.native.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 receipt=n/'cache-lut-edit.json';r=json.loads(receipt.read_text());base=Path(r['base'])
 old=(base/'native_source.v').read_text();new=(n/'native_source.v').read_text()
 assert sha(base/'native_source.v')==r['base_source_sha256']
 assert new.replace(r['after'],r['before'],1).replace(r['param_after'],r['param_before'],1)==old
 wanted={r['cell'],r['upstream']};cells={m[2].lstrip('\\'):m[0] for m in CELL.finditer(old) if m[2].lstrip('\\') in wanted}
 oldg=graph('\n'.join(cells.values()));newg=graph(r['after']);po=params(old);pn=params(new)
 k=r['cell'];up=r['upstream'];assert oldg[k][1]['I1']==oldg[up][1]['O']
 assert newg[k][1]=={'I0':oldg[k][1]['I0'],'I1':oldg[up][1]['I0'],'I2':oldg[up][1]['I1'],'O':oldg[k][1]['O']}
 assert all(sha(f)==sha(n/f.name) for f in base.glob('*.sv'))
 out.mkdir();mask0=po[up]['LUTMASK'][1];mask1=po[k]['LUTMASK'][1];masknew=pn[k]['LUTMASK'][1]
 tb=f'''module tb;
 reg p,a,b,unused;wire intermediate,oldq,newq;integer i,j,k,l,checks=0;
 function val(input integer v);case(v)0:val=0;1:val=1;2:val=1'bx;3:val=1'bz;endcase endfunction
 EFX_LUT4 #(.LUTMASK(16'd{mask0})) first(.I0(a),.I1(b),.I2(unused),.I3(unused),.O(intermediate));
 EFX_LUT4 #(.LUTMASK(16'd{mask1})) oldlast(.I0(p),.I1(intermediate),.I2(unused),.I3(unused),.O(oldq));
 EFX_LUT4 #(.LUTMASK(16'd{masknew})) newlast(.I0(p),.I1(a),.I2(b),.I3(unused),.O(newq));
 initial begin
  for(i=0;i<4;i=i+1)for(j=0;j<4;j=j+1)for(k=0;k<4;k=k+1)for(l=0;l<4;l=l+1)begin
   p=val(i);a=val(j);b=val(k);unused=val(l);#1;
   if(oldq!==newq)$fatal(1,"cache address mismatch %b %b %b %b old=%b new=%b",p,a,b,unused,oldq,newq);
   checks=checks+1;
  end
  $display("PASS CACHE_LUT_COLLAPSE checks=%0d all_four_state=1 latency_added=0",checks);$finish;
 end
 endmodule
'''
 (out/'tb.sv').write_text(tb);shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',out/'efx_lut4.v')
 inputs={str(f):sha(f) for f in (Path(__file__).resolve(),receipt,base/'native_source.v',n/'native_source.v',out/'tb.sv',out/'efx_lut4.v')}
 (out/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic();result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
  run([ms/'vlib.exe','work'],out,'vlib.log',30)
  log,_=run([ms/'vlog.exe','-sv','efx_lut4.v','tb.sv'],out,'compile.log',60);assert 'Errors: 0, Warnings: 0' in log
  log,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],out,'simulation.log',60)
  assert 'PASS CACHE_LUT_COLLAPSE checks=256' in log and 'Errors: 0, Warnings: 0' in log
  for f,h in inputs.items():assert sha(f)==h
  result.update(status='EXACT_CACHE_LUT_VENDOR_FOUR_STATE_PASS',cases=256,latency_added=0,full_text_inverse=True)
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-start,inputs_sha256=sha(out/'inputs.json'));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
