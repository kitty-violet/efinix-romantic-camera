"""Independent vendor four-state miter for all changed read-address output bits."""
import argparse,json,os,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from native_local_drivers import CELL,graph
from probe_full_native_split_map import params
from map_video_service_safe import run

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 n=a.native.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 receipt=n/'ram-address-edit.json';r=json.loads(receipt.read_text());base=Path(r['base'])
 old=(base/'native_source.v').read_text();new=(n/'native_source.v').read_text();assert sha(base/'native_source.v')==r['base_source_sha256']
 restored=new
 for row in reversed(r['records']):
  assert restored.count(row['after'])==restored.count(row['param_after'])==1
  restored=restored.replace(row['after'],row['before'],1).replace(row['param_after'],row['param_before'],1)
 assert restored==old and all(sha(f)==sha(n/f.name) for f in base.glob('*.sv'))
 names={x[k] for x in r['records'] for k in ('parent','child')}
 cells={m[2].lstrip('\\'):m[0] for m in CELL.finditer(old) if m[2].lstrip('\\') in names}
 og=graph('\n'.join(cells.values()));op=params(old);np=params(new);blocks=[]
 for idx,row in enumerate(r['records']):
  up=row['parent'];child=row['child'];ng=graph(row['after'])[child][1]
  boundary=[ng[f'I{i}'][0] for i in range(4)];assert len(set(boundary))==4 and ng['O']==og[child][1]['O']
  token_map={t:f'iv[{i}]' for i,t in enumerate(boundary)};token_map[og[up][1]['O'][0]]=f'mid[{idx}]'
  def ports(pins):
   args=[]
   for i in range(4):
    if 'I'+str(i) not in pins:
     args.append(f'.I{i}(unused)');continue
    t=pins.get('I'+str(i),[('literal',"1'b0",None)])[0]
    val=t[1] if t[0]=='literal' else token_map.get(t,'unused')
    args.append(f'.I{i}({val})')
   return ','.join(args)
  blocks.extend([f'EFX_LUT4 #(.LUTMASK(16\'d{op[up]["LUTMASK"][1]})) u{idx}({ports(og[up][1])},.O(mid[{idx}]));',
    f'EFX_LUT4 #(.LUTMASK(16\'d{op[child]["LUTMASK"][1]})) o{idx}({ports(og[child][1])},.O(oldq[{idx}]));',
    f'EFX_LUT4 #(.LUTMASK(16\'d{np[child]["LUTMASK"][1]})) n{idx}({ports(ng)},.O(newq[{idx}]));'])
 out.mkdir();tb='''module tb;
 reg[3:0] iv;reg unused;wire[44:0] mid,oldq,newq;integer x,i,checks=0;
 function v(input integer a);case(a)0:v=0;1:v=1;2:v=1'bx;3:v=1'bz;endcase endfunction
 '''+'\n'.join(blocks)+'''
 initial begin
  for(x=0;x<1024;x=x+1)begin
   for(i=0;i<4;i=i+1)iv[i]=v((x>>(2*i))&3);unused=v((x>>8)&3);#1;
   if(oldq!==newq)$fatal(1,"RAM address composition mismatch case=%0d old=%h new=%h",x,oldq,newq);
   checks=checks+45;
  end
  $display("PASS RAM_ADDRESS_COMPOSITION bits=45 checks=%0d four_state=1",checks);$finish;
 end endmodule
 '''
 (out/'tb.sv').write_text(tb);shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',out/'efx_lut4.v')
 files=[Path(__file__),receipt,base/'native_source.v',n/'native_source.v',out/'tb.sv',out/'efx_lut4.v']
 (out/'inputs.json').write_text(json.dumps({str(f):sha(f) for f in files},indent=2));t=time.monotonic();result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
  run([ms/'vlib.exe','work'],out,'vlib.log',30)
  log,_=run([ms/'vlog.exe','-sv','efx_lut4.v','tb.sv'],out,'compile.log',60);assert 'Errors: 0, Warnings: 0' in log
  log,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],out,'simulation.log',60)
  assert 'PASS RAM_ADDRESS_COMPOSITION bits=45 checks=46080' in log and 'Errors: 0, Warnings: 0' in log
  result.update(status='EXACT_RAM_ADDRESS_VENDOR_FOUR_STATE_PASS',address_bits=45,checks=46080,full_source_inverse=True,latency_added=0)
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-t,inputs_sha256=sha(out/'inputs.json'));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
