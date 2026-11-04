"""Actual vendor four-state exhaustive filtering for all disjoint LUT fusions."""
import argparse,json,os,re,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--survey',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
s=a.survey.resolve();o=a.root.resolve();assert o.is_relative_to(ROOT/'tmp') and not o.exists()
r=json.loads(s.read_text());assert sha(Path(r['native'])/'native_source.v')==r['source_sha256']
for f,h in r['inputs'].items():assert sha(f)==h
rows=r['rows'];v=[f'module tb;reg[3:0] stim;integer i,j;wire[{len(rows)-1}:0] old_q,new_q;']
for i,row in enumerate(rows):
 boundary=list(map(tuple,row['boundary']));mid=tuple(row['old_graph'][row['parent']][1]['O'][0]);v.append(f'wire mid{i};')
 def expr(t):
  t=tuple(t)
  if t==mid:return f'mid{i}'
  if t[0]=='literal':return t[1]
  return f'stim[{boundary.index(t)}]'
 for role,name,out in [('p',row['parent'],f'mid{i}'),('c',row['target'],f'old_q[{i}]')]:
  k,pins=row['old_graph'][name];assert k=='EFX_LUT4';mask=row['old_parameters'][name]['LUTMASK'][1]
  ports=','.join('.I'+str(j)+'('+expr(pins.get('I'+str(j),[['literal',"1'b0",None]])[0])+')' for j in range(4))
  v.append(f"EFX_LUT4 #(.LUTMASK(16'd{mask})) {role}{i}({ports},.O({out}));")
 mask=row['new_parameters'][row['target']]['LUTMASK'][1]
 ports=','.join('.I'+str(j)+'('+(f'stim[{j}]' if j<len(boundary) else "1'b0")+')' for j in range(4))
 v.append(f"EFX_LUT4 #(.LUTMASK(16'd{mask})) n{i}({ports},.O(new_q[{i}]));")
v+=['initial begin for(i=0;i<256;i=i+1)begin for(j=0;j<4;j=j+1)case((i>>(j*2))&3)0:stim[j]=0;1:stim[j]=1;2:stim[j]=1\'bx;3:stim[j]=1\'bz;endcase #2;']
v += [f'if(old_q[{i}]!==new_q[{i}])$display("MISMATCH pair={i} vector=%d",i);' for i in range(len(rows))]
v+=['end $display("ALL_256_VECTORS_DONE");$finish;end endmodule']
o.mkdir();(o/'tb.sv').write_text('\n'.join(v));shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',o/'efx_lut4.v')
inputs={str(f):sha(f) for f in [Path(__file__),s,Path(r['native'])/'native_source.v',o/'tb.sv',o/'efx_lut4.v']};(o/'inputs.json').write_text(json.dumps(inputs,indent=2))
ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
try:
 run([ms/'vlib.exe','work'],o,'vlib.log',30)
 body,_=run([ms/'vlog.exe','-sv','efx_lut4.v','tb.sv'],o,'compile.log',90);assert 'Errors: 0, Warnings: 0' in body
 body,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],o,'simulation.log',90)
 assert 'ALL_256_VECTORS_DONE' in body and 'Errors: 0, Warnings: 0' in body
 bad={int(x) for x in re.findall(r'MISMATCH pair=(\d+)',body)}
 result=dict(status='EXCLUSIVE_FUSIONS_VENDOR_FOUR_STATE_FILTER_PASS',accepted=[i for i in range(len(rows)) if i not in bad],rejected=sorted(bad),vectors_per_pair=256,survey_sha256=sha(s),source_sha256=r['source_sha256'],seconds=time.monotonic()-start,inputs=inputs,hardware_access=False,board_release=False)
 for f,h in inputs.items():assert sha(f)==h
 (o/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','accepted','rejected')},indent=2));print('accepted',len(result['accepted']),'rejected',len(bad))
except Exception as e:
 (o/'failure.json').write_text(json.dumps(dict(error=repr(e),seconds=time.monotonic()-start)));raise
