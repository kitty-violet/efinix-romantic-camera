"""Actual old/new vendor LUTs for every cofactor in the combined source/map."""
import argparse,json,os,pickle,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--board',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
n=a.native.resolve();o=a.root.resolve();assert o.is_relative_to(ROOT/'tmp') and not o.exists()
proof=json.loads((n/'global-batch-independent-audit.json').read_text());assert proof['status']=='GLOBAL_BATCH_INDEPENDENT_FULL_SOURCE_PASS_NOT_PHYSICAL' and proof['source_sha256']==sha(n/'native_source.v')
for f,h in proof['inputs'].items():assert sha(f)==h
parent=ROOT/'tmp/frame-execution/video-service-native53-parallel-group';key,old,op=pickle.loads((parent/'zero-source-parse.pkl').read_bytes());assert key==proof['parent_sha256']==sha(parent/'native_source.v')
sk,source,sp=pickle.loads((n/'zero-source-parse.pkl').read_bytes());assert sk==sha(n/'native_source.v')
if a.board:
 board=a.board.resolve();ma=json.loads((board/'retained-map-audit.json').read_text());assert ma['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
 mk,_,new,_,np=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert mk[0]==sk and mk[1]==sha(board/'outflow/ti60f225_oob.map.v')
else:board=None;new=source;np=sp
groups=proof['cofactors'];lines=['module tb;reg[31:0] stim=0;reg[31:0] rng=32\'hcd156fa9;integer i;']
checks=[]
for gi,r in enumerate(groups):
 boundary=list(map(tuple,r['boundary']));assert len(boundary)<=32
 # The old54 cofactor cells are identical to old53; the independent
 # operation-replay proof has verified this assertion before simulation.
 targets=r['targets'];lines.append(f'wire[{len(targets)-1}:0] before{gi},after{gi};')
 for side,(g,ps,sel) in enumerate(((old,op,r['old_cells']),(new,np,r['new_cells']))):
  tokens=sorted({t for name in sel for ts in g[name][1].values() for t in ts if t[0]=='net'},key=str);ix={t:j for j,t in enumerate(tokens)}
  lines.append(f'wire[{len(tokens)-1}:0] w{gi}_{side};')
  def value(t):return f'w{gi}_{side}[{ix[t]}]' if t[0]=='net' else t[1]
  driven={g[name][1]['O'][0] for name in sel};external=set(tokens)-driven;bindings={t:j for j,t in enumerate(boundary)}
  if side and board:
   bindings={}
   for name in sel:
    for pin,ts in source[name][1].items():
     if pin=='O':continue
     nt=g[name][1][pin];assert len(nt)==len(ts)
     for ot,mt in zip(ts,nt):
      if ot in boundary:
       j=boundary.index(ot);assert mt not in bindings or bindings[mt]==j;bindings[mt]=j
  assert external<=set(bindings),(r['label'],side,external-set(bindings))
  for t in external:lines.append(f'assign {value(t)}=stim[{bindings[t]}];')
  for j,name in enumerate(sel):
   kind,pins=g[name];assert kind=='EFX_LUT4';ports=dict(pins)
   for pin in ('I0','I1','I2','I3'):ports.setdefault(pin,[('literal',"1'b0",None)])
   lines.append(f"EFX_LUT4 #(.LUTMASK(16'd{ps[name]['LUTMASK'][1]})) c{gi}_{side}_{j}("+','.join('.'+pin+'('+value(ts[0])+')' for pin,ts in ports.items())+');')
  for j,tgt in enumerate(targets):lines.append(f'assign {"before" if side==0 else "after"}{gi}[{j}]={value(g[tgt][1]["O"][0])};')
 checks.append(f'if(before{gi}!==after{gi} || (^after{gi}===1\'bx))$fatal(1,"group {gi} differs i=%0d old=%b new=%b",i,before{gi},after{gi});')
lines += ['task check;begin #2;']+checks+['end endtask','initial begin stim=0;check;stim=~0;check;for(i=0;i<32;i=i+1)begin stim=32\'b1<<i;check;stim=~stim;check;end',
 'for(i=0;i<262144;i=i+1)begin stim=i;check;end',
 'for(i=0;i<131072;i=i+1)begin rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};stim=rng;check;end',
 '$display("PASS GLOBAL_CONES vectors=393282");$finish;end endmodule']
o.mkdir();(o/'tb.sv').write_text('\n'.join(lines));shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',o/'efx_lut4.v')
paths=[Path(__file__),n/'native_source.v',n/'global-batch-independent-audit.json',parent/'native_source.v',o/'tb.sv',o/'efx_lut4.v']
if board:paths += [board/'retained-map-audit.json',board/'outflow/ti60f225_oob.map.v']
inputs={str(f):sha(f) for f in paths};(o/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic();result=dict(status='RUNNING')
try:
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([ms/'vlib.exe','work'],o,'vlib.log',30)
 body,_=run([ms/'vlog.exe','-sv','efx_lut4.v','tb.sv'],o,'compile.log',60);assert 'Errors: 0, Warnings: 0' in body
 body,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],o,'simulation.log',180)
 assert 'PASS GLOBAL_CONES vectors=393282' in body and 'Errors: 0, Warnings: 0' in body
 for f,h in inputs.items():assert sha(f)==h
 result.update(status='ACTUAL_GLOBAL_CONTROL_CONES_PASS',source_sha256=sk,map_sha256=mk[1] if board else None,mapped=bool(board),groups=len(groups),vectors_per_group=393282,all_boolean_proof='separate canonical BDD; this test is exhaustive for cuts<=18, directed/random for32',hardware_access=False,board_release=False)
except Exception as e:result.update(status='FAIL',error=repr(e));raise
finally:
 result.update(seconds=time.monotonic()-start,inputs_sha256=sha(o/'inputs.json'));(o/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
