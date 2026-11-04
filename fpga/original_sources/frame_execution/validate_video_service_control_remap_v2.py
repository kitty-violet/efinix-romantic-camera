"""Complete output cones in bounded Questa partitions; no dropped dependencies."""
import argparse,json,os,pickle,shutil,time,functools
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
from bind_video_service_mapped_cut import bind_cut
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--board',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
N=a.native.resolve();O=a.root.resolve();assert O.is_relative_to(ROOT/'tmp') and not O.exists()
r=json.loads((N/'control-remap-edit.json').read_text());pr=json.loads((N/'control-remap-independent-audit.json').read_text());assert pr['status']=='CONTROL_REMAP_FULL_SOURCE_CEC_PASS_NOT_PHYSICAL'
for f,h in pr['inputs'].items():assert sha(f)==h
P=Path(r['parent']);key,old,op=pickle.loads((P/'zero-source-parse.pkl').read_bytes());sk,source,sp=pickle.loads((N/'zero-source-parse.pkl').read_bytes());assert key==sha(P/'native_source.v') and sk==sha(N/'native_source.v')==pr['source_sha256']
BD=a.board.resolve() if a.board else None
if BD:
 ma=json.loads((BD/'retained-map-audit.json').read_text());assert ma['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
 mk,_,new,_,np=pickle.loads((BD/'retained-audit-parse.pkl').read_bytes());assert mk[0]==sk and mk[1]==sha(BD/'outflow/ti60f225_oob.map.v')
else:new,np=source,sp
boundary=list(map(tuple,r['cut_inputs']));targets=r['outputs'];width=len(boundary);vectors=2+2*width+16384
graphs=[old,new];pars=[op,np];owned=[set(r['source_cells']),set(r['new_cells'])]
drivers=[{g[n][1]['O'][0]:n for n in own} for g,own in zip(graphs,owned)]
@functools.lru_cache(None)
def cone(side,n):
 result={n}
 for pin,ts in graphs[side][n][1].items():
  if pin!='O':
   for t in ts:
    if t in drivers[side]:result.update(cone(side,drivers[side][t]))
 return frozenset(result)
groups=[];outputs=[];sets=[set(),set()]
for n in targets:
 candidate=[sets[i]|set(cone(i,n)) for i in (0,1)]
 if sum(map(len,candidate))+1>4400:
  assert outputs;groups.append((outputs,sets));outputs=[];sets=[set(),set()];candidate=[set(cone(i,n)) for i in (0,1)]
 assert sum(map(len,candidate))+1<=4400,('one output exceeds budget',n)
 outputs.append(n);sets=candidate
if outputs:groups.append((outputs,sets))
assert [n for out,ss in groups for n in out]==targets
O.mkdir();cases=[];paths=[Path(__file__),Path(__file__).with_name('bind_video_service_mapped_cut.py'),P/'native_source.v',N/'native_source.v',N/'control-remap-edit.json',N/'control-remap-independent-audit.json']
for gi,(tg,sets) in enumerate(groups):
 count=len(tg);lines=[f"module tb{gi};reg[{width-1}:0]stim=0;reg[31:0]rng=32'hc61c6a7b;integer i,j;wire[{count-1}:0]before_v,after_v;"]
 for side,(g,ps,ss) in enumerate(zip(graphs,pars,sets)):
  sel=sorted(ss);ts=sorted({t for n in sel for vs in g[n][1].values() for t in vs if t[0]=='net'},key=str);ix={t:i for i,t in enumerate(ts)}
  lines.append(f'wire[{len(ts)-1}:0] w{side};')
  def val(t):return f'w{side}[{ix[t]}]' if t[0]=='net' else t[1]
  driven={g[n][1]['O'][0] for n in sel};external=set(ts)-driven;bindings={t:i for i,t in enumerate(boundary)}
  if side and BD:bindings,_=bind_cut(source,new,bindings,external)
  assert external<=set(bindings),('incomplete cone',gi,side)
  for t in external:lines.append(f'assign {val(t)}=stim[{bindings[t]}];')
  for i,n in enumerate(sel):
   kind,pins=g[n];assert kind=='EFX_LUT4';pins=dict(pins)
   for j in range(4):pins.setdefault(f'I{j}',[('literal',"1'b0",None)])
   lines.append(f"EFX_LUT4 #(.LUTMASK(16'd{ps[n]['LUTMASK'][1]})) c{side}_{i}("+','.join('.'+pin+'('+val(v[0])+')' for pin,v in pins.items())+');')
  for i,n in enumerate(tg):lines.append(f'assign {"before_v" if side==0 else "after_v"}[{i}]={val(g[n][1]["O"][0])};')
 lines += ['task check;begin #2;if(before_v!==after_v || (^after_v===1\'bx))$fatal(1,"control remap differs vector=%0d",i);end endtask',
  'initial begin stim=0;check;stim=~0;check;',f"for(i=0;i<{width};i=i+1)begin stim=0;stim[i]=1;check;stim=~stim;check;end",
  f"for(i=0;i<16384;i=i+1)begin for(j=0;j<{width};j=j+1)begin rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);stim[j]=rng[0];end check;end",
  f'$display("PASS CONTROL_REMAP group={gi} vectors={vectors} outputs={count}");$finish;end endmodule']
 fn=O/f'tb{gi}.sv';fn.write_text('\n'.join(lines));paths.append(fn)
 cases.append(dict(group=gi,outputs=tg,old_cells=sorted(sets[0]),new_cells=sorted(sets[1]),instances=sum(map(len,sets))+1))
(O/'partitions.json').write_text(json.dumps(cases,indent=2));paths.append(O/'partitions.json')
shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',O/'efx_lut4.v');paths.append(O/'efx_lut4.v')
if BD:paths += [BD/'retained-map-audit.json',BD/'outflow/ti60f225_oob.map.v']
inputs={str(f):sha(f) for f in paths};(O/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic();result={'status':'RUNNING'}
try:
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([ms/'vlib.exe','work'],O,'vlib.log',30)
 body,_=run([ms/'vlog.exe','-sv','efx_lut4.v',*[f'tb{i}.sv' for i in range(len(groups))]],O,'compile.log',60);assert 'Errors: 0, Warnings: 0' in body
 for gi,(tg,ss) in enumerate(groups):
  body,_=run([ms/'vsim.exe','-c',f'work.tb{gi}','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],O,f'simulation{gi}.log',240)
  assert f'PASS CONTROL_REMAP group={gi} vectors={vectors} outputs={len(tg)}' in body and 'Errors: 0, Warnings: 0' in body
 for f,h in inputs.items():assert sha(f)==h
 result.update(status='ACTUAL_VENDOR_CONTROL_REMAP_PASS',source_sha256=sk,map_sha256=mk[1] if BD else None,mapped=bool(BD),vectors=vectors,outputs=len(targets),comparisons=vectors*len(targets),groups=len(groups),max_instances=max(x['instances'] for x in cases),complete_cones_all_external_outputs_tested=True,hardware_access=False,board_release=False)
except Exception as e:result.update(status='FAIL',error=repr(e));raise
finally:result.update(seconds=time.monotonic()-start,inputs_sha256=sha(O/'inputs.json'));(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
