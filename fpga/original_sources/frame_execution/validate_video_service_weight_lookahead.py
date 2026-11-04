"""Paired actual source/mapped cache cones at explicitly enumerated cuts.

Checks all 512 affected storage outputs and same-edge downstream capture.
Controls are known Boolean; X/Z data are intentionally covered. This is not
a whole CNN equivalence claim and never accesses hardware.
"""
import argparse,json,os,pickle,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser()
 for n in ('board','native','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.board.resolve();n=a.native.resolve();o=a.root.resolve()
 assert o.is_relative_to(ROOT/'tmp') and not o.exists()
 edit=json.loads((n/'weight-lookahead-edit.json').read_text())
 proof=json.loads((b/'retained-map-audit.json').read_text())
 assert proof['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
 assert proof['source_sha256']==sha(n/'native_source.v')==sha(b/'native_source.v')
 parent=ROOT/'tmp/frame-execution/connected-board-video-service-probe33-data-reset-only'
 key,old,_,op,_=pickle.loads((parent/'retained-audit-parse.pkl').read_bytes())
 assert key[0]==edit['source_sha256']==sha(parent/'native_source.v')
 nk,ns,new,np,mp=pickle.loads((b/'retained-audit-parse.pkl').read_bytes())
 assert nk[1]==proof['map_sha256']==sha(b/'outflow/ti60f225_oob.map.v')
 survey=ROOT/'tmp/board-test/video-service-review1/weight-address-retiming-survey1.json'
 sv=json.loads(survey.read_text());ffs=set(sv['state_registers'])
 comb={v for r in sv['rows'] for v in r['comb']}
 storage=sorted(r['cell'] for r in edit['edits'])
 assert len(storage)==512 and len(ffs)==29 and len(comb)==94
 sel=[ffs|comb|set(storage),ffs|set(edit['new_graph'])|set(storage)]
 gs=[old,new];ps=[op,mp];clock=('net','process_clk_150',None)
 # Exact external cut tokens must match; changed LUT pin permutation is
 # taken directly from the mapped graph, not reconstructed from source.
 cuts=[];controls=set()
 for g,s in zip(gs,sel):
  driven={t for name in s for pin,ts in g[name][1].items() if pin in OUT[g[name][0]] for t in ts}
  used={t for name in s for pin,ts in g[name][1].items() if pin not in OUT[g[name][0]] for t in ts if t[0]=='net'}
  cuts.append(used-driven-{clock})
  controls|={t for name in s for pin,ts in g[name][1].items() if pin in ('SR','CE') for t in ts if t[0]=='net'}
 assert cuts[0]==cuts[1],('changed external cuts',cuts[0]^cuts[1])
 cut=sorted(cuts[0]);ci={t:i for i,t in enumerate(cut)}
 assert controls<=set(cut)
 for name in storage:
  assert old[name][0]==new[name][0]=='EFX_SRL8' and op[name]==mp[name]
  assert {k:v for k,v in old[name][1].items() if k!='A'}=={k:v for k,v in new[name][1].items() if k!='A'}
 o.mkdir();start=time.monotonic()
 lines=['module tb; reg clk=0;always #5 clk=~clk;',f'reg [{len(cut)-1}:0] stimulus=0;',
  'wire [511:0] old_out,new_out;reg [511:0] old_capture=0,new_capture=0;',
  'integer cycle,j,checks=0,unknown_data=0;reg[31:0] rng=32\'h39a48e5b;']
 for side,(g,par,selected) in enumerate(zip(gs,ps,sel)):
  tokens=sorted({t for name in selected for ts in g[name][1].values() for t in ts if t[0]=='net'})
  ix={t:i for i,t in enumerate(tokens)};lines.append(f'wire [{len(tokens)-1}:0] w{side};')
  def scalar(t):return f'w{side}[{ix[t]}]' if t[0]=='net' else t[1]
  def value(ts):return scalar(ts[0]) if len(ts)==1 else '{'+','.join(map(scalar,ts))+'}'
  lines.append(f'assign {scalar(clock)}=clk;')
  for t in cut:lines.append(f'assign {scalar(t)}=stimulus[{ci[t]}];')
  for i,name in enumerate(sorted(selected)):
   kind,pins=g[name];pins=dict(pins)
   if kind=='EFX_LUT4':
    for pin in ('I0','I1','I2','I3'):pins.setdefault(pin,[('literal',"1'b0",None)])
   args=','.join('.'+k+'('+str(w)+"'d"+str(v)+')' for k,(w,v) in par[name].items())
   ports=','.join('.'+k+'('+value(ts)+')' for k,ts in pins.items())
   if kind=='EFX_SRL8' and 'Q7' not in pins:ports+=',.Q7()'
   lines.append(f'{kind} #({args}) cell_{side}_{i}({ports});')
  for i,name in enumerate(storage):lines.append(f'assign {"old" if side==0 else "new"}_out[{i}]={value(g[name][1]["Q"])};')
 control_indices=','.join(map(str,sorted(ci[t] for t in controls)))
 lines+=['always @(posedge clk)begin old_capture<=old_out;new_capture<=new_out;end',
  'initial begin #2;if(old_out!==new_out)$fatal(1,"initial cache mismatch");',
  'for(cycle=0;cycle<8192;cycle=cycle+1)begin @(negedge clk);',
  f'for(j=0;j<{len(cut)};j=j+1)begin rng={{rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]}};stimulus[j]=rng[0];',
  f'case(j) {control_indices}: begin if(cycle%97<3)stimulus[j]=1;else if(cycle%127<4)stimulus[j]=0;end',
  "default:begin if(cycle%113==0)begin stimulus[j]=1'bx;unknown_data=unknown_data+1;end if(cycle%131==0)begin stimulus[j]=1'bz;unknown_data=unknown_data+1;end end endcase end",
  '@(posedge clk);#2;if(old_out!==new_out || old_capture!==new_capture)$fatal(1,"mapped cache mismatch cycle=%0d",cycle);checks=checks+1024;end',
  '$display("PASS ACTUAL_MAPPED_WEIGHT_CACHE cycles=8192 storage=512 checks=%0d unknown_data=%0d",checks,unknown_data);$finish;end endmodule']
 (o/'tb.sv').write_text('\n'.join(lines))
 (o/'cuts.json').write_text(json.dumps(dict(external=cut,known_controls=sorted(controls),old_cells=sorted(sel[0]),mapped_cells=sorted(sel[1])),indent=2))
 for f in ('efx_ff.v','efx_lut4.v','efx_srl8.v'):shutil.copy2(EF/'sim_models/verilog'/f,o/f)
 files=[Path(__file__).resolve(),survey,n/'weight-lookahead-edit.json',n/'weight-lookahead-independent-audit.json',b/'retained-map-audit.json',b/'outflow/ti60f225_oob.map.v',parent/'native_source.v',*o.iterdir()]
 inputs={str(f):sha(f) for f in files};(o/'inputs.json').write_text(json.dumps(inputs,indent=2))
 result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
  run([ms/'vlib.exe','work'],o,'vlib.log',30)
  log,_=run([ms/'vlog.exe','-sv','efx_ff.v','efx_lut4.v','efx_srl8.v','tb.sv'],o,'compile.log',60);assert 'Errors: 0, Warnings: 0' in log
  log,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],o,'simulation.log',120)
  assert 'PASS ACTUAL_MAPPED_WEIGHT_CACHE' in log and 'Errors: 0, Warnings: 0' in log
  for f,h in inputs.items():assert sha(f)==h
  result.update(status='ACTUAL_MAPPED_WEIGHT_LOOKAHEAD_CACHE_PASS',map_sha256=nk[1],storage=512,cycles=8192,checks=8388608,scope='Source vs mapped exact cache cones; known Boolean CE/SR, arbitrary X/Z data; checks SRL outputs and same-edge capture. Not whole CNN.',lines=[l for l in log.splitlines() if 'PASS ACTUAL' in l])
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-start,inputs_sha256=sha(o/'inputs.json'));(o/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
