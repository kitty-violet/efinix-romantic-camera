"""Same-edge projection: actual original/current EFX_FF and LUT, source or map."""
import argparse,json,os,pickle,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--board',type=Path);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
n=a.native.resolve();o=a.root.resolve();assert o.is_relative_to(ROOT/'tmp') and not o.exists()
pr=json.loads((n/'endpoint-batch-independent-audit-v2.json').read_text());assert pr['status']=='ENDPOINT_BATCH_FULL_SOURCE_INDUCTIVE_PASS_NOT_PHYSICAL' and pr['source_sha256']==sha(n/'native_source.v') and pr['induction_rows']==512
for f,h in pr['inputs'].items():assert sha(f)==h
parent=ROOT/'tmp/frame-execution/video-service-native56-global-targeted';key,old,op=pickle.loads((parent/'zero-source-parse.pkl').read_bytes());assert key==pr['parent_sha256']==sha(parent/'native_source.v')
sk,source,sp=pickle.loads((n/'zero-source-parse.pkl').read_bytes());assert sk==sha(n/'native_source.v')
if a.board:
 board=a.board.resolve();ma=json.loads((board/'retained-map-audit.json').read_text());assert ma['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
 mk,_,new,_,np=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert mk[0]==sk and mk[1]==sha(board/'outflow/ti60f225_oob.map.v')
else:board=None;new=source;np=sp
rt=pr['retime'];states=rt['original_state'];addresses=rt['old_addresses']
cuts={old[states[0]][1]['CLK'][0]:'clk'}
for i,name in enumerate(states):
 pins=old[name][1]
 cuts[pins['D'][0]]=f'd[{i}]'
 if pins['CE'][0][0]=='net':cuts[pins['CE'][0]]='ce'
 cuts[pins['SR'][0]]='sr_count' if i<2 else 'sr_partial'
lines=['module tb;reg clk=0;reg[2:0] d=0;reg ce=0,sr_count=1,sr_partial=1;wire[2:0] before_q,after_q;integer q,v,c,s,t,i,checks=0;reg[31:0] rng=32\'hc3179da1;']
for side,(g,ps,selected) in enumerate(((old,op,states+addresses),(new,np,states+rt['new_cells']))):
 tokens=sorted({t for name in selected for ts in g[name][1].values() for t in ts if t[0]=='net'},key=str);ix={t:i for i,t in enumerate(tokens)}
 lines.append(f'wire[{len(tokens)-1}:0] w{side};')
 def val(t):return f'w{side}[{ix[t]}]' if t[0]=='net' else t[1]
 driven={g[name][1]['O' if g[name][0]=='EFX_LUT4' else 'Q'][0] for name in selected};external=set(tokens)-driven
 binds=cuts.copy()
 if side and board:
  binds={}
  for name in selected:
   for pin,ts in source[name][1].items():
    if pin in ('Q','O'):continue
    nt=g[name][1][pin];assert len(ts)==len(nt)
    for ot,mt in zip(ts,nt):
     if ot in cuts:
      assert mt not in binds or binds[mt]==cuts[ot];binds[mt]=cuts[ot]
 assert external<=set(binds),(side,external-set(binds))
 for token in external:lines.append(f'assign {val(token)}={binds[token]};')
 for j,name in enumerate(selected):
  kind,pins=g[name];pins=dict(pins)
  if kind=='EFX_LUT4':
   for pin in ('I0','I1','I2','I3'):pins.setdefault(pin,[('literal',"1'b0",None)])
  pars=','.join('.'+k+'('+str(w)+"'d"+str(v)+')' for k,(w,v) in ps[name].items())
  ports=','.join('.'+p+'('+val(ts[0])+')' for p,ts in pins.items())
  lines.append(f'{kind} #({pars}) c{side}_{j}({ports});')
 for j,name in enumerate(addresses):lines.append(f'assign {"before_q" if side==0 else "after_q"}[{j}]={val(g[name][1]["O" if side==0 else "Q"][0])};')
lines += ['task edge_check;begin #3;clk=1;#3;checks=checks+1;if(before_q!==after_q)$fatal(1,"FETCH_PROJECTION_MISMATCH check=%0d before=%b after=%b",checks,before_q,after_q);clk=0;end endtask',
 'initial begin edge_check;for(q=0;q<8;q=q+1)for(v=0;v<8;v=v+1)for(c=0;c<2;c=c+1)for(s=0;s<2;s=s+1)for(t=0;t<2;t=t+1)begin',
 'd=q;ce=1;sr_count=0;sr_partial=0;edge_check;d=v;ce=c;sr_count=s;sr_partial=t;edge_check;end',
 'for(i=0;i<100000;i=i+1)begin rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};d=rng[2:0];ce=rng[3];sr_count=rng[4];sr_partial=rng[5];edge_check;end',
 'd=3\'bxxx;ce=0;sr_count=1;sr_partial=1;edge_check;d=0;ce=1;sr_count=0;sr_partial=0;edge_check;',
 '$display("PASS FETCH_PROJECTION checks=%0d induction_rows=512 added_latency=0",checks);$finish;end endmodule']
o.mkdir();(o/'tb.sv').write_text('\n'.join(lines))
for f in ('efx_lut4.v','efx_ff.v'):shutil.copy2(EF/'sim_models/verilog'/f,o/f)
paths=[Path(__file__),n/'native_source.v',n/'endpoint-batch-independent-audit-v2.json',parent/'native_source.v',o/'tb.sv',o/'efx_lut4.v',o/'efx_ff.v']
if board:paths += [board/'retained-map-audit.json',board/'outflow/ti60f225_oob.map.v']
inputs={str(f):sha(f) for f in paths};(o/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic();result=dict(status='RUNNING')
try:
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([ms/'vlib.exe','work'],o,'vlib.log',30)
 log,_=run([ms/'vlog.exe','-sv','efx_lut4.v','efx_ff.v','tb.sv'],o,'compile.log',60);assert 'Errors: 0, Warnings: 0' in log
 log,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],o,'simulation.log',180)
 assert 'PASS FETCH_PROJECTION checks=101027' in log and 'Errors: 0, Warnings: 0' in log
 for f,h in inputs.items():assert sha(f)==h
 result.update(status='ACTUAL_FETCH_PROJECTION_SAME_EDGE_PASS',source_sha256=sk,map_sha256=mk[1] if board else None,mapped=bool(board),clock_checks=101027,induction_rows=512,added_latency=0,unknown_reset_data_ignored=True,arbitrary_unknown_controls=False,hardware_access=False,board_release=False)
except Exception as e:result.update(status='FAIL',error=repr(e));raise
finally:
 result.update(seconds=time.monotonic()-start,inputs_sha256=sha(o/'inputs.json'));(o/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
