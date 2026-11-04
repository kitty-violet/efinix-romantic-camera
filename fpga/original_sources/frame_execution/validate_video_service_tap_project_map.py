"""Real EFX_LUT4/EFX_FF replay for same-edge address predecoding, reset and stalls."""
import argparse,json,os,pickle,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--board',type=Path,required=True);a=p.parse_args();N=a.native.resolve();O=a.root.resolve();assert not O.exists()
r=json.loads((N/'tap-project-edit.json').read_text());P=Path(r['parent']);plan=json.loads((Path(r['plan_root'])/'plan.json').read_text());ad=plan['address']
key,old,op=pickle.loads((P/'zero-source-parse.pkl').read_bytes());nk,new,np=pickle.loads((N/'zero-source-parse.pkl').read_bytes());assert key==sha(P/'native_source.v') and nk==sha(N/'native_source.v')
board=a.board.resolve();mapped=board/'outflow/ti60f225_oob.map.v'
ma=json.loads((board/'retained-map-audit.json').read_text());assert ma['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and ma['map_sha256']==sha(mapped)
mk,src,new,srcp,np=pickle.loads((board/'retained-audit-parse.pkl').read_bytes());assert mk[:2]==[nk,sha(mapped)]
oldsel=set(ad['source_ffs'])|set(ad['owned']);newsel=set(ad['owned'])|{v['new_ff'] for v in ad['projection']};outputs=[old[v['old_output_cell']][1]['O'][0] for v in ad['projection']]
ff=old[ad['source_ffs'][0]][1];external={old[n][1]['D'][0]:'din['+str(i)+']' for i,n in enumerate(ad['source_ffs'])};external.update({ff['CE'][0]:'ce',ff['SR'][0]:'rstn',ff['CLK'][0]:'clk'})
W=len(outputs);v=[f'module tb;reg clk=0;always #5 clk=~clk;reg rstn=1,ce=0;reg[31:0] din=0,rng=32\'h473b1a65;integer i,j,checks=0,resets=0,holds=0;wire[{W-1}:0] old_q,new_q;']
def circuit(label,g,pa,sel):
 tokens=sorted({t for n in sel for ts in g[n][1].values() for t in ts if t[0]=='net'},key=str);ix={t:i for i,t in enumerate(tokens)};v.append(f'wire[{len(tokens)-1}:0] {label};')
 def expr(t):return f'{label}[{ix[t]}]' if t[0]=='net' else t[1]
 driven={t for n in sel for p,ts in g[n][1].items() if p in ('Q','O') for t in ts};outside=set(tokens)-driven;assert outside<=external.keys(),outside-external.keys()
 for t in sorted(outside,key=str):v.append('assign '+expr(t)+'='+external[t]+';')
 for i,n in enumerate(sorted(sel)):
  kind,pins=g[n];pins=dict(pins)
  if kind=='EFX_LUT4':
   for j in range(4):pins.setdefault('I'+str(j),[('literal',"1'b0",None)])
  pars=','.join('.'+p+'('+str(w)+"'d"+str(val)+')' for p,(w,val) in pa[n].items());ports=','.join('.'+p+'('+expr(ts[0])+')' for p,ts in pins.items());v.append(f'{kind} #({pars}) c_{label}_{i}({ports});')
 for i,t in enumerate(outputs):v.append(f'assign {"old" if label=="lhs" else "new"}_q[{i}]={expr(t)};')
circuit('lhs',old,op,oldsel);circuit('rhs',new,np,newsel)
v+=['task check;begin #1;if(old_q!==new_q)$fatal(1,"TAP_PROJECTION_DIFF i=%0d rst=%b ce=%b old=%h new=%h",i,rstn,ce,old_q,new_q);checks=checks+1;end endtask',
'initial begin repeat(3)begin @(posedge clk);check;end rstn=0;',
'for(i=0;i<20000;i=i+1)begin @(negedge clk);rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};ce=rng[2];rstn=(i%197==0);if(rstn)resets=resets+1;if(!ce)holds=holds+1;din=rng;if(i<64)din=(32\'b1<<(i%32))^(i>=32?32\'hffffffff:0);@(posedge clk);check;@(negedge clk);din=~din;check;end',
'$display("PASS TAP_PROJECTION checks=%0d resets=%0d holds=%0d",checks,resets,holds);$finish;end endmodule']
O.mkdir();(O/'tb.sv').write_text('\n'.join(v));
for f in ('efx_lut4.v','efx_ff.v'):shutil.copy2(EF/'sim_models/verilog'/f,O/f)
inputs={str(f):sha(f) for f in [Path(__file__),N/'native_source.v',N/'tap-project-edit.json',Path(r['plan_root'])/'plan.json',P/'native_source.v',O/'tb.sv',O/'efx_lut4.v',O/'efx_ff.v',mapped,board/'retained-map-audit.json']};(O/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic();result={}
try:
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([ms/'vlib.exe','work'],O,'vlib.log',30);body,_=run([ms/'vlog.exe','-sv','efx_lut4.v','efx_ff.v','tb.sv'],O,'compile.log',90);assert 'Errors: 0, Warnings: 0' in body
 body,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],O,'simulation.log',90);assert 'PASS TAP_PROJECTION checks=40003 resets=102' in body and 'Errors: 0, Warnings: 0' in body
 for f,h in inputs.items():assert sha(f)==h
 result=dict(status='ACTUAL_VENDOR_MAPPED_TAP_PROJECTION_PASS',source_sha256=nk,map_sha256=sha(mapped),checks=40003,outputs=W,comparisons=40003*W,stimulus_iterations=20000,resets=102,extra_cycles=0)
except Exception as e:result=dict(status='FAIL',error=repr(e));raise
finally:result.update(seconds=time.monotonic()-start,inputs_sha256=sha(O/'inputs.json'),hardware_access=False);(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
