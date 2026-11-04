"""Actual vendor primitives: full changed Boolean cut plus four-state fixed carry."""
import argparse,json,os,pickle,re,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();N=a.native.resolve();O=a.root.resolve();assert not O.exists() and O.is_relative_to(ROOT/'tmp')
proof=json.loads((N/'shared-frontier-independent-audit.json').read_text());assert proof['status']=='SHARED_FRONTIER_COMPOSITIONAL_CEC_PASS_NOT_PHYSICAL' and proof['source_sha256']==sha(N/'native_source.v')
for f,h in proof['inputs'].items():assert sha(f)==h
key,old,op,new,np,cut=pickle.loads((N/'frontier-proof-graph.pkl').read_bytes());boundary=list(map(tuple,cut['boundary']));outputs=list(map(tuple,cut['outputs']));W=len(boundary);vectors=2+2*W+4096
v=[f'module tb;reg[{W-1}:0] stim=0;reg[31:0] rng=32\'h729db521;integer i,j,k;integer carry_exact=0,carry_refinements=0;wire[{len(outputs)-1}:0] old_q,new_q;reg[11:0] fs=0;wire[3:0] carry_old,carry_new;']
def circuit(label,g,ps,selected,inputs,outs,driver,outname):
 tokens=sorted({t for n in selected for ts in g[n][1].values() for t in ts if t[0]=='net'},key=str);ix={t:i for i,t in enumerate(tokens)};v.append(f'wire[{len(tokens)-1}:0] {label};')
 def expr(t):return f'{label}[{ix[t]}]' if t[0]=='net' else t[1]
 driven={t for n in selected for p,ts in g[n][1].items() if p in ('O','CO') for t in ts};external=set(tokens)-driven;assert external<=set(inputs),(label,external-set(inputs))
 for t in sorted(external,key=str):v.append(f'assign {expr(t)}={driver}[{inputs.index(t)}];')
 for i,n in enumerate(selected):
  kind,pins=g[n];assert kind in ('EFX_LUT4','EFX_ADD');pins=dict(pins)
  if kind=='EFX_LUT4':
   for j in range(4):pins.setdefault('I'+str(j),[('literal',"1'b0",None)])
  else:
   for port in ('O','CO'):
    if port not in pins:v.append(f'wire unused_{label}_{i}_{port};')
  pars=','.join('.'+p+'('+str(w)+"'d"+str(val)+')' for p,(w,val) in ps[n].items())
  ports=','.join('.'+p+'('+expr(ts[0])+')' for p,ts in pins.items())
  if kind=='EFX_ADD':
   for port in ('O','CO'):
    if port not in pins:ports+=f',.{port}(unused_{label}_{i}_{port})'
  v.append(f'{kind} #({pars}) c_{label}_{i}({ports});')
 for i,t in enumerate(outs):v.append(f'assign {outname}[{i}]={expr(t)};')
circuit('lhs',old,op,cut['old_cells'],boundary,outputs,'stim','old_q');circuit('rhs',new,np,cut['new_cells'],boundary,outputs,'stim','new_q')
plan=json.loads((Path(json.loads((N/'shared-frontier-edit.json').read_text())['plan_root'])/'plan.json').read_text());carry=next(e['rows'] for e in plan['edits'] if e['kind']=='four_writer_overflow')
for i,row in enumerate(carry):
 def gg(d):return {n:(k,{p:[tuple(t) for t in ts] for p,ts in pins.items()}) for n,(k,pins) in d.items()}
 def pp(d):return {n:{k:tuple(v) for k,v in ps.items()} for n,ps in d.items()}
 v.append(f'wire[0:0] c_old{i},c_new{i};');bi=list(map(tuple,row['boundary']));oo=[tuple(row['output'])]
 circuit('carrylhs'+str(i),gg(row['old_graph']),pp(row['old_parameters']),list(row['old_graph']),bi,oo,'fs','c_old'+str(i));circuit('carryrhs'+str(i),gg(row['new_graph']),pp(row['new_parameters']),list(row['new_graph']),bi,oo,'fs','c_new'+str(i))
 v.append(f'assign carry_old[{i}]=c_old{i}[0];assign carry_new[{i}]=c_new{i}[0];')
v+=['task check;begin #2;if(old_q!==new_q || (^new_q===1\'bx))$fatal(1,"SHARED_FRONTIER_DIFF vector=%0d old=%h new=%h",i,old_q,new_q);end endtask',
 f'initial begin stim=0;check;stim=~0;check;for(i=0;i<{W};i=i+1)begin stim=0;stim[i]=1;check;stim=~stim;check;end',
 f'for(i=0;i<4096;i=i+1)begin for(j=0;j<{W};j=j+1)begin rng={{rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]}};stim[j]=rng[0];end check;end',
 'for(i=0;i<4608;i=i+1)begin if(i<4096)fs=i;else for(j=0;j<12;j=j+1)begin rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};case(rng[2:1])0:fs[j]=0;1:fs[j]=1;2:fs[j]=1\'bx;3:fs[j]=1\'bz;endcase end #2;for(k=0;k<4;k=k+1)if(carry_old[k]===carry_new[k])carry_exact=carry_exact+1;else if(carry_old[k]===1\'bx && carry_new[k]!==1\'bz)carry_refinements=carry_refinements+1;else $fatal(1,"CARRY_DEFINED_OUTPUT_DIFF i=%0d old=%b new=%b",i,carry_old,carry_new);end',
 '$display("CARRY_CLASSIFICATION exact=%0d X_refinement=%0d",carry_exact,carry_refinements);',
 f'$display("PASS SHARED_FRONTIER vectors={vectors} outputs={len(outputs)} carry_cases=18432");$finish;end endmodule']
O.mkdir();(O/'tb.sv').write_text('\n'.join(v),encoding='utf8')
for f in ('efx_lut4.v','efx_add.v'):shutil.copy2(EF/'sim_models/verilog'/f,O/f)
inputs={str(f):sha(f) for f in [Path(__file__),N/'native_source.v',N/'shared-frontier-independent-audit.json',N/'frontier-cut.json',N/'frontier-proof-graph.pkl',O/'tb.sv',O/'efx_lut4.v',O/'efx_add.v']};(O/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic();result={}
try:
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([ms/'vlib.exe','work'],O,'vlib.log',30)
 body,_=run([ms/'vlog.exe','-sv','efx_lut4.v','efx_add.v','tb.sv'],O,'compile.log',90);assert 'Errors: 0, Warnings: 0' in body
 body,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],O,'simulation.log',180);assert f'PASS SHARED_FRONTIER vectors={vectors}' in body and 'Errors: 0, Warnings: 0' in body
 for f,h in inputs.items():assert sha(f)==h
 classification=re.search(r'CARRY_CLASSIFICATION exact=(\d+) X_refinement=(\d+)',body);assert classification and sum(map(int,classification.groups()))==18432
 result=dict(status='ACTUAL_VENDOR_SHARED_FRONTIER_CARRY_PASS',source_sha256=sha(N/'native_source.v'),vectors=vectors,outputs=len(outputs),comparisons=vectors*len(outputs),four_state_carry_cases=18432,carry_binary_cases=16384,carry_mixed_XZ_cases=2048,four_state_carry_exact=int(classification[1]),four_state_carry_X_refinements=int(classification[2]),carry_scope='All4096operand combinations per lane plus512mixed-XZ samples per lane; every old defined output must match. EFX_ADD arithmetic X pessimism may become defined under the exact Boolean carry; refinements counted, not claimed arbitrary-X equality. Current reset/valid interface checks still required.',copy_aliases='independent exact primitive/connection proof',all_Boolean_inputs='separate whole-cut CEC',arbitrary_X_for_complete_cut=False)
except Exception as e:result=dict(status='FAIL',error=repr(e));raise
finally:
 result.update(seconds=time.monotonic()-start,inputs_sha256=sha(O/'inputs.json'),hardware_access=False,board_release=False);(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
