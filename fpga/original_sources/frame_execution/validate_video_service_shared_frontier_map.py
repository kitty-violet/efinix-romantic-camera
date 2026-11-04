"""Exhaustive actual-vendor four-state mapping check for every changed62 LUT."""
import argparse,json,os,pickle,shutil,time
from pathlib import Path
from run_units import ROOT,EF,sha
from bind_video_service_control_cut import bind_control_cut
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();N=a.native.resolve();BD=a.board.resolve();O=a.root.resolve();assert not O.exists();start=time.monotonic()
r=json.loads((N/'shared-frontier-edit.json').read_text());key,g,pa=pickle.loads((N/'zero-source-parse.pkl').read_bytes());mk,raw,b,rawp,pb=pickle.loads((BD/'retained-audit-parse.pkl').read_bytes());assert key==sha(N/'native_source.v')==mk[0] and mk[1]==sha(BD/'outflow/ti60f225_oob.map.v')
proof=json.loads((BD/'retained-map-audit.json').read_text());assert proof['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
pruned=proof['collapsed_identity_luts'][0];assert pruned['all_aliases_four_state_proved']
alias={tuple(k):tuple(v) for k,v in pruned['alias_pairs']}
def root(t):
 seen=set()
 while t in alias:assert t not in seen;seen.add(t);t=alias[t]
 return t
from camera_joint_native import OUT
for n in pruned['cells']:assert g.pop(n)[0]=='EFX_LUT4'
for kind,pins in g.values():
 for pin,ts in pins.items():
  if pin not in OUT[kind]:pins[pin]=[root(t) for t in ts]
names=sorted(n for n in set(r['changed'])|set(r['added']) if n in g and g[n][0]=='EFX_LUT4');assert names
assert all(n in b for n in names)
source_inputs={t for n in names for p,ts in g[n][1].items() if p!='O' for t in ts if t[0]=='net'}
mapped_inputs={t for n in names for p,ts in b[n][1].items() if p!='O' for t in ts if t[0]=='net'}
bound,rows=bind_control_cut(g,b,{t:t for t in source_inputs},mapped_inputs,BD)
v=['module tb;reg[3:0] stim;integer i,j;wire['+str(len(names)-1)+':0] old_q,new_q;']
for idx,n in enumerate(names):
 leaves=list(dict.fromkeys(t for p,ts in g[n][1].items() if p!='O' for t in ts if t[0]=='net'));assert len(leaves)<=4
 for role,gr,ps,out in [('s',g,pa,'old_q'),('m',b,pb,'new_q')]:
  def expr(t):
   if t[0]=='literal':return t[1]
   if role=='m':t=bound[t]
   return 'stim['+str(leaves.index(t))+']'
  ports=','.join('.I'+str(i)+'('+expr(gr[n][1].get('I'+str(i),[('literal',"1'bz",None)])[0])+')' for i in range(4))
  v.append(f"EFX_LUT4 #(.LUTMASK(16'd{ps[n]['LUTMASK'][1]})) {role}{idx}({ports},.O({out}[{idx}]));")
v+=['initial begin for(i=0;i<256;i=i+1)begin for(j=0;j<4;j=j+1)case((i>>(j*2))&3)0:stim[j]=0;1:stim[j]=1;2:stim[j]=1\'bx;3:stim[j]=1\'bz;endcase #2;']
v += [f'if(old_q[{i}]!==new_q[{i}])$fatal(1,"LUT_MAP_DIFFERENCE cell={n} vector=%0d",i);' for i,n in enumerate(names)]
v+=['end $display("SHARED_FRONTIER_MAPPED_LUTS_PASS");$finish;end endmodule']
O.mkdir();(O/'tb.sv').write_text('\n'.join(v));shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',O/'efx_lut4.v');(O/'binding.json').write_text(json.dumps(rows,indent=2));inputs={str(f):sha(f) for f in [Path(__file__),N/'native_source.v',N/'shared-frontier-edit.json',BD/'outflow/ti60f225_oob.map.v',BD/'retained-map-audit.json',O/'tb.sv',O/'efx_lut4.v',O/'binding.json']};(O/'inputs.json').write_text(json.dumps(inputs,indent=2))
os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';MS=Path('E:/quatra/questa_fse/win64');run([MS/'vlib.exe','work'],O,'vlib.log',30)
body,_=run([MS/'vlog.exe','-sv','efx_lut4.v','tb.sv'],O,'compile.log',60);assert 'Errors: 0, Warnings: 0' in body
body,_=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],O,'simulation.log',90);assert 'SHARED_FRONTIER_MAPPED_LUTS_PASS' in body and 'Errors: 0, Warnings: 0' in body
for f,h in inputs.items():assert sha(f)==h
result=dict(status='ACTUAL_VENDOR_SHARED_FRONTIER_ALL_CHANGED_LUTS_MAP_PASS',source_sha256=key,map_sha256=mk[1],pruned_luts_four_state_proof=pruned,luts=len(names),four_state_cases_per_lut=256,comparisons=len(names)*256,inputs_sha256=sha(O/'inputs.json'),seconds=time.monotonic()-start,hardware_access=False,board_release=False);(O/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
