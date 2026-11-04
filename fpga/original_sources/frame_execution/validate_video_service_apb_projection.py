"""Actual vendor CPU-state/APB step replay, including asynchronous reset transitions."""
import argparse,json,os,pickle,shutil,time,re
from pathlib import Path
from run_units import ROOT,EF,sha
from continuous_control_native import connections
from map_video_service_safe import run
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();N=a.native.resolve();O=a.root.resolve();assert not O.exists()
r=json.loads((N/'same-edge-edit.json').read_text());P=Path(r['parent']);key,g,pa=pickle.loads((P/'zero-source-parse.pkl').read_bytes());nk,ng,np=pickle.loads((N/'zero-source-parse.pkl').read_bytes());assert key==sha(P/'native_source.v') and nk==sha(N/'native_source.v')
dr,loads=connections(g);state=[f'cpu/state_oh[{i}]~FF' for i in range(10)];ff=state+['ceg_net4871~FF'];owned=set(ff);todo=[t for n in ff for p,ts in g[n][1].items() if p not in ('Q','CLK','SR') for t in ts]+g['LUT__133025'][1]['O']
clock_boundary=('net','process_clk_150',None);reset_boundary=('net','w_arstn',None);pipe_boundary=g['ceg_net4871~FF'][1]['D'][0]
while todo:
 t=todo.pop();n=dr.get(t)
 if t in (clock_boundary,reset_boundary,pipe_boundary) or n in owned or n is None or g[n][0]!='EFX_LUT4':continue
 owned.add(n);todo += [t for p,ts in g[n][1].items() if p!='O' for t in ts]
og={n:g[n] for n in owned};op={n:pa[n] for n in owned}
for n in ('SERVICE62_APB_NEXT','SERVICE62_APB_ENABLE'):og[n]=ng[n];op[n]=np[n]
# A distinct output token is needed for the new projection in the comparison harness.
apb=('net','apb_enable',None);projected=('net','projected_apb',None);og['SERVICE62_APB_ENABLE']=(ng['SERVICE62_APB_ENABLE'][0],{**ng['SERVICE62_APB_ENABLE'][1],'Q':[projected]})
tokens={t for kind,pins in og.values() for ts in pins.values() for t in ts if t[0]=='net'};tn={t:'n'+str(i) for i,t in enumerate(sorted(tokens,key=str))};dr2,_=connections(og);free=sorted(tokens-set(dr2),key=str)
clk=('net','process_clk_150',None);arst=('net','w_arstn',None);pipe=g['ceg_net4871~FF'][1]['D'][0]
random=[t for t in free if t not in (clk,arst,pipe)];assert clk in free and arst in free and pipe in free
v=['`timescale 1ns/1ps','module tb;integer i,j,k,checks=0,resets=0;reg[9:0] visited=0;reg[31:0] rnd=32\'hdeadbeef;']
v+=['reg '+tn[t]+';' for t in free]+['wire '+tn[t]+';' for t in tokens if t not in free]
def expr(t):return tn[t] if t[0]=='net' else t[1]
for idx,n in enumerate(sorted(og)):
 kind,pins=og[n];pins=dict(pins)
 if kind=='EFX_LUT4':
  for pin in ('I0','I1','I2','I3'):pins.setdefault(pin,[('literal',"1'bz",None)])
 ps=','.join('.'+k+'('+str(w)+"'d"+str(x)+')' for k,(w,x) in op[n].items());ports=','.join('.'+pin+'('+expr(ts[0])+')' for pin,ts in pins.items());v.append(f'{kind} #({ps}) cell_{idx}({ports});')
v += ['always #5 '+tn[clk]+'=~'+tn[clk]+';', 'task compare; begin #1; if('+tn[apb]+'!=='+tn[projected]+')$fatal(1,"APB projection mismatch cycle=%0d old=%b new=%b",i,'+tn[apb]+','+tn[projected]+'); checks=checks+1;end endtask',
 'initial begin '+tn[clk]+'=0;'+tn[arst]+'=1;'+tn[pipe]+'=0;'+''.join(tn[t]+'=0;' for t in random)+' #1;'+tn[arst]+'=0; repeat(3)@(negedge '+tn[clk]+'); compare;'+tn[arst]+'=1;',
 'for(i=0;i<20000;i=i+1) begin @(negedge '+tn[clk]+');']
for t in random:v.append('rnd={rnd[30:0],rnd[31]^rnd[21]^rnd[1]^rnd[0]}; '+tn[t]+'=rnd[0];')
v += [tn[pipe]+'='+tn[arst]+';', 'if((i%197)==73)begin #2;'+tn[arst]+'=0;'+tn[pipe]+'=0;resets=resets+1;compare;end',
 'if((i%197)==78)'+tn[arst]+'=1;', '@(posedge '+tn[clk]+');compare;', 'visited=visited|{'+','.join(tn[g[n][1]['Q'][0]] for n in reversed(state))+'};',
 'end',tn[arst]+'=1;'+tn[pipe]+'=1;repeat(3)@(negedge '+tn[clk]+');',
 'for(k=0;k<10;k=k+1)begin for(j=0;j<100;j=j+1)begin @(negedge '+tn[clk]+');']
for t in random:v.append('rnd={rnd[30:0],rnd[31]^rnd[21]^rnd[1]^rnd[0]}; '+tn[t]+'=rnd[0];')
v.append('case(k)')
for k in range(10):v.append(str(k)+':begin '+''.join('force '+tn[g[n][1]['Q'][0]]+'='+str(int(i==k))+';' for i,n in enumerate(state))+'end')
v+=['endcase visited=visited|(1<<k);@(posedge '+tn[clk]+');#1;'+''.join('release '+tn[g[n][1]['Q'][0]]+';' for n in state)+'compare;end end',
 '$display("APB_PROJECTION_PASS checks=%0d resets=%0d visited=%h",checks,resets,visited);$finish;end endmodule']
O.mkdir();(O/'tb.sv').write_text('\n'.join(v));start=time.monotonic()
for name in ('efx_lut4.v','efx_ff.v'):shutil.copy2(EF/'sim_models/verilog'/name,O/name)
inputs={str(f):sha(f) for f in [Path(__file__),P/'native_source.v',N/'native_source.v',N/'same-edge-edit.json',O/'tb.sv',O/'efx_lut4.v',O/'efx_ff.v']};(O/'inputs.json').write_text(json.dumps(inputs,indent=2))
os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';MS=Path('E:/quatra/questa_fse/win64')
run([MS/'vlib.exe','work'],O,'vlib.log',30);body,_=run([MS/'vlog.exe','-sv','efx_lut4.v','efx_ff.v','tb.sv'],O,'compile.log',60);assert 'Errors: 0, Warnings: 0' in body
body,_=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1};onbreak {quit -code 1};run -all;quit -code 0'],O,'simulation.log',90)
assert 'APB_PROJECTION_PASS' in body and 'Errors: 0, Warnings: 0' in body
match=re.search(r'APB_PROJECTION_PASS checks=(\d+) resets=(\d+) visited=([0-9a-fA-F]+)',body);assert match and int(match[1])==21103 and int(match[2])==102 and int(match[3],16)==1023
for f,h in inputs.items():assert sha(f)==h
(O/'result.json').write_text(json.dumps(dict(status='ACTUAL_VENDOR_SAME_EDGE_APB_RESET_REPLAY_PASS',source_sha256=nk,parent_sha256=key,cycles=21000,checks=int(match[1]),asynchronous_reset_reassertions=int(match[2]),state_coverage=int(match[3],16),directed_legal_state_single_step_checks=1000,scope='Binary randomized external CPU controls, actual four-state EFX_LUT4/FF state/reset semantics; 100 one-edge trials of each legal state injected only in testbench. Formal induction covers all legal binary states.',inputs_sha256=sha(O/'inputs.json'),seconds=time.monotonic()-start,hardware_access=False,board_release=False),indent=2));print(body[-1800:])
