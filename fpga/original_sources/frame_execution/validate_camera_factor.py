"""Actual mapped decoder and control cuts with official Efinity LUT models."""
import argparse,json,os,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
from camera_factor_board import BOARD,TRANSFORM,PARENT,transform_check,audit_delta
from native_local_drivers import graph
from probe_full_native_split_map import params
from camera_joint_native import OUT,ZERO
MS=Path('E:/quatra/questa_fse/win64')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir();audit_delta(BOARD);proof=transform_check()
 old=(PARENT/'outflow/ti60f225_oob.map.v').read_text();src=(BOARD/'native_source.v').read_text();new=(BOARD/'outflow/ti60f225_oob.map.v').read_text()
 og,sg,ng=graph(old),graph(src),graph(new);op,np=params(old),params(new);mapping={}
 for n,(k,pins) in sg.items():
  if n not in ng:continue
  for pin in pins.keys()&ng[n][1].keys():
   for x,y in zip(pins[pin],ng[n][1][pin]):
    assert x not in mapping or mapping[x]==y,(n,pin)
    mapping[x]=y
 oldlut={p['O'][0]:n for n,(k,p) in og.items() if k=='EFX_LUT4'}
 newlut={p['O'][0]:n for n,(k,p) in ng.items() if k=='EFX_LUT4'}
 oldnames=set();newnames=set();oldleaves=set();newleaves=set();pairs=[]
 def gather(t,g,drivers,names,leaves):
  if t[0]=='literal':return
  if t not in drivers:leaves.add(t);return
  n=drivers[t]
  if n in names:return
  names.add(n)
  for pin,ts in g[n][1].items():
   if pin!='O':
    for x in ts:gather(x,g,drivers,names,leaves)
 changed={n for r in proof['factor_stages'] for row in r['outputs'] for n,pin in row['loads']}
 for n in sorted(changed):
  a,b=og[n][1]['CE'][0],ng[n][1]['CE'][0]
  if (a,b) in pairs:continue
  pairs.append((a,b));gather(a,og,oldlut,oldnames,oldleaves);gather(b,ng,newlut,newnames,newleaves)
 assert {mapping[t] for t in oldleaves}==newleaves
 leaves=sorted(oldleaves,key=str);lines=[f'module tb_factor; reg [{len(leaves)-1}:0] vector; reg [12:0] dv; integer checks=0; integer pessimism_relaxed=0; integer both_unknown=0; integer rnd=32\'h21958403;']
 def emit(g,ps,names,bindings,prefix):
  wires=dict(bindings)
  def wire(t):
   if t[0]=='literal':return t[1]
   if t not in wires:wires[t]=prefix+'_w'+str(len(wires));lines.append('wire '+wires[t]+';')
   return wires[t]
  for i,n in enumerate(sorted(names)):
   k,pins=g[n];assert k=='EFX_LUT4'
   defs=','.join(f".{key}({w}'h{v:x})" for key,(w,v) in ps[n].items())
   ports=','.join('.'+pin+'('+wire(ts[0])+')' for pin,ts in pins.items())
   lines.append(f'{k} #({defs}) {prefix}_c{i}({ports});')
  return wire
 left=emit(og,op,oldnames,{t:f'vector[{i}]' for i,t in enumerate(leaves)},'old')
 right=emit(ng,np,newnames,{mapping[t]:f'vector[{i}]' for i,t in enumerate(leaves)},'new')
 compare=[(left(a),right(b)) for a,b in pairs]
 dr=[tuple(t) for t in proof['divider_inputs']]
 dl=emit(og,op,proof['removed_divider_cells'],{t:f'dv[{i}]' for i,t in enumerate(dr)},'divold')
 dn=emit(ng,np,[n for n in ng if n.startswith('JOINT_DIV_')],{mapping[t]:f'dv[{i}]' for i,t in enumerate(dr)},'divnew')
 dc=[(dl(tuple(row['old'])),dn(mapping[tuple(row['new'])])) for row in proof['divider_outputs']]
 lines+=['initial begin vector=0;dv=0;','for(integer i=0;i<8192;i=i+1) begin dv=i;#1;']
 for i,(a,b) in enumerate(dc):lines.append(f'if({a} !== {b} || $isunknown({b})) $fatal(1,"DIVIDER {i} vector=%d",i);checks=checks+1;')
 lines+=['end','$display("PASS FACTOR_DIVIDER checks=%0d",checks);checks=0;',f'for(integer i=0;i<8192;i=i+1) begin for(integer j=0;j<{len(leaves)};j=j+1) vector[j]=$random(rnd);#1;']
 for i,(a,b) in enumerate(compare):lines.append(f'if({a} !== {b} || $isunknown({b})) $fatal(1,"CONTROL {i} sample=%d",i);checks=checks+1;')
 lines+=['end','$display("PASS FACTOR_CONTROLS checks=%0d",checks);checks=0;',f'for(integer i=0;i<{2*len(leaves)};i=i+1) begin for(integer j=0;j<{len(leaves)};j=j+1) vector[j]=$random(rnd);vector[i/2]=(i%2)?1\'bz:1\'bx;#1;']
 # Restructuring may eliminate X pessimism. For every sampled X/Z case, a
 # defined original output must remain exactly defined and equal. Also log
 # old-X/new-known and both-X; never call this universal four-state equality.
 for i,(a,b) in enumerate(compare):
  lines.append(f'if(!$isunknown({a}) && {a} !== {b}) $fatal(1,"CONTROL_UNKNOWN_REGRESSION {i} sample=%d",i);if($isunknown({a})) begin if($isunknown({b})) both_unknown=both_unknown+1;else pessimism_relaxed=pessimism_relaxed+1;end checks=checks+1;')
 lines+=['end','$display("PASS FACTOR_XZ_NO_KNOWN_REGRESSION checks=%0d relaxed=%0d both_unknown=%0d",checks,pessimism_relaxed,both_unknown);$finish;end endmodule']
 tb=out/'tb_factor.sv';tb.write_text('\n'.join(lines));shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',out/'efx_lut4.v')
 meta=dict(map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v'),map_result_sha256=sha(BOARD/'result.json'),transform_sha256=sha(TRANSFORM/'result.json'),runner_sha256=sha(__file__),files={f.name:sha(f) for f in (tb,out/'efx_lut4.v')},cut_leaves=len(leaves),control_output_pairs=len(pairs),old_luts=len(oldnames),new_luts=len(newnames),changed_ce_loads=len(changed),hardware_access=False)
 (out/'inputs.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
 for k in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):os.environ[k]='E:/quatra/LR-195771_License.dat'
 run([MS/'vlib.exe','work'],out,'vlib.log',timeout=15)
 body,_=run([MS/'vlog.exe','-sv','efx_lut4.v',tb.name],out,'compile.log',timeout=60);assert 'Errors: 0' in body
 body,secs=run([MS/'vsim.exe','-c','work.tb_factor','-do','onerror {quit -code 1}; run -all; quit -code 0'],out,'simulation.log',timeout=120)
 assert 'PASS FACTOR_DIVIDER checks=98304' in body and f'PASS FACTOR_CONTROLS checks={8192*len(pairs)}' in body
 m=re.search(r'PASS FACTOR_XZ_NO_KNOWN_REGRESSION checks=(\d+) relaxed=(\d+) both_unknown=(\d+)',body);assert m and int(m[1])==len(pairs)*2*len(leaves)
 assert not re.search(r'\*\* (Error|Fatal):',body)
 result=dict(status='FACTOR_ACTUAL_MAP_CONTROLS_AND_DECODER_PASS',decoder_checks=98304,control_checks=8192*len(pairs),xz_checks=int(m[1]),xz_resolved=int(m[2]),xz_both_unknown=int(m[3]),seconds=secs,inputs_sha256=sha(out/'inputs.json'),log_sha256=sha(out/'simulation.log'),hardware_access=False,board_release=False,scope='Exact Boolean proof plus actual-map known/random and single-X/Z cut tests. No universal four-state equivalence or board PASS claim.')
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
