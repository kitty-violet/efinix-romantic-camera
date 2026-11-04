"""Official primitive simulation of both joint changes from the actual map.

Complete known-input decoder cut, and four-state/reset/enable/history tests
for every replica. Whole-board graph binding is a separate required proof.
"""
import argparse,json,os,shutil,re
from pathlib import Path
from run_units import ROOT,EF,sha,run
from camera_joint_board import BOARD,TRANSFORM,PARENT,transform_check
from native_local_drivers import graph
from probe_full_native_split_map import params
MS=Path('E:/quatra/questa_fse/win64')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 proof=transform_check();result=json.loads((BOARD/'result.json').read_text());assert result['status']=='CAMERA_JOINT_FULL_BOARD_MAP_PASS'
 old=(PARENT/'outflow/ti60f225_oob.map.v').read_text();src=(BOARD/'native_source.v').read_text();new=(BOARD/'outflow/ti60f225_oob.map.v').read_text()
 og,sg,ng=graph(old),graph(src),graph(new);op,np=params(old),params(new)
 assert sha(BOARD/'outflow/ti60f225_oob.map.v')==result['map_sha256']
 mapping={}
 for n,(kind,pins) in sg.items():
  if n not in ng:continue
  for pin in pins.keys()&ng[n][1].keys():
   for x,y in zip(pins[pin],ng[n][1][pin]):
    assert x not in mapping or mapping[x]==y,(n,pin)
    mapping[x]=y
 roots=[tuple(t) for t in proof['divider_inputs']]
 newroots=[mapping[t] for t in roots]
 lines=['module tb_joint; reg [12:0] vector=0; reg clk=0; reg [4:0] d=0; integer checks=0;']
 def emit_cut(g,ps,names,inputs,outputs,prefix):
  wires={t:f'vector[{i}]' for i,t in enumerate(inputs)}
  def wire(t):
   if t[0]=='literal':return t[1]
   if t not in wires:
    w=f'{prefix}_w{len(wires)}';wires[t]=w;lines.append('wire '+w+';')
   return wires[t]
  for idx,n in enumerate(sorted(names)):
   kind,pins=g[n];assert kind=='EFX_LUT4'
   defs=','.join(f".{k}({w}'h{v:x})" for k,(w,v) in ps[n].items())
   ports=','.join('.'+p+'('+wire(ts[0])+')' for p,ts in pins.items())
   lines.append(f'{kind} #({defs}) {prefix}_cell{idx}({ports});')
  return [wire(t) for t in outputs]
 oo=[tuple(row['old']) for row in proof['divider_outputs']];no=[mapping[tuple(row['new'])] for row in proof['divider_outputs']]
 left=emit_cut(og,op,proof['removed_divider_cells'],roots,oo,'before')
 right=emit_cut(ng,np,[n for n in ng if n.startswith('JOINT_DIV_')],newroots,no,'after')
 pairs=[]
 for idx,(clone,original) in enumerate(proof['clone_orig'].items()):
  kind=ng[clone][0];assert kind==ng[original][0] and np[clone]==np[original]
  width=2 if kind=='EFX_ADD' else 1
  for nm,suffix in ((original,'o'),(clone,'n')):
   output=f'pair_{idx}{suffix}';lines.append(f'wire [{width-1}:0] {output};')
   if kind=='EFX_LUT4':ports=','.join(f'.I{i}(d[{i}])' for i in range(4))+f',.O({output})'
   elif kind=='EFX_FF':ports=f'.CLK(clk),.D(d[0]),.CE(d[1]),.SR(d[2]),.Q({output})'
   elif kind=='EFX_SRL8':ports=f'.CLK(clk),.D(d[0]),.CE(d[1]),.A(d[4:2]),.Q({output})'
   else:
    assert kind=='EFX_ADD';ports=f'.I0(d[0]),.I1(d[1]),.CI(d[2]),.O({output}[0]),.CO({output}[1])'
   defs=','.join(f".{k}({w}'h{v:x})" for k,(w,v) in np[nm].items())
   lines.append(f'{kind} #({defs}) replica_{idx}{suffix}({ports});')
  pairs.append((f'pair_{idx}o',f'pair_{idx}n'))
 lines.append('''function automatic symbol(input integer v);
 case(v) 0:symbol=0;1:symbol=1;2:symbol=1'bx;default:symbol=1'bz;endcase endfunction
 initial begin
 for(integer v=0;v<8192;v=v+1) begin vector=v; #1;
 ''')
 for i,(leftbit,rightbit) in enumerate(zip(left,right)):
  lines.append(f'if({leftbit} !== {rightbit} || $isunknown({rightbit})) $fatal(1,"DIVIDER_MISMATCH output={i} input=%d",v); checks=checks+1;')
 lines.append('end\n$display("PASS DIVIDER_KNOWN_INPUTS checks=%0d",checks); checks=0;')
 # Long known histories exercise all shift depths; all four-state controls
 # then exercise reset, disabled clocks, and unknown data handling identically.
 lines.append('for(integer v=0;v<2048;v=v+1) begin clk=0; if(v<1024) d=v; else d={symbol((v>>8)&3),symbol((v>>6)&3),symbol((v>>4)&3),symbol((v>>2)&3),symbol(v&3)}; #1; clk=1; #1;')
 for i,(leftbit,rightbit) in enumerate(pairs):
  lines.append(f'if({leftbit} !== {rightbit}) $fatal(1,"REPLICA_MISMATCH pair={i} input=%d",v); checks=checks+1;')
 lines += ['end',f'if(checks!={len(pairs)*2048}) $fatal(1,"COUNT");', '$display("PASS REPLICAS_FOURSTATE checks=%0d",checks); $finish; end endmodule']
 tb=root/'tb_joint.sv';tb.write_text('\n'.join(lines))
 models=['efx_lut4.v','efx_ff.v','efx_srl8.v','efx_add.v']
 for n in models:shutil.copy2(EF/'sim_models/verilog'/n,root/n)
 meta=dict(map_sha256=result['map_sha256'],map_result_sha256=sha(BOARD/'result.json'),transform_sha256=sha(TRANSFORM/'result.json'),runner_sha256=sha(__file__),files={n:sha(root/n) for n in models+[tb.name]},hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(meta,indent=2))
 for key in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):os.environ[key]='E:/quatra/LR-195771_License.dat'
 run([MS/'vlib.exe','work'],root,'vlib.log',timeout=15)
 body,_=run([MS/'vlog.exe','-sv',*models,tb.name],root,'compile.log',timeout=60);assert 'Errors: 0' in body
 body,secs=run([MS/'vsim.exe','-c','work.tb_joint','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=90)
 assert 'PASS DIVIDER_KNOWN_INPUTS checks=98304' in body
 assert f'PASS REPLICAS_FOURSTATE checks={len(pairs)*2048}' in body and not re.search(r'\*\* (Error|Fatal):',body)
 r=dict(status='CAMERA_JOINT_ACTUAL_MAP_PRIMITIVES_PASS',decoder_checks=8192*12,replica_checks=len(pairs)*2048,replicas=len(pairs),seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),hardware_access=False,board_release=False,
  scope='Actual mapped decoder exhaustively checked for known cut inputs; identical mapped replicas tested with four-state controls and sequential history. Not a full-board simulation or timing claim.')
 (root/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()
