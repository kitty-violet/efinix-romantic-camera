"""Exhaustively compare every changed cut using the vendor's real X/Z model."""
import argparse,json,os,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,EF,sha,run
from camera_ready_factor import BASE,circuit,plan,cone,ZERO
MS=Path('E:/quatra/questa_fse/win64')
def render(text,cases,tag):
 g,ps,drivers=circuit(text);lines=[]
 for ci,c in enumerate(cases):
  cells=cone(g,drivers,c['target'],set(c['boundary']));nodes={t:f'v[{i}]' for i,t in enumerate(c['boundary'])};nodes[ZERO]="1'b0"
  lines.append(f'module cone_{ci}_{tag}(input [{len(c["boundary"])-1}:0] v,output result);')
  for i,n in enumerate(cells):
   out=g[n][1]['O'][0];nodes[out]=f'w{i}';lines.append(f'wire w{i};')
  for i,n in enumerate(cells):
   p=g[n][1];value=ps[n]['LUTMASK'][1]
   ports=[f'.I{k}({nodes[p.get("I"+str(k),[ZERO])[0]]})' for k in range(4)]+[f'.O({nodes[p["O"][0]]})']
   lines.append(f"EFX_LUT4 #(.LUTMASK(16'h{value:04x})) u{i}("+','.join(ports)+');')
  lines += [f'assign result={nodes[c["target"]]};','endmodule']
 return '\n'.join(lines)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 src=a.source.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 r=json.loads((src/'result.json').read_text());assert r['status']=='READY_FACTOR_GRAPH_AND_EXHAUSTIVE_BINARY_PASS'
 for f,h in r['inputs'].items():assert sha(f)==h
 for f,h in r['files'].items():assert sha(src/f)==h
 old=(BASE/'outflow/ti60f225_oob.map.v').read_text();new=(src/'native_source.v').read_text();defs,cases=plan(old)
 for tag,text in [('old',old),('new',new)]:(out/f'cones_{tag}.v').write_text(render(text,cases,tag))
 lines=['`timescale 1ns/1ps','module tb_ready;integer count=0,q;','function symbol(input integer v);case(v)0:symbol=0;1:symbol=1;2:symbol=1\'bx;default:symbol=1\'bz;endcase endfunction']
 total=0
 for i,c in enumerate(cases):
  n=len(c['boundary']);lines += [f'reg [{n-1}:0] v{i}=0;wire o{i},n{i};',f'cone_{i}_old a{i}(v{i},o{i});cone_{i}_new b{i}(v{i},n{i});']
 lines += ['initial begin #2;']
 for i,c in enumerate(cases):
  n=len(c['boundary']);total+=4**n
  lines += [f'for(integer x=0;x<{4**n};x++)begin q=x;',f'for(integer k=0;k<{n};k++)begin v{i}[k]=symbol(q&3);q=q>>2;end #1;',f'if(o{i}!==n{i})$fatal(1,"MISMATCH {c["name"]} input=%b old=%b new=%b",v{i},o{i},n{i});count++;end',f'$display("CASE_PASS {c["name"]} count=%0d",count);v{i}=0;#1;']
 lines += [f'if(count!={total})$fatal(1,"COVERAGE");$display("PASS READY_FACTOR_FOURSTATE cases=%0d",count);$finish;end','endmodule']
 (out/'tb_ready.sv').write_text('\n'.join(lines));shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',out/'efx_lut4.v')
 files={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.sv','.v')}
 m=dict(source_result_sha256=sha(src/'result.json'),runner_sha256=sha(__file__),source=str(src),files=files,expected_checks=total,hardware_access=False)
 (out/'inputs.json').write_text(json.dumps(m,indent=2))
 for v in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):os.environ[v]='E:/quatra/LR-195771_License.dat'
 try:
  run([MS/'vlib.exe','work'],out,'vlib.log',20)
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*files],out,'compile.log',35);assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_ready','-do','onerror {quit -code 1}; run -all; quit -code 0'],out,'simulation.log',90)
  assert f'PASS READY_FACTOR_FOURSTATE cases={total}' in body and 'Errors: 0, Warnings: 0' in body
  for f,h in files.items():assert sha(out/f)==h
  result=dict(status='READY_FACTOR_EXHAUSTIVE_OFFICIAL_FOURSTATE_PASS',checks=total,cuts=len(cases),seconds=secs,inputs_sha256=sha(out/'inputs.json'),log_sha256=sha(out/'simulation.log'),hardware_access=False,board_release=False)
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as exc:
  (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise
if __name__=='__main__':main()
