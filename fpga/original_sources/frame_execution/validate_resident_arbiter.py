import argparse,json,re,shutil
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--arena',action='store_true')
p.add_argument('--override',type=Path);p.add_argument('--compare-reference',type=Path);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
records=[];top='arena_client_router2' if a.arena else 'resident_read_arbiter2';label='ARENA_ROUTER' if a.arena else 'RESIDENT_ARBITER'
paths=[HERE/(top+'.sv'),HERE/('tb_'+top+'.sv')]
if a.override:
 assert a.override.name==top+'.sv'
 paths[0]=a.override.resolve()
if a.arena:paths=[EF/'sim_models/verilog/efx_dpram10.v',HERE/'tensor_arena12.sv']+paths
for f in paths:
 dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
if a.compare_reference:
 assert a.arena and a.compare_reference.name==top+'.sv'
 reference=root/(top+'_reference.sv')
 text=a.compare_reference.read_text();assert text.count('module '+top+'(')==1
 reference.write_text(text.replace('module '+top+'(', 'module '+top+'_reference('))
 records.append(dict(path=str(a.compare_reference.resolve()),snapshot=str(reference),sha256=sha(reference),source_sha256=sha(a.compare_reference)))
 tb=root/('tb_'+top+'.sv');text=tb.read_text()
 # Drive both routers with identical live requests and memory responses;
 # compare every output at the settled falling edge, including stalled data.
 instance=re.search(r' arena_client_router2 dut\(.*?\);',text,re.S).group(0)
 outputs={'ar':24,'br':24,'asv':24,'bsv':24,'mv':24,'mw':24,'ma':240,'md':1536,'mt':384,'sr':24,'busy':1,'error':1}
 compare=instance.replace('arena_client_router2 dut(', 'arena_client_router2_reference reference(')
 for name in outputs:compare=compare.replace('('+name+')','(ref_'+name+')')
 declarations='\n'.join('wire [%d:0] ref_%s;'%(width-1,name) for name,width in outputs.items())
 actual=','.join(outputs);expected=','.join('ref_'+name for name in outputs)
 check='\ninteger comparisons=0;\nalways @(negedge clk)begin #1;if(!reset)begin comparisons=comparisons+1;if({'+actual+'}!=={'+expected+'})$fatal(1,"Router reference mismatch");end end\n'
 text=text.replace(instance,instance+'\n'+declarations+'\n'+compare+check)
 text=text.replace('$display("PASS ARENA_ROUTER', '$display("ROUTER_REFERENCE_COMPARISONS %0d",comparisons);$display("PASS ARENA_ROUTER')
 tb.write_text(text)
 next(r for r in records if Path(r['snapshot'])==tb)['sha256']=sha(tb)
(root/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_'+top,'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
lines=re.findall(r'^# (PASS '+label+r'.*)$',body,re.M);assert len(lines)==1 and 'FAIL' not in body
result=dict(status=label+'_SOURCE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'))
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
