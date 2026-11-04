"""Fixed local response contract: real RAM collision/credit and owner routing."""
import argparse,json,re,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
from frozen_sources import frozen_paths

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
variant=HERE/'variants/direct_sram_response'
records=[]
try:
 paths=[EF/'sim_models/verilog/efx_dpram10.v',variant/'tensor_arena12.sv',
        *frozen_paths([HERE/'arena_client_router2.sv'],a.base),HERE/'tb_tensor_arena12.sv',HERE/'tb_arena_client_router2.sv']
 for src in paths:
  body=src.read_text();dst=root/src.name
  if src.name=='tb_tensor_arena12.sv':
   assert body.count('rsp_tag[p*16+:16]!==tags[p]')==1
   body=body.replace('rsp_tag[p*16+:16]!==tags[p]',"rsp_tag[p*16+:16]!==16'd0")
  if src.name=='tb_arena_client_router2.sv':
   for owner in ('a','b'):
    needle="st[p*16+:16]!=={1'b"+str(int(owner=='b'))+",5'(p),10'(seen_"+owner+"[p])}"
    assert body.count(needle)==1;body=body.replace(needle,"st[p*16+:16]!==16'd0")
   body=body.replace('reg [31:0] rng=71;', 'parameter SEED=71;reg [31:0] rng=SEED;')
   body=body.replace('integer ticks,total;', '''integer replacements=0;
 always @(posedge clk)if(!reset)
  for(integer pp=0;pp<24;pp=pp+1)
   if(mv[pp] && mr[pp] && sv[pp] && sr[pp])replacements=replacements+1;
 integer ticks,total;''')
   body=body.replace('$display("PASS ARENA_ROUTER', '''if(replacements<100)$fatal(1,"FAIL same-clock replacement coverage");
  $display("OWNER_REPLACEMENTS %0d",replacements);
  $display("PASS ARENA_ROUTER''')
  dst.write_text(body);records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),tag_echo_removed=True),indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
 assert 'Errors: 0, Warnings: 0' in body
 runs=[]
 for top,label in (('tb_tensor_arena12','TENSOR_ARENA'),('tb_arena_client_router2','ARENA_ROUTER')):
  for seed in (71,391,1027):
   log=f'{top}_{seed}.log'
   body,seconds=run([MS/'vsim.exe','-c',f'-gSEED={seed}','work.'+top,'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,log,timeout=180)
   assert 'Errors: 0, Warnings: 0' in body and not re.search(r'\bFAIL\b|\*\* (Error|Fatal)',body)
   lines=re.findall(r'^# (PASS '+label+r'.*)$',body,re.M);assert len(lines)==1
   runs.append(dict(top=top,seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(root/log)))
 for r in records:assert sha(r['source'])==r['source_sha256'] and sha(r['snapshot'])==r['sha256']
 result=dict(status='DIRECT_SRAM_PORTS_FOUR_STATE_PASS',runs=runs,inputs_sha256=sha(root/'inputs.json'),full_frame=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
