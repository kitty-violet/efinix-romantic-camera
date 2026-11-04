"""Four-state cycle A/B for the compact delay representation of stride dispatch."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--reference',type=Path,default=HERE/'variants/stride_decimate/unified_column_dispatch32.sv')
p.add_argument('--candidate',type=Path,default=HERE/'variants/stride_compact_history/unified_column_dispatch32.sv')
p.add_argument('--fullrate',action='store_true',help='Use original paired stride input/warmup protocol')
p.add_argument('--poison-trailing-half',action='store_true',help='Prove stride trailing B half cannot reach any valid MAC operand')
a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
records=[]
try:
 for src,name in [(a.reference.resolve(),'reference.sv'),(a.candidate.resolve(),'candidate.sv'),(EF/'sim_models/verilog/efx_srl8.v','efx_srl8.v')]:
  dst=root/name;body=src.read_text()
  if name=='reference.sv':body=body.replace('module unified_column_dispatch32','module dispatch_reference')
  dst.write_text(body);records.append(dict(path=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
 src=HERE/'tb_unified_column_dispatch32.sv';body=src.read_text()
 if not a.fullrate:body=body.replace('wire warm=!pair_mode && !pw && ci==0;','wire warm=(!pair_mode || !rgb) && !pw && ci==0;')
 start=body.index(' wire dr,di,dd,db,de,ds;');end=body.index(' integer checks=')
 replacement=''' wire oi,ow,od,ob,oe,os,orr;wire [31:0] ov,ofirst,olast;wire [191:0] ot;wire [2047:0] oa,oweight;wire [63:0] og;wire [511:0] otag;wire [255:0] om;
 dispatch_reference reference(.clk(clk),.reset(reset),.cancel(cancel),.cmd_valid(cv),.cmd_ready(orr),
 .cmd_pair(pair_mode),.cmd_rgb(rgb),.cmd_mode_pw(pw),.cmd_stem(stem),.cmd_pixels(pixels),.cmd_rows(rows),.cmd_in_groups(gi),.cmd_out_groups(go),.cmd_group(group_q),.cmd_mask(mask),
 .in_valid(iv),.in_ready(oi),.in_data(data),.in_index(ci),.in_warmup(warm),.weight_valid(wv),.weight_ready(ow),.weight_data(weight),.weight_index(wi),
 .done_valid(od),.done_ready(1'b1),.busy(ob),.error(oe),.pool_valid(ov),.pool_ready(ready),.pool_first(ofirst),.pool_last(olast),.pool_terms(ot),
 .pool_data(oa),.pool_weights(oweight),.pool_group(og),.pool_tag(otag),.pool_mask(om),.phase_advance(os),.cycles(),.wait_input_cycles(),.wait_pool_cycles());
'''
 body=body[:start]+replacement+body[end:]
 body=body.replace('for(integer c=0;c<40;c=c+1)','for(integer c=0;c<80;c=c+1)').replace('completed!=40','completed!=80')
 body=body.replace('    @(negedge clk);\n   end','''    @(negedge clk);
    if(c>=40 && ticks==17)begin
     cancel=1;@(negedge clk);iv=0;wv=0;cancel=0;break;
    end
   end''')
 body=body.replace('PASS UNIFIED_DISPATCH','PASS COMPACT_DISPATCH_AB')
 if a.poison_trailing_half:
  assert a.fullrate
  # Only the candidate gets X. Every valid MAC owner, held token, weight,
  # first/last and output tag is still compared against the clean reference.
  body=body.replace(' unified_column_dispatch32 candidate(',''' wire poison=pair_mode && !rgb && ci=={4'd0,pixels};
 wire [767:0] candidate_data={poison ? {384{1'bx}}:data[767:384],data[383:0]};
 integer poisoned_accepts=0;
 always @(posedge clk)if(!reset && !cancel && poison && iv && ni)poisoned_accepts=poisoned_accepts+1;
 unified_column_dispatch32 candidate(''')
  body=body.replace('.in_data(data),','.in_data(candidate_data),',1)
  body=body.replace("   rows=pair_mode && !rgb ?", "   if(c>=20 && c<40 && pair_mode && !rgb)pixels=c==23 ? 9'd64:c==28 ? 9'd128:c==33 ? 9'd129:9'd160;\n   rows=pair_mode && !rgb ?")
  body=body.replace('  if(checks<10000', '  if(poisoned_accepts<8)$fatal(1,"poison coverage");\n  $display("POISON_ACCEPTED %0d",poisoned_accepts);\n  if(checks<10000')
 body=body.replace('completed=completed+1;', 'completed=completed+1;$display("CASE_COMPLETE c=%0d ticks=%0d checks=%0d",c,ticks,checks);')
 dst=root/'tb.sv';dst.write_text(body);records.append(dict(path=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
 # The prior official-SRL 40-case gate measured 120 s. 80 cases include
 # 40 short cancellation probes; 240 s is a bounded, evidence-based budget.
 body,seconds=run([MS/'vsim.exe','-c','work.tb_unified_column_dispatch32','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=240)
 lines=re.findall(r'^# (PASS COMPACT_DISPATCH_AB.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 for r in records:assert sha(r['path'])==r['source_sha256'] and sha(r['snapshot'])==r['sha256']
 result=dict(status='COMPACT_DISPATCH_FOUR_STATE_CYCLE_AB_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_frame=False,poison_trailing_half=a.poison_trailing_half)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
