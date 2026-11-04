"""Four-state, same-cycle reader comparison against the selected board."""
import argparse, json, re, shutil
from pathlib import Path
from run_units import ROOT, HERE, MS, run, sha
from frozen_sources import frozen_paths

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--reference-board',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
a=p.parse_args();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists()
r.mkdir(parents=True)
reference=frozen_paths([HERE/'unified_column_reader.sv'],a.reference_board)[0]
candidate=a.candidate.resolve();assert candidate.is_relative_to(HERE/'variants')
tb=HERE/'variants/specialized_b1/tb_multi_issue_reader.sv'
body=tb.read_text().replace('// Independent variable-latency memories compare packet streams, not cycle timing.',
    '// Identical stalls and latencies compare requests, control and valid data every cycle.')
body=body.replace('for(integer c=0;c<88;c=c+1)', 'for(integer c=0;c<120;c=c+1)')
body=body.replace('if(c>=60)begin', 'if(c>=60 && c<88)begin')
body=body.replace(".cmd_source_width(11'd512),.cmd_source_origin(11'd0)", '.cmd_source_width(source_width),.cmd_source_origin(origin)')
body=body.replace('wire[1:0] cr', "reg[10:0] source_width=512,origin=0;\n wire[1:0] cr")
body=body.replace('if(e==0)reader_reference', 'if(e==0)begin:REF\n  reader_reference')
body=body.replace('else unified_column_reader', 'end else begin:CAND\n  unified_column_reader')
body=body.replace(' end\n always @(negedge clk)begin', ' end\n end\n always @(negedge clk)begin')
body=body.replace('source_rows=pair_mode ?', '''if(c>=88)begin
    // All column origins/parities, right clamping, NN repeats and maximum
    // 320-column descriptors. Origin 0 makes source column parity explicit.
    pair_mode=c%3!=0;rgb=c%3==1;pw=rgb;nn=c%3==0;
    rows=pair_mode && !rgb ? 2:3;groups=1;out_groups=1;group_q=0;
    pixels=c<108 ? 9:pair_mode && !rgb ? 255:320;
    source_width=1280;origin=0;
    ox=11'(c%4);
    if(c%4==3)ox=source_width-(pair_mode && !rgb ? (pixels*2-1):pixels);
    if(nn)ox=11'(c%4);
   end
   source_rows=pair_mode ?''')
start=body.index(' always @(negedge clk)begin')
end=body.index(' always @(posedge clk)begin',start)
body=body[:start]+''' always @(negedge clk)begin
  rng=next_rng(rng);
  ordy={2{rng[0] || rng[4]}};pa[0]=rng[23:0]|(rng[23:0]<<1);pa[1]=pa[0];
  rng=next_rng(rng);
  for(integer p=0;p<24;p=p+1)begin
   qr[0][p]=(!memory_live[0][p] || (rv[0][p] && rr[0][p])) && (rng[p] || rng[(p+7)%32]);
   qr[1][p]=qr[0][p];
  end
 end
 integer edges=0;
 always @(posedge clk)if(!reset && !cancel)begin
  edges=edges+1;
  if({cr[0],ov[0],ow[0],dv[0],busy[0],error[0],oi[0],qv[0],rr[0],locked[0],nreads[0]} !==
     {cr[1],ov[1],ow[1],dv[1],busy[1],error[1],oi[1],qv[1],rr[1],locked[1],nreads[1]})$fatal(1,"reader control cycle mismatch");
  for(integer p=0;p<24;p=p+1)if(qv[0][p] &&
    {qa[0][p*10+:10],qt[0][p*16+:16]}!=={qa[1][p*10+:10],qt[1][p*16+:16]})$fatal(1,"reader request cycle mismatch");
  if(ov[0])begin
   if(od[0]!==od[1])$fatal(1,"reader valid data cycle mismatch case=%0d index=%0d",case_id,oi[0]);
   if(ENGINE[1].CAND.dut.q_b[ENGINE[1].CAND.dut.head]!=ENGINE[1].CAND.dut.q_a[ENGINE[1].CAND.dut.head] &&
      ENGINE[1].CAND.dut.q_b[ENGINE[1].CAND.dut.head]!=2'(ENGINE[1].CAND.dut.q_a[ENGINE[1].CAND.dut.head]+1))$fatal(1,"adjacency contract");
  end
 end
'''+body[end:]
body=body.replace('((p+e)%28)','(p%28)')
body=body.replace('PASS MULTI_ISSUE_READER cases=88', 'PASS READER_SELECT_CYCLE cases=120')
records=[]
for src,name in [(reference,'reader_reference.sv'),(candidate,candidate.name),(tb,'reference_tb.sv')]:
 dst=r/name
 if name=='reader_reference.sv':dst.write_text(src.read_text().replace('module unified_column_reader(','module reader_reference(')+'\n')
 else:shutil.copy2(src,dst)
 records.append(dict(source=str(src),snapshot=str(dst),source_sha256=sha(src),sha256=sha(dst)))
(r/'tb_multi_issue_reader.sv').write_text(body)
records.append(dict(snapshot=str(r/'tb_multi_issue_reader.sv'),sha256=sha(r/'tb_multi_issue_reader.sv')))
shutil.copy2(__file__,r/'frozen_runner.py')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
compiled,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',r/'reader_reference.sv',r/candidate.name,r/'tb_multi_issue_reader.sv'],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in compiled
body,secs=run([MS/'vsim.exe','-c','work.tb_multi_issue_reader','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log')
lines=re.findall(r'^# (PASS READER_SELECT_CYCLE.*)$',body,re.M)
assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:
 assert sha(x['snapshot'])==x['sha256']
 if 'source' in x:assert sha(x['source'])==x['source_sha256']
result=dict(status='READER_SELECT_FOUR_STATE_CYCLE_PASS',line=lines[0],seconds=secs,
 inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),reference_sha256=sha(reference),candidate_sha256=sha(candidate),full_frame=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
