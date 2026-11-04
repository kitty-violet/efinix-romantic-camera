"""Response-retention candidate: four-state defined-payload/transaction tests."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
from frozen_sources import frozen_paths

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--candidate',type=Path,default=HERE/'variants/held_reader/unified_column_reader.sv');a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir(parents=True)
reference=frozen_paths([HERE/'unified_column_reader.sv'],a.base)[0]
candidate=a.candidate.resolve();assert candidate.is_relative_to(HERE/'variants')
# Reuse the qualified 120-case stimulus, not its old candidate or verdict.
tb=ROOT/'tmp/frame-execution/reader-select-gate2/tb_multi_issue_reader.sv'
body=tb.read_text()
body=body.replace('qr[1][p]=qr[0][p];','qr[1][p]=(!memory_live[1][p] || (rv[1][p] && rr[1][p])) && (rng[p] || rng[(p+7)%32]);')
start=body.index(' integer edges=0;')
end=body.index(' always @(posedge clk)begin',start)
body=body[:start]+''' // Retaining responses deliberately changes lock/ready timing. Compare
 // packet sequence and read counts below, and assert each held payload.
 reg [767:0] held_output[0:1]; reg [1:0] was_held=0;
 integer response_holds=0;
 always @(posedge clk)begin
  if(reset || cancel)was_held<=0;
  else for(integer e=0;e<2;e=e+1)begin
   if(was_held[e] && checking)begin
    if(!ov[e])$fatal(1,"lost output valid");
    for(integer row=0;row<6;row=row+1)if(row<source_rows)begin
     if(od[e][row*64+:64]!==held_output[e][row*64+:64])$fatal(1,"A changed while held");
     if(pair_mode && od[e][(row+6)*64+:64]!==held_output[e][(row+6)*64+:64])$fatal(1,"B changed while held");
    end
   end
   was_held[e]<=ov[e] && !ordy[e];held_output[e]<=od[e];
   if(e==1)for(integer pp=0;pp<24;pp=pp+1)if(rv[e][pp] && !rr[e][pp])begin
    response_holds=response_holds+1;
    if(!locked[e][pp])$fatal(1,"response lost its lock");
   end
  end
 end
'''+body[end:]
body=body.replace('PASS READER_SELECT_CYCLE','PASS HELD_READER_PACKET')
body=body.replace('$display("COVERAGE packets=', '$display("RESPONSE_HOLDS %0d",response_holds);\n  if(response_holds<1000)$fatal(1,"missing held response coverage");\n  $display("COVERAGE packets=')
records=[]
for src,name in [(reference,'reader_reference.sv'),(candidate,candidate.name)]:
 dst=r/name
 if name=='reader_reference.sv':dst.write_text(src.read_text().replace('module unified_column_reader(','module reader_reference('))
 else:shutil.copy2(src,dst)
 records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
(r/'tb_multi_issue_reader.sv').write_text(body)
records.append(dict(snapshot=str(r/'tb_multi_issue_reader.sv'),sha256=sha(r/'tb_multi_issue_reader.sv')))
shutil.copy2(__file__,r/'frozen_runner.py')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,stimulus_source=str(tb),stimulus_sha256=sha(tb),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
compiled,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in compiled
body,secs=run([MS/'vsim.exe','-c','work.tb_multi_issue_reader','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log')
lines=re.findall(r'^# (PASS HELD_READER_PACKET.*)$',body,re.M)
assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['snapshot'])==x['sha256']
result=dict(status='HELD_READER_FOUR_STATE_PACKET_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(r/'inputs.json'),
 log_sha256=sha(r/'simulation.log'),reference_sha256=sha(reference),candidate_sha256=sha(candidate),actual_ram=False,full_frame=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
