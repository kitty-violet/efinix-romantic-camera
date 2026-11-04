"""Actual RAM + current router + two readers with a contending memory client.

Uses the 120-case qualified packet stimulus, but replaces its behavioral
memories with the exact selected 12-bank arena and two-client router. The
background client continuously reads/writes disjoint addresses on all ports.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
from frozen_sources import frozen_paths

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--base',type=Path,required=True);p.add_argument('--stimulus',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir(parents=True)
passed=json.loads((a.stimulus/'result.json').read_text());assert passed['status']=='HELD_READER_FOUR_STATE_PACKET_PASS'
tb=a.stimulus/'tb_multi_issue_reader.sv';body=tb.read_text()
body=body.replace('reg[23:0] qr[0:1],pa[0:1],rv[0:1],memory_live[0:1],held[0:1];',
 'wire[23:0] qr[0:1],rv[0:1];reg[23:0] pa[0:1],held[0:1];')
body=body.replace('reg[239:0] held_addr[0:1];reg[383:0] held_tag[0:1],rt[0:1];reg[1535:0] rd[0:1];',
 'reg[239:0] held_addr[0:1];reg[383:0] held_tag[0:1];wire[383:0] rt[0:1];wire[1535:0] rd[0:1];')
start=body.index('  rng=next_rng(rng);\n  for(integer p=0;p<24;p=p+1)begin\n   qr[0][p]=')
end=body.index('\n end',start)
body=body[:start]+body[end:]
body=body.replace('rv[e]<=0;memory_live[e]<=0;held[e]<=0;', 'held[e]<=0;')
start=body.index('     if(memory_live[e][p]')
end=body.index('\n    end\n    held[e]',start)
body=body[:start]+body[end:]
body=body.replace('begin rv[e]=0;memory_live[e]=0;held[e]=0;qr[e]=0;pa[e]=0;rt[e]=0;rd[e]=0;end',
                  'begin held[e]=0;pa[e]=0;end')
body=body.replace('repeat(4)@(negedge clk);reset=0;', '''repeat(4)@(negedge clk);reset=0;
  // Fill real RAM using legal requests, not hierarchical deposits.
  while(!init_done[0] || !init_done[1])@(negedge clk);
  repeat(4)@(negedge clk);''')
body=body.replace('PASS HELD_READER_PACKET cases=120 layouts=28 rejected_layouts=3',
                  'PASS HELD_READER_ARENA cases=32 selected_layouts=8 rejected_layouts=3')
# The complete 120-case behavioral test is a separate gate. This integration
# samples every mode, all three starting planes, clamps, NN, maximum width,
# and all nine cancellation points with actual primitive models/contending B.
cases=[0,7,14,21,28,35,42,49,56,5,6,12,15,40,63,64,65,67,68,69,70,71,88,89,90,91,92,108,109,110,114,119]
assert len(cases)==len(set(cases))==32
body=body.replace('for(integer c=0;c<120;c=c+1)begin',
 'for(integer c=0;c<120;c=c+1)if('+ ' || '.join('c=='+str(c) for c in cases)+')begin')
body=body.replace('checks<5000','checks<3000')
body=body.replace('$display("RESPONSE_HOLDS', '''if(bg_writes[0]<1000 || bg_writes[1]<1000 || bg_reads[0]<1000 || bg_reads[1]<1000 || contend[1]<1000)$fatal(1,"missing arena contention");
  $display("ARENA_BACKGROUND writes=%0d/%0d reads=%0d/%0d contending=%0d/%0d",bg_writes[0],bg_writes[1],bg_reads[0],bg_reads[1],contend[0],contend[1]);
  $display("RESPONSE_HOLDS''')
body=body.replace(' for(genvar e=0;e<2;e=e+1)begin:ENGINE', '''
 reg [1:0] init_done=0;
 integer bg_writes[0:1],bg_reads[0:1],contend[0:1];
 for(genvar e=0;e<2;e=e+1)begin:MEMORY
  wire [23:0] mv,mr,mw,msv,msr,sw,bready,bresponse;
  wire [239:0] ma;wire [1535:0] md;wire [383:0] mt;
  reg [23:0] bv=0,bw=0;reg[239:0] ba=0;reg[1535:0] bd=0;reg[383:0] bt=0;
  wire route_busy,route_error;
  reg[23:0] fired=0;
  integer seq[0:23],init_address=0;
  arena_client_router2 #(.A_EVEN_READ_ONLY(1)) route(.clk(clk),.reset(reset||cancel),
   .a_valid(qv[e]),.a_ready(qr[e]),.a_write(24'd0),.a_address(qa[e]),.a_data(1536'd0),.a_tag(qt[e]),.a_rsp_valid(rv[e]),.a_rsp_ready(rr[e]),
   .b_valid(bv),.b_ready(bready),.b_write(bw),.b_address(ba),.b_data(bd),.b_tag(bt),.b_rsp_valid(bresponse),.b_rsp_ready(24'hffffff),
   .mem_valid(mv),.mem_ready(mr),.mem_write(mw),.mem_address(ma),.mem_data(md),.mem_tag(mt),
   .mem_rsp_valid(msv),.mem_rsp_ready(msr),.busy(route_busy),.error(route_error));
  tensor_arena12 arena(.clk(clk),.reset(reset),.cancel(cancel),.req_valid(mv),.req_ready(mr),.req_write(mw),
   .req_address(ma),.req_data(md),.req_tag(mt),.rsp_valid(msv),.rsp_ready(msr),.rsp_data(rd[e]),.rsp_tag(rt[e]),.rsp_was_write(sw));
  always @(posedge clk)begin
   fired=bv & bready;
   if(!reset && !cancel)begin
    if(route_error)$fatal(1,"arena router error");
    for(integer pp=0;pp<24;pp=pp+1)begin
     if(qv[e][pp] && qr[e][pp] && qa[e][pp*10+:10]>=896)$fatal(1,"background address overlaps input");
     if(init_done[e] && fired[pp])begin
      if(bw[pp])bg_writes[e]=bg_writes[e]+1;else bg_reads[e]=bg_reads[e]+1;
     end
     if(checking && qv[e][pp] && bv[pp])contend[e]=contend[e]+1;
    end
   end
  end
  initial begin
   bg_reads[e]=0;bg_writes[e]=0;contend[e]=0;
   for(integer pp=0;pp<24;pp=pp+1)seq[pp]=0;
   wait(!reset);@(negedge clk);
   for(init_address=0;init_address<1024;init_address=init_address+1)begin
    bv=24'hffffff;bw=24'hffffff;
    for(integer pp=0;pp<24;pp=pp+1)begin
     ba[pp*10+:10]=10'(init_address);bd[pp*64+:64]=word_at(pp/2,init_address);bt[pp*16+:16]=16'(init_address);
    end
    do begin @(negedge clk);bv=bv & ~fired;end while(bv!=0);
   end
   wait(!route_busy);init_done[e]=1;
   forever begin
    @(negedge clk);
    for(integer pp=0;pp<24;pp=pp+1)if(reset || cancel)bv[pp]=0;
    else if(!bv[pp] || fired[pp])begin
     seq[pp]=seq[pp]+1;bv[pp]=1;bw[pp]=seq[pp]%2;
     ba[pp*10+:10]=10'(896+(pp%2)*32+(seq[pp]%32));
     bd[pp*64+:64]=64'(seq[pp]*37+pp);bt[pp*16+:16]=16'(seq[pp]);
    end
   end
  end
 end
 for(genvar e=0;e<2;e=e+1)begin:ENGINE''')
records=[]
for name in ('reader_reference.sv','unified_column_reader.sv'):
 src=a.stimulus/name;dst=r/name;shutil.copy2(src,dst)
 records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
paths=frozen_paths([HERE/'tensor_arena12.sv',HERE/'arena_client_router2.sv'],a.base)+[EF/'sim_models/verilog/efx_dpram10.v']
for src in paths:
 dst=r/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
(r/'tb_multi_issue_reader.sv').write_text(body)
records.append(dict(snapshot=str(r/'tb_multi_issue_reader.sv'),sha256=sha(r/'tb_multi_issue_reader.sv')))
shutil.copy2(__file__,r/'frozen_runner.py')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,stimulus_result_sha256=sha(a.stimulus/'result.json'),stimulus_sha256=sha(tb),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
compiled,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in compiled
body,secs=run([MS/'vsim.exe','-c','work.tb_multi_issue_reader','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=180)
lines=re.findall(r'^# (PASS HELD_READER_ARENA.*)$',body,re.M)
assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['snapshot'])==x['sha256']
result=dict(status='HELD_READER_ARENA_FOUR_STATE_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(r/'inputs.json'),
 log_sha256=sha(r/'simulation.log'),background=re.findall(r'^# ARENA_BACKGROUND.*$',body,re.M),actual_ram=True,full_frame=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
