"""Reader cycle/defined-payload comparison, including 120 layouts/modes/cancel."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
from frozen_sources import frozen_paths
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
ref=frozen_paths([HERE/'unified_column_reader.sv'],a.base)[0];cand=HERE/'variants/logical_row_capture/unified_column_reader.sv'
tb=ROOT/'tmp/frame-execution/reader-select-gate2/tb_multi_issue_reader.sv';body=tb.read_text()
needle='if(od[0]!==od[1])$fatal(1,"reader valid data cycle mismatch case=%0d index=%0d",case_id,oi[0]);'
assert body.count(needle)==1
body=body.replace(needle,'''// Only configured source rows are consumed by dispatch. Unused rows
   // need not contain the old physical-slot cache's stale payload.
   for(integer row=0;row<6;row=row+1)if(row<source_rows)begin
    if(od[0][row*64+:64]!==od[1][row*64+:64])$fatal(1,"capture A cycle mismatch case=%0d row=%0d",case_id,row);
    if(pair_mode && od[0][(row+6)*64+:64]!==od[1][(row+6)*64+:64])$fatal(1,"capture B cycle mismatch");
   end''')
body=body.replace('PASS READER_SELECT_CYCLE','PASS LOGICAL_CAPTURE_CYCLE')
records=[]
for src,name in ((ref,'reader_reference.sv'),(cand,cand.name)):
 dst=root/name
 if name=='reader_reference.sv':dst.write_text(src.read_text().replace('module unified_column_reader(','module reader_reference('))
 else:shutil.copy2(src,dst)
 records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
dst=root/'tb_multi_issue_reader.sv';dst.write_text(body);records.append(dict(source=str(tb),source_sha256=sha(tb),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),unused_rows_compared=False),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_multi_issue_reader','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=120)
line=re.findall(r'^# (PASS LOGICAL_CAPTURE_CYCLE.*)$',body,re.M);assert len(line)==1 and 'Errors: 0, Warnings: 0' in body
for r in records:assert sha(r['snapshot'])==r['sha256'] and sha(r['source'])==r['source_sha256']
result=dict(status='LOGICAL_CAPTURE_FOUR_STATE_CYCLE_PASS',line=line[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),actual_ram=False,full_frame=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
