"""Short official four-state skip protocol/numeric regression, two RTL versions."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
(root/'runner.py').write_bytes(Path(__file__).read_bytes());rows=[]
for name,src in [('reference',ROOT/'tmp/frame-execution/connected-board-source-capture1/assembly/fixed_skip_vectors4.sv'),('candidate',HERE/'variants/fixed_skip_credit/fixed_skip_vectors4.sv')]:
 r=root/name;r.mkdir();records=[]
 for f in (src,HERE/'variants/fixed_skip_credit_test/tb_skip_credit.sv',EF/'sim_models/verilog/efx_lut4.v'):
  dst=r/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst)))
 (r/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2)+'\n')
 run([MS/'vlib.exe','work'],r,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[i['snapshot'] for i in records]],r,'compile.log')
 assert 'Errors: 0, Warnings: 0' in body
 body,seconds=run([MS/'vsim.exe','-c','work.tb_skip_credit','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=90)
 lines=re.findall(r'^# (PASS SKIP_CREDIT.*)$',body,re.M)
 assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body and not re.search(r'\bFAIL\b|\*\* (Error|Fatal)',body)
 for i in records: assert sha(i['source'])==sha(i['snapshot'])==i['sha256']
 rows.append(dict(name=name,line=lines[0],seconds=seconds,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log')))
result=dict(status='SKIP_CREDIT_FOUR_STATE_NUMERIC_PROTOCOL_PASS',rows=rows,whole_board=False)
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
