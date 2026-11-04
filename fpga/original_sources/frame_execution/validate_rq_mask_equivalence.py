"""Four-state equivalence of duplicate operand-mask removal, including masked X."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
base=ROOT/'tmp/frame-execution/reuse-integrated-map1';records=[]
ref=base/'shared_rq_oc8_cluster.sv';dst=root/'rq_mask_reference.sv'
dst.write_text(ref.read_text().replace('module shared_rq_oc8_cluster #(','module rq_mask_reference #('))
records.append(dict(path=str(ref),snapshot=str(dst),sha256=sha(dst),source_sha256=sha(ref),rename_only=True))
for src in (base/'requantize_i32_mul18_q17_pipeline.sv',HERE/'variants/specialized_rq/shared_rq_oc8_cluster.sv',HERE/'tb_rq_mask_equivalence.sv'):
 dst=root/src.name;shutil.copy2(src,dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_rq_mask_equivalence','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
lines=re.findall(r'^# (PASS RQ_MASK_EQUIVALENCE.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for row in records:assert sha(row['snapshot'])==row['sha256']
r=dict(status='RQ_DUPLICATE_MASK_FOUR_STATE_EQUIVALENCE_PASS',line=lines[0],seconds=seconds,
 inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_frame=False)
(root/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
