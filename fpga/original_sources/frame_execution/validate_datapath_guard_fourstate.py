"""Replay exactly the frozen native guard source in ModelSim with official IP models."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--audit-existing',action='store_true');a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
if a.audit_existing:assert root.is_dir() and not (root/'result.json').exists()
else:root.mkdir(parents=True,exist_ok=False)
ref=a.reference.resolve();passed=json.loads((ref/'result.json').read_text());manifest=json.loads((ref/'inputs.json').read_text())
assert passed['status']=='INTEGRATED_DATAPATH_GUARD_NATIVE_PASS' and sha(ref/'inputs.json')==passed['inputs_sha256']
records=[]
for r in manifest['inputs']:
 src=Path(r['snapshot']);assert sha(src)==r['sha256'];dst=root/src.name
 if not a.audit_existing:shutil.copy2(src,dst)
 assert sha(dst)==r['sha256']
 records.append(dict(snapshot=str(dst),sha256=sha(dst)))
if a.audit_existing:
 original=json.loads((root/'inputs.json').read_text())
 assert original['inputs']==records and original['reference_sha256']==sha(ref/'result.json')
 body=(root/'compile.log').read_text()
else:
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,reference_sha256=sha(ref/'result.json'),runner_sha256=sha(__file__)),indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
active=bool(passed.get('active_lines'))
tb=(root/'tb_cnn_datapath_guard.sv').read_text()
params=['-gACTIVE_TEST='+str(int(active))] if 'parameter ACTIVE_TEST=0;' in tb else []
if a.audit_existing:
 body=(root/'simulation.log').read_text()
 h,m,s=map(int,re.findall(r'Elapsed time: (\d+):(\d+):(\d+)',body)[-1]);seconds=h*3600+m*60+s
else:
 body,seconds=run([MS/'vsim.exe','-c',*params,'work.tb_cnn_datapath_guard','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
assert not re.search(r'\bFAIL\b|\*\* (Error|Fatal)',body)
lines=re.findall(r'^# (PASS DATAPATH_GUARD.*)$',body,re.M);assert lines==[passed['line']]
active_lines=re.findall(r'^# (PASS DATAPATH_ACTIVE_CANCEL.*)$',body,re.M)
assert active_lines==passed.get('active_lines',[])
# This focused TB observes only control/AXI and explicitly leaves diagnostic
# OUTPUTS unused. Never waive a missing input or any other warning class.
unused_outputs=set('done_valid busy error hsvalid hssuccess hbusy herror load_ready verify_ready diag_accept diag_response diag_data diag_error diag_return_bank diag_return_address diag_return_tag load_complete store_error error_code image_tag loaded_words cycles reads writes acks dcycles dbeats arb_busy arena_router_busy'.split())
# Current extended TB connects these outputs. Keep compatibility with frozen
# old 13-case TBs without broadening the whitelist to other warning classes.
if 'parameter ACTIVE_TEST=0;' in tb:
 unused_outputs-=set('done_valid busy error hsvalid hssuccess hbusy load_ready verify_ready writes acks'.split())
 # Four further diagnostic outputs are observed by hierarchical assertions;
 # ModelSim therefore reports no missing-port warning for those outputs.
 unused_outputs-=set('herror reads arb_busy arena_router_busy'.split())
missing=re.findall(r"Missing connection for port '([^']+)'",body)
warnings=re.findall(r'\*\* Warning: \(vsim-(\d+)\)',body)
assert set(missing)==unused_outputs and len(missing)==len(unused_outputs)
assert warnings.count('3017')==1 and warnings.count('3722')==len(unused_outputs) and len(warnings)==len(unused_outputs)+1
assert 'Errors: 0, Warnings: '+str(len(unused_outputs)+1) in body
for r in records:assert sha(r['snapshot'])==r['sha256']
(root/'frozen_audit.py').write_bytes(Path(__file__).read_bytes())
result=dict(status='INTEGRATED_DATAPATH_GUARD_FOUR_STATE_PASS',line=lines[0],active_lines=active_lines,seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),audit_sha256=sha(root/'frozen_audit.py'),reused_simulation=a.audit_existing,reviewed_unused_output_warnings=sorted(missing),full_frame=False,production_scheduler=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
