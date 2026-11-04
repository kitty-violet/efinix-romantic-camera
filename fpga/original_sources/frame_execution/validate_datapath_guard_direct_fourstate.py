"""Run the unchanged guard TB directly in four-state ModelSim; no native prerequisite.
Same public sources and official models, all expected checks and warnings retained.
This path avoids a Verilator hierarchical-force inlining internal error.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frozen-base',type=Path,required=True);p.add_argument('--overrides-dir',type=Path);p.add_argument('--vector-prefix',action='store_true');a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
from run_units import HERE,EF
from map_parallel_candidates import selection
from frozen_sources import frozen_paths
paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+selection('datapath')[1]+[HERE/'tb_cnn_datapath_guard.sv']
paths=frozen_paths(paths,a.frozen_base)
if a.overrides_dir:
 override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
 paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
projection=None
if a.vector_prefix:
 from prefix_history_projection import project,require_proof
 projection=require_proof()
records=[]
for f in paths:
 dst=root/f.name
 if a.vector_prefix and f.name=='shared_rq_oc8_cluster.sv':dst.write_text(project(f.read_text()))
 else:shutil.copy2(f,dst)
 records.append(dict(path=str(f),source_sha256=sha(f),snapshot=str(dst),sha256=sha(dst)))
reference=ROOT/'tmp/frame-execution/shared-taps-guard-fourstate1/result.json'
passed=json.loads(reference.read_text())
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,prefix_projection=projection,reference_sha256=sha(reference),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
active=bool(passed.get('active_lines'))
tb=(root/'tb_cnn_datapath_guard.sv').read_text()
params=['-gACTIVE_TEST='+str(int(active))] if 'parameter ACTIVE_TEST=0;' in tb else []
body,seconds=run([MS/'vsim.exe','-c',*params,'work.tb_cnn_datapath_guard','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
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
result=dict(status='INTEGRATED_DATAPATH_GUARD_FOUR_STATE_PASS',line=lines[0],active_lines=active_lines,seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),audit_sha256=sha(root/'frozen_audit.py'),reused_simulation=False,reviewed_unused_output_warnings=sorted(missing),full_frame=False,production_scheduler=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
