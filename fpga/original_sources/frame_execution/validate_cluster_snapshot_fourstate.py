"""Replay a passed native cluster's exact frozen RTL with official four-state IP."""
import argparse
import os
import json
import re
import shutil
import sys
from pathlib import Path
from run_units import ROOT, HERE, MS, run, sha

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--selected-cases', type=int, nargs='+')
p.add_argument('--simulator',choices=['modelsim','questa'],default='modelsim')
p.add_argument('--timeout',type=int,default=900)
p.add_argument('--dsp24-proof',type=Path,help='Existing hash-bound four-state proof of fixed-production DSP24 simulation view')
a = p.parse_args()
assert 30<=a.timeout<=900
if a.simulator=='questa':
    MS=Path('E:/quatra/questa_fse/win64')
    assert (MS/'vsim.exe').is_file()
    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
root = a.root.resolve()
assert root.is_relative_to(ROOT/'tmp')
root.mkdir(parents=True, exist_ok=False)
ref = a.reference.resolve()
passed = json.loads((ref/'result.json').read_text())
assert passed['status'] == 'NATIVE_LOCAL_CLUSTER_NUMERIC_PROTOCOL_PASS'
assert sha(ref/'native_inputs.json') == passed['inputs_sha256']
native = json.loads((ref/'native_inputs.json').read_text())
snapshot = Path(native['snapshot'])
assert sha(snapshot/'inputs.json') == native['source_manifest_sha256']
source = json.loads((snapshot/'inputs.json').read_text())
vectors = snapshot/'vectors'
for name, digest in source['vectors'].items():
    assert sha(vectors/name) == digest
if a.selected_cases is not None:
    original_vectors=vectors;vectors=root/'vectors'
    shutil.copytree(original_vectors,vectors)
    rows=(vectors/'cases.txt').read_text().splitlines()[1:]
    by_id={int(row.split()[0]):row for row in rows}
    assert len(a.selected_cases)==len(set(a.selected_cases))
    selected=[by_id[i] for i in a.selected_cases]
    kinds=[int(row.split()[2]) for row in selected]
    assert sum(x!=y for x,y in zip(kinds,kinds[1:]))>=4
    assert any([int(row.split()[j]) for j in (2,3,4)]==[1,80,2] for row in selected)
    (vectors/'cases.txt').write_text(str(len(selected))+'\n'+'\n'.join(selected)+'\n')
vector_hashes={f.name:sha(f) for f in vectors.iterdir() if f.is_file()}
tb_source=next(Path(row['snapshot']) for row in source['inputs'] if Path(row['snapshot']).name=='tb_compute_column_cluster.sv')
if 'residual_abort_checks!=5' in tb_source.read_text():
    assert 9 in [int(line.split()[0]) for line in (vectors/'cases.txt').read_text().splitlines()[1:]],'Residual active-cancel test requires case9'
records = []
primitive_proof=None
if a.dsp24_proof:
    proof_root=a.dsp24_proof.resolve()
    proof=json.loads((proof_root/'result.json').read_text())
    evidence=json.loads((proof_root/'inputs.json').read_text())
    assert proof['status']=='DSP24_BOARD_PIPE_HIERARCHY_EQUIVALENCE_PASS' and proof['compact_board']
    assert sha(proof_root/'inputs.json')==proof['inputs_sha256']
    assert sha(proof_root/'simulation.log')==proof['log_sha256']
    assert sha(proof_root/'efx_dsp24_flat.v')==proof['flat_sha256']==evidence['flat_sha256']
    primitive_proof=dict(root=str(proof_root),result_sha256=sha(proof_root/'result.json'),original_sha256=evidence['original_sha256'],view_sha256=proof['flat_sha256'])
    assert sum(Path(row['snapshot']).name=='efx_dsp24.v' for row in source['inputs'])==1
for row in source['inputs']:
    src = Path(row['snapshot'])
    assert sha(src) == row['sha256']
    dst = root/src.name
    shutil.copy2(src, dst)
    rec=dict(snapshot=str(dst), sha256=sha(dst))
    if primitive_proof and src.name=='efx_dsp24.v':
        assert sha(src)==primitive_proof['original_sha256']
        shutil.copy2(proof_root/'efx_dsp24_flat.v',dst)
        rec.update(sha256=sha(dst),vendor_original_sha256=primitive_proof['original_sha256'],view='guarded fixed-production simulation specialization')
    records.append(rec)
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,
    reference_sha256=sha(ref/'result.json'), vectors=str(vectors),
    vector_hashes=vector_hashes, selected_cases=a.selected_cases, simulator=str(MS), timeout=a.timeout, dsp24_proof=primitive_proof, runner_sha256=sha(__file__)), indent=2))
run([MS/'vlib.exe', 'work'], root, 'vlib.log')
body, _ = run([MS/'vlog.exe', '-sv', '-timescale', '1ns/1ps',
               *[r['snapshot'] for r in records]], root, 'compile.log')
assert 'Errors: 0, Warnings: 0' in body
dump = root/'readback.hex'
checker = HERE.parent/'coefficient_store/validate_readback.py'
do = 'onerror {quit -code 1}\nrun 120 us\n'
do += 'if {[examine -radix unsigned sim:/tb_compute_column_cluster/verify_checkpoint] != 1} {error "Missing actual readback"}\n'
do += 'puts [exec {'+Path(sys.executable).as_posix()+'} -B {'+checker.as_posix()+'} --image-dir {'+vectors.as_posix()+'} --readback {'+dump.as_posix()+'}]\n'
do += 'force -deposit sim:/tb_compute_column_cluster/software_verified 1\nrun -all\nquit -code 0\n'
(root/'run.do').write_text(do, encoding='ascii')
body, seconds = run([MS/'vsim.exe', '-c', *(['-voptargs=+acc'] if a.simulator=='questa' else []), 'work.tb_compute_column_cluster',
    '-gSEED=71', '+VECTORS='+vectors.as_posix(), '+DUMP='+dump.as_posix(),
    '-do', 'do run.do'], root, 'simulation.log', timeout=a.timeout)
lines = re.findall(r'^# (PASS COMPUTE_CLUSTER.*)$', body, re.M)
if a.selected_cases is None:
    assert lines == [passed['line']]
else:
    assert len(lines)==1 and f'blocks={len(a.selected_cases)*2} ' in lines[0]
assert 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and 'Errors: 0, Warnings: 0' in body
for row in records:
    assert sha(row['snapshot']) == row['sha256']
for name,digest in vector_hashes.items():assert sha(vectors/name)==digest
result = dict(status='FROZEN_CLUSTER_OFFICIAL_FOUR_STATE_SUBSET_PASS' if a.selected_cases else 'FROZEN_CLUSTER_OFFICIAL_FOUR_STATE_PASS', line=lines[0], selected_cases=a.selected_cases,
    seconds=seconds, inputs_sha256=sha(root/'inputs.json'), log_sha256=sha(root/'simulation.log'),
    full_frame=False, pnr=False)
if primitive_proof:
    result.update(status='FROZEN_CLUSTER_QUALIFIED_DSP24_FOUR_STATE_PASS',dsp24_proof=primitive_proof,unmodified_vendor_models=False)
(root/'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
