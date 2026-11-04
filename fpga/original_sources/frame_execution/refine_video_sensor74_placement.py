"""Bounded official detailed-placement test from identical packed clusters.

This does not import an old route or impose ordinary-cell location constraints.
The fresh sensor74 netlist is used, and all placement/resource checks must run.
"""
import json,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/video-sensor74-place1'
REFERENCE=ROOT/'tmp/frame-execution/video-service-place50-input-local'
DEST=ROOT/'tmp/frame-execution/video-sensor74-refine1'

def main():
    assert not DEST.exists()
    for name in ('ti60f225_oob.clstdb','ti60f225_oob.cnst','ti60f225_oob.htree'):
        assert sha(BASE/'outflow'/name)==sha(REFERENCE/'outflow'/name),name
    assert sha(BASE/'ti60f225_oob.sdc')==sha(REFERENCE/'ti60f225_oob.sdc')
    assert sha(BASE/'ti60f225_oob.peri.xml')==sha(REFERENCE/'ti60f225_oob.peri.xml')
    assert sha(BASE/'fixed_dsp.placeloc')==sha(REFERENCE/'fixed_dsp.placeloc')
    # Keep fresh synthesized/packed databases, discard only copied stage reports.
    shutil.copytree(BASE,DEST,ignore=shutil.ignore_patterns('*.rpt','*.log','*.out','result.json','stages.json','failure.json','capacity_gate.json','capacity-failure-review.json'))
    meta=json.loads((BASE/'physical_inputs.json').read_text())
    for row in meta['inputs']:
        row['snapshot']=str(DEST/Path(row['snapshot']).relative_to(BASE))
        assert sha(row['snapshot'])==row['sha256']
    proof=dict(status='IDENTICAL_PACKED_CLUSTERS_EXISTING_PLACEMENT_REFINEMENT_NOT_RELEASE',
        fresh_netlist=str(BASE/'outflow/ti60f225_oob.netlist'),reference=str(REFERENCE),
        packed_clusters_byte_identical=True,ordinary_anchors=0,old_route_imported=False,
        source_and_new_netlist_unchanged=True,budget_seconds=180,
        rationale='Five existing sensor LUT functions and ROM only; packed cluster database and membership are byte-identical. Use official detailed_place with existing placement as a starting point, allowing the tool to check/adjust it. Never reuse old STA or final resources as current evidence.',
        evidence={str(f):sha(f) for f in [Path(__file__),BASE/'physical_inputs.json',BASE/'outflow/ti60f225_oob.netlist',BASE/'outflow/ti60f225_oob.clstdb',REFERENCE/'outflow/ti60f225_oob.clstdb',REFERENCE/'outflow/ti60f225_oob.place',BASE/'fixed_dsp.placeloc',BASE/'ti60f225_oob.sdc']},hardware_access=False,board_release=False)
    (DEST/'refinement-inputs.json').write_text(json.dumps(proof,indent=2))
    meta['placement_refinement']=proof
    (DEST/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
    shutil.copy2(REFERENCE/'outflow/ti60f225_oob.place',DEST/'outflow/ti60f225_oob.place')
    cmd=DEST/'refine.cmd'
    cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts detailed_place --timeout 180\nexit /b %errorlevel%\n')
    start=time.monotonic()
    try:
        with (DEST/'refine.log').open('x') as log:
            proc=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=DEST,stdout=log,stderr=subprocess.STDOUT)
            try:rc=proc.wait(timeout=225)
            except subprocess.TimeoutExpired:
                (DEST/'timeout-stop.json').write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
        result=dict(status='DETAILED_PLACEMENT_EXPERIMENT_FINISHED_NOT_RELEASE',exit_code=rc,seconds=time.monotonic()-start,
            inputs_sha256=sha(DEST/'refinement-inputs.json'),hardware_access=False,board_release=False)
        (DEST/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (DEST/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
