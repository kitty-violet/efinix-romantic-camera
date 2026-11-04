"""Refine the fitting sensor73 placement, with every ordinary cell still free.

Uses the official detailed placer and records a separate result before routing.
No old route, timing result, or bitstream is reused as evidence for this result.
"""
import json, shutil, subprocess, time
from pathlib import Path
from run_units import ROOT, EF, sha
from windows_process_tree_safe import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/video-sensor73-place2'
DEST=ROOT/'tmp/frame-execution/video-sensor73-refine4-congestion'
TIMING=ROOT/'tmp/frame-execution/video-sensor73-canonical3'

def main():
    failed=ROOT/'tmp/frame-execution/video-sensor73-route4-refined/failure.json'
    assert failed.is_file(), 'Wait for bounded TIMING_1 route to finish; do not overlap trials'
    prior_stages=json.loads(failed.with_name('stages.json').read_text())
    assert prior_stages[-1]['status']=='finished' and prior_stages[-1]['exit_code']==1
    prior_route=ROOT/'tmp/frame-execution/video-sensor73-refine3/outflow/ti60f225_oob.route.out'
    assert 'Routing for the circuit has not been converging for the last 3 iterations.' in prior_route.read_text()
    if DEST.exists():
        assert not (DEST/'refine.cmd').exists(), 'A launched experiment cannot be overwritten'
        assert sha(DEST/'native_source.v')==sha(BASE/'native_source.v')
    timing=json.loads((TIMING/'result.json').read_text())
    assert timing['rows']['setup']['worst_ns']==-0.238
    assert timing['rows']['hold']['worst_ns']==0.026
    shutil.copytree(BASE,DEST,dirs_exist_ok=True,ignore=shutil.ignore_patterns(
        '*.rpt','*.log','*.troutingtraces','*.route','*.bit',
        'result.json','stages.json','failure.json','route-review.json'))
    meta=json.loads((BASE/'physical_inputs.json').read_text())
    for row in meta['inputs']:
        row['snapshot']=str(DEST/Path(row['snapshot']).relative_to(BASE))
        assert sha(row['snapshot'])==row['sha256']
    shutil.copy2(Path(__file__),DEST/'runner.py')
    evidence=[failed,failed.with_name('stages.json'),prior_route,ROOT/'tmp/frame-execution/video-sensor73-refine3/refinement-global-review.json',DEST/'runner.py',TIMING/'result.json',TIMING/'outflow/canonical_setup.rpt',
        BASE/'physical_inputs.json',BASE/'outflow/ti60f225_oob.place',
        BASE/'outflow/ti60f225_oob.netlist',BASE/'outflow/ti60f225_oob.vdb',
        BASE/'fixed_dsp.placeloc',BASE/'ti60f225_oob.sdc']
    review=dict(status='DETAILED_PLACEMENT_PLANNED_NOT_RELEASE',source=str(BASE),
        optimization='CONGESTION_1',budget_seconds=600,hard_anchors=404,ordinary_anchors=0,
        old_route_imported=False,source_map_vdb_constraints_unchanged=True,
        rationale=[
            'The TIMING_1 detailed placement improved estimated timing but increased all three congestion peaks; its subsequent bounded router failed. Preserve that failure, do not repeat it.',
            'Use the official congestion-oriented detailed placer from the unchanged original global-placement checkpoint. Keep original C2 global placement, 404 hard macro anchors and zero ordinary-cell anchors.',
            'This is a geometry change aimed at measured routing congestion, not an unchanged-placement route or seed sweep. Do not change RTL, capacity, clocks, pins, or original SDC.',
            'After placement, inspect capacity, all hard macros, shared control nets, congestion/weighted length and all prior 2000 setup paths before deciding on any route. No placement-only release.'
        ],evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
    meta['placement_refinement']=review
    (DEST/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
    (DEST/'refinement-inputs.json').write_text(json.dumps(review,indent=2))
    cmd=DEST/'refine.cmd'
    cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts detailed_place optimization_level=CONGESTION_1 --timeout 600\nexit /b %errorlevel%\n')
    start=time.monotonic()
    with (DEST/'refine.log').open('x') as log:
        proc=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=DEST,stdout=log,stderr=subprocess.STDOUT)
        try: rc=proc.wait(timeout=645)
        except subprocess.TimeoutExpired:
            (DEST/'timeout-stop.json').write_text(json.dumps(stop_launched_tree(proc),indent=2))
            raise
    result=dict(status='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE',exit_code=rc,
        seconds=round(time.monotonic()-start,3),inputs_sha256=sha(DEST/'refinement-inputs.json'),
        hardware_access=False,board_release=False)
    (DEST/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__': main()
