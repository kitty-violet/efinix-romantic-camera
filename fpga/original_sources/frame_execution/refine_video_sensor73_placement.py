"""Refine the fitting sensor73 placement, with every ordinary cell still free.

Uses the official detailed placer and records a separate result before routing.
No old route, timing result, or bitstream is reused as evidence for this result.
"""
import json, shutil, subprocess, time
from pathlib import Path
from run_units import ROOT, EF, sha
from windows_process_tree_safe import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/video-sensor73-place2'
DEST=ROOT/'tmp/frame-execution/video-sensor73-refine3'
TIMING=ROOT/'tmp/frame-execution/video-sensor73-canonical3'

def main():
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
    evidence=[DEST/'runner.py',TIMING/'result.json',TIMING/'outflow/canonical_setup.rpt',
        BASE/'physical_inputs.json',BASE/'outflow/ti60f225_oob.place',
        BASE/'outflow/ti60f225_oob.netlist',BASE/'outflow/ti60f225_oob.vdb',
        BASE/'fixed_dsp.placeloc',BASE/'ti60f225_oob.sdc']
    review=dict(status='DETAILED_PLACEMENT_PLANNED_NOT_RELEASE',source=str(BASE),
        optimization='TIMING_1',budget_seconds=600,hard_anchors=404,ordinary_anchors=0,
        old_route_imported=False,source_map_vdb_constraints_unchanged=True,
        rationale=[
            'Same placement routed three times; the final two recipes both leave WNS-.238ns. No further unchanged-placement routing recipe sweep.',
            'Actual worst path crosses eight STOP control LUTs before a long carry chain. Consecutive control cells travel (120,99)->(145,121)->(145,124)->(115,81); associated loader/DMA/RAW negative endpoints are interconnect dominated.',
            'Use official detailed placement with timing priority to reconsider the entire existing placement. No hand-moved ordinary cell or new location lock; retain required hard macro/cascade anchors.',
            'Before any new route, inspect current fit, all hard anchors, shared controls, whole-chip congestion/weighted length and the full prior 2000-path population. Placement alone is not release.'
        ],evidence={str(f):sha(f) for f in evidence},hardware_access=False,board_release=False)
    meta['placement_refinement']=review
    (DEST/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
    (DEST/'refinement-inputs.json').write_text(json.dumps(review,indent=2))
    cmd=DEST/'refine.cmd'
    cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts detailed_place optimization_level=TIMING_1 --timeout 600\nexit /b %errorlevel%\n')
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
