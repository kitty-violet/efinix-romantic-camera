"""Review a newly completed detailed placement; retain the actual stage history."""
import argparse, json, re, shutil
from pathlib import Path
from run_units import ROOT,sha
from camera_ratio_before_place import metrics
from review_video_sensor73 import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    raw=json.loads((root/'result.json').read_text());assert raw['status']=='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE' and raw['exit_code']==0
    proof=json.loads((root/'refinement-inputs.json').read_text());assert raw['inputs_sha256']==sha(root/'refinement-inputs.json')
    for f,h in proof['evidence'].items():assert sha(f)==h,f
    base=Path(proof['source']);meta=json.loads((root/'physical_inputs.json').read_text())
    for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
    for suffix in ('.map.v','.vdb'):
        assert sha(root/'outflow'/('ti60f225_oob'+suffix))==sha(base/'outflow'/('ti60f225_oob'+suffix))
    assert sha(root/'outflow/ti60f225_oob.place')!=sha(base/'outflow/ti60f225_oob.place'),'Detailed placement did not move anything'
    for suffix in ('.place','.out.placeloc','.place.out','.place.rpt'):
        assert (root/'outflow'/('ti60f225_oob'+suffix)).stat().st_mtime>(root/'refine.cmd').stat().st_mtime
    current=metrics(root);assert current['placement_legal']
    assert all(r['used']<=r['capacity'] for r in current['resources'].values())
    shutil.copy2(root/'result.json',root/'refinement-result.json')
    inherited=json.loads((base/'before_route/result.json').read_text())
    stages=[dict(s,reused_from=str(base),not_reexecuted=True) for s in inherited['stages']]
    stages.append(dict(flow='pnr',phase='detailed_place_TIMING_1',exit_code=0,seconds=raw['seconds'],
        inputs_sha256=raw['inputs_sha256'],log_sha256=sha(root/'refine.log')))
    (root/'result.json').write_text(json.dumps(dict(status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE',stages=stages,
        source_unchanged=True,strict_sta_pass=False,board_download_allowed=False),indent=2))
    (root/'stages.json').write_text(json.dumps(stages,indent=2))
    run(root)
    # Same netlist allows an exact comparison of every original reported path.
    from review_input_isp_geometry import locations
    oldpos=locations(base/'outflow/ti60f225_oob.out.placeloc');newpos=locations(root/'outflow/ti60f225_oob.out.placeloc')
    report=ROOT/'tmp/frame-execution/video-sensor73-canonical3/outflow/canonical_setup.rpt'
    rows=[]
    for b in re.split(r'(?=^Slack\s*:)',report.read_text(),flags=re.M)[1:]:
        slack=float(re.search(r'^Slack\s*:\s*([\d.+-]+)',b)[1]);end=re.search(r'^Path End\s*:\s*(.*)',b,re.M)[1].strip()
        data=re.split(r'^Data Path\s*$',b,flags=re.M)[1].split('Capture Clock Path',1)[0];cells=[]
        for line in data.splitlines():
            m=re.match(r'\s*\(\d+,\d+\)\s+EFX_\w+(?:\s+\([^)]+\))?\s+[\d.-]+\s+[\d.-]+\s+(.*)',line)
            if m:
                n=m[1].strip().split('|')[0]
                if not cells or n!=cells[-1]:cells.append(n)
        edges=list(zip(cells,cells[1:]));usable=[(s,t) for s,t in edges if s in oldpos and t in oldpos and s in newpos and t in newpos]
        def dist(pos):return sum(abs(pos[s][0]-pos[t][0])+abs(pos[s][1]-pos[t][1]) for s,t in usable)
        rows.append(dict(end=end,slack_ns=slack,edges=len(edges),compared=len(usable),old=dist(oldpos),new=dist(newpos)))
    assert len(rows)==2000
    summary=dict(status='REFINEMENT_GLOBAL_REVIEW_READY_NOT_ROUTE_PERMISSION',
        old_metrics=metrics(base),new_metrics=current,paths=rows,
        negatives=dict(count=sum(r['slack_ns']<0 for r in rows),old=sum(r['old'] for r in rows if r['slack_ns']<0),new=sum(r['new'] for r in rows if r['slack_ns']<0)),
        all_paths=dict(old=sum(r['old'] for r in rows),new=sum(r['new'] for r in rows)),
        evidence={str(f):sha(f) for f in (Path(__file__),root/'refinement-result.json',root/'refinement-inputs.json',root/'global-sensor-review.json',report)},
        limitation='Geometry is not actual routing delay, and incomplete name matches are counted explicitly.',hardware_access=False,board_release=False)
    (root/'refinement-global-review.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('paths','evidence')}),flush=True)

if __name__=='__main__':main()
