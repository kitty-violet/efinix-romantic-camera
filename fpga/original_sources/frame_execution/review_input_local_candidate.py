"""Collect global placement comparisons before deciding whether to route.

Read-only with respect to implementation. This produces evidence, not a route
authorization or a timing pass. Runs only after placement has fully exited.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_units import ROOT, HERE, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--physical',type=Path,required=True)
    a=p.parse_args(); physical=a.physical.resolve()
    assert physical.is_relative_to(ROOT/'tmp/frame-execution')
    result=json.loads((physical/'result.json').read_text())
    assert result['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    assert all(s['exit_code']==0 for s in result['stages'])
    assert (physical/'placement_review_inputs.json').is_file(), 'Wait for placement wrapper review completion'
    base=ROOT/'tmp/frame-execution'
    board=base/'connected-board-video-service-probe60-input-isp-explicit'
    old=base/'video-service-place48-input-isp'
    checks=[
        ('controls','survey_video_service_controls.py',[
            '--board',board,'--physical',physical,'--output',physical/'control-network-survey.json']),
        ('paths','compare_video_input_placement_paths.py',[
            '--report',base/'video-input-canonical1/outflow/canonical_setup.rpt',
            '--failed',old,'--candidate',physical,'--out',physical/'failed-path-geometry.json']),
        ('input','review_input_isp_geometry.py',[
            '--physical',physical,'--reference',old,'--output',physical/'input-region-comparison.json'])]
    evidence={}; stages=[]
    for name,script,args in checks:
        log=physical/('review-'+name+'.log')
        if name=='controls' and (physical/'control-network-survey.json').exists():
            previous=json.loads((physical/'control-network-survey.json').read_text())
            assert previous['checker_sha256']==sha(HERE/script)
            assert previous['placement_sha256']==sha(physical/'outflow/ti60f225_oob.out.placeloc')
            assert previous['map_sha256']==sha(physical/'outflow/ti60f225_oob.map.v')
            stages.append(dict(name=name,reused_verified=True,log_sha256=sha(log)))
            evidence[str(HERE/script)]=sha(HERE/script)
            continue
        if log.exists():
            log=physical/('review-'+name+'-canonical.log')
        with log.open('x',encoding='utf8') as stream:
            subprocess.run([sys.executable,str(HERE/script),*map(str,args)],
                           stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=180)
        stages.append(dict(name=name,log_sha256=sha(log)))
        evidence[str(HERE/script)]=sha(HERE/script)
    metrics={}
    for label,folder in [('passing67',base/'video-service-place47-color-exposure'),
                         ('failed69C2',old),('failed69C1',base/'video-service-place49-input-c1'),
                         ('current',physical)]:
        path=folder/'placement_review_inputs.json'
        current=json.loads(path.read_text())['current']
        metrics[label]={k:current[k] for k in ('resources','packing','congestion_peaks',
            'weighted_hpwl_per_net','canonical_setup_estimate_ns','placement_legal')}
        evidence[str(path)]=sha(path)
    paths=json.loads((physical/'failed-path-geometry.json').read_text())
    controls=json.loads((physical/'control-network-survey.json').read_text())
    assert len(controls['rows'])==109 and not any(r['unplaced_sink_cells'] for r in controls['rows'])
    assert paths['same_map'] and len(paths['rows'])==509
    comparisons={}
    for folder in (old,base/'video-service-place49-input-c1'):
        path=folder/'control-network-survey.json'
        prior={r['driver']:r for r in json.loads(path.read_text())['rows']}
        changes=[]
        for r in controls['rows']:
            before=prior[r['driver']]
            assert r['fanout']==before['fanout'] and r['pins']==before['pins']
            size=lambda b:b[2]-b[0]+b[3]-b[1]
            changes.append(dict(driver=r['driver'],fanout=r['fanout'],
                before_bbox=before['bbox'],after_bbox=r['bbox'],
                envelope_delta=size(r['bbox'])-size(before['bbox'])))
        changes.sort(key=lambda r:r['envelope_delta'],reverse=True)
        comparisons[folder.name]=changes
        evidence[str(path)]=sha(path)
    for name in ('control-network-survey.json','failed-path-geometry.json',
                 'input-region-comparison.json','input_isp_ram_placement.json',
                 'input_isp_ram_actual_locations.json','cnn_ram_actual_locations.json'):
        evidence[str(physical/name)]=sha(physical/name)
    output=dict(status='PLACEMENT_EVIDENCE_COLLECTED_REQUIRES_REVIEW',metrics=metrics,
        failed_path_distance=dict(old=sum(r['failed_distance'] for r in paths['rows']),
                                  new=sum(r['candidate_distance'] for r in paths['rows'])),
        control_envelope_changes=comparisons,stages=stages,evidence=evidence,
        generator_sha256=sha(__file__),hardware_access=False,board_release=False)
    target=physical/'global-input-review.json'
    with target.open('x',encoding='utf8') as f:json.dump(output,f,indent=2)
    print(json.dumps(dict(status=output['status'],metrics=metrics,
                          failed_path_distance=output['failed_path_distance'])))


if __name__=='__main__':main()
