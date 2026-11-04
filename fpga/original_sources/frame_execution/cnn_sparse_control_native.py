"""Small exact control-driver replication from two completed routed surveys."""
import argparse,importlib.util,inspect,json,re,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
import native_local_drivers as replica
from cnn_observer_trim_board import BOARD as PARENT,audit_delta
from control_command_repair import once

OUT=ROOT/'tmp/frame-execution/cnn-sparse-control-native1'
TARGETS={
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/writer/pending[4]',
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/source/running',
 'camera_payload_owner_020',
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/n18014',
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/n17984',
 'video_processing_pipeline/pipeline/cnn/fetch/n16035',
 'video_processing_pipeline/pipeline/cnn/fetch/n16070',
 'video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/arithmetic/lane_depthwise[10]',
 'n64026_2',
 'n67511_2','n67489_2','n67392_2','n63612_2',
}

def partition(loads,positions):
    # The measured 3.696ns seven-load first hop is isolated explicitly.
    # Other groups are coarser than the earlier broad replication attempt.
    critical=[load for load in loads if load[0] in ('LUT__133119','FACTOR_ERROR_019')]
    if critical and 2<=len(loads)<=8:
        assert len(critical)==1
        return [[load for load in loads if load not in critical],critical]
    if len(loads)<2:return [loads]
    extents=[max(positions[n][d] for n,p in loads)-min(positions[n][d] for n,p in loads) for d in (0,1)]
    if len(loads)<=64 and max(extents)<=128:return [loads]
    axis=0 if extents[0]>=extents[1] else 1
    ordered=sorted(loads,key=lambda x:(positions[x[0]][axis],positions[x[0]][1-axis],x))
    middle=len(ordered)//2
    return partition(ordered[:middle],positions)+partition(ordered[middle:],positions)

def generate(source,placement,paths):
    code=inspect.getsource(replica.transform)
    code=once(code,"if net['fanout']>=12 and net['delay_ns']>=1.5:","if net['net'] in TARGETS:")
    code=once(code,"name=f'LOCAL_DRIVER_{idx:03d}';wire=f'local_driver_net_{idx:03d}'", "name=f'SPARSE_DRIVER_{idx:03d}';wire=f'sparse_driver_net_{idx:03d}'")
    code=once(code,'0<len(new)<=320','0<len(new)<=40')
    code=once(code,'        if len(groups)<2:continue',
              '        if len(groups)<2:\n            assert len(loads[net])>=2\n            ordered=sorted(loads[net],key=lambda v:(pos[v[0]][0],pos[v[0]][1],v))\n            mid=len(ordered)//2\n            groups=[ordered[:mid],ordered[mid:]]')
    scope=dict(replica.__dict__,TARGETS=TARGETS,partition=partition)
    exec(compile(code,__file__+':exact-control-clones','exec'),scope)
    changed,rows,proof=scope['transform'](source,placement,dict(paths=paths))
    assert {r['net'] for r in rows}==TARGETS
    assert 3<=proof['added_kinds'].get('EFX_FF',0)<=5
    assert set(proof['added_kinds'])=={'EFX_FF','EFX_LUT4'} and proof['clones']<=40
    ff_targets={'video_processing_pipeline/pipeline/cnn/core/dut/engine/writer/pending[4]',
                'video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/source/running','camera_payload_owner_020'}
    assert all(r['kind']=='EFX_LUT4' or r['net'] in ff_targets for r in rows)
    # Existing graph proof checks every untouched cell/port/parameter and all
    # cloned clock/reset/CE/initial-state parameters, with zero added latency.
    first_lut=next(r['clones'][0] for r in rows if r['kind']=='EFX_LUT4')
    ff=next(r['clones'][0] for r in rows if r['kind']=='EFX_FF')
    mask=re.search(r'defparam '+first_lut['name']+r"\.LUTMASK\s*=\s*(\d+)'h([0-9a-f]+);",changed);assert mask
    badmask=changed[:mask.start(2)]+format(int(mask[2],16)^1,'x')+changed[mask.end(2):]
    badwire=changed.replace('.O('+first_lut['net']+')',".O(1'b0)",1)
    badreset,n=re.subn(r'(defparam '+ff['name']+r"\.SR_SYNC\s*=\s*)1'h1;",r"\g<1>1'h0;",changed,count=1);assert n==1
    for label,bad in [('mask',badmask),('wire',badwire),('reset_mode',badreset)]:
        assert bad!=changed
        try:replica.audit(source,bad,rows)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Corruption accepted: '+label)
    proof['negative_checks']=['mask','wire','reset_mode']
    return changed,rows,proof

def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.check_only:
        compile(inspect.getsource(generate),__file__,'exec');print('SPARSE_CONTROL_GENERATOR_COMPILES');return
    assert not OUT.exists()
    audit_delta(PARENT)
    canonical=ROOT/'tmp/frame-execution/cnn-observer-goal-route1/canonical-seed3/result.json'
    completed=json.loads(canonical.read_text());assert completed['status']=='CANONICAL_ROUTED_TIMING_FAIL'
    assert completed['rows']['setup']['worst_ns']==-.894
    survey_root=ROOT/'tmp/frame-execution/cnn-observer-goal-survey1'
    sr=json.loads((survey_root/'result.json').read_text())
    assert sr['status']=='CANONICAL_GEOMETRY_ENDPOINT_SURVEY_COMPLETE_NOT_RELEASE' and sr['negative_endpoint_coverage_complete']
    assert sr['report_sha256']==sha(sr['report'])
    parser=ROOT/'tmp/board-test/analyze-camera-compact-routed-paths.py'
    spec=importlib.util.spec_from_file_location('route_parser',parser);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    old_report=ROOT/'tmp/frame-execution/cnn-observer-trim-survey1/outflow/isolated_setup.rpt'
    old_paths=m.parse(old_report);new_paths=m.parse(Path(sr['report']))
    assert len(old_paths)==len(new_paths)==1000 and sum(p['slack_ns']<0 for p in old_paths)==64
    paths=[p for p in old_paths+new_paths if p['slack_ns']<0]
    placement=ROOT/'tmp/frame-execution/cnn-observer-trim-place1/outflow/ti60f225_oob.out.placeloc'
    assert sha(placement)==sha(ROOT/'tmp/frame-execution/cnn-observer-goal-bound1/outflow/ti60f225_oob.out.placeloc')
    source=PARENT/'outflow/ti60f225_oob.map.v'
    OUT.mkdir()
    try:
        changed,rows,proof=generate(source.read_text(),placement.read_text(),paths)
        (OUT/'native_source.v').write_text(changed,encoding='utf8')
        (OUT/'replications.json').write_text(json.dumps(rows,indent=2))
        proof.update(status='SPARSE_CONTROL_EXACT_CLONE_GRAPH_PASS',parent=str(PARENT),
                     source_sha256=sha(OUT/'native_source.v'),replications_sha256=sha(OUT/'replications.json'),
                     target_nets=sorted(TARGETS),original_negative_endpoints=64,goal_negative_endpoints=sr['negative_paths'],
                     no_arithmetic_or_clock_reset_change=True,ordinary_placement_anchors_added=0,
                     inputs={str(f):sha(f) for f in (Path(__file__),HERE/'native_local_drivers.py',source,placement,parser,old_report,
                              Path(sr['report']),survey_root/'result.json',survey_root/'inputs.json',canonical)},
                     physical_pass=False,board_release=False)
        (OUT/'result.json').write_text(json.dumps(proof,indent=2))
        print(json.dumps({k:v for k,v in proof.items() if k!='inputs'}),flush=True)
    except Exception as exc:
        (OUT/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
