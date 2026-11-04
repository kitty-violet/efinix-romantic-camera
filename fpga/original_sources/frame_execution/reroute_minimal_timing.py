"""One route-profile comparison on the exact C2 placement; no HDL/SDC edits."""
import json,runpy,shutil,sys
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from minimal_feedback_flow import bind
from minimal_feedback_board import audit_delta

def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')

def prepare():
    base=ROOT/'tmp/frame-execution';old=base/'minimal-feedback-place2'
    physical=base/'minimal-feedback-timing-route-place1';batch=base/'minimal-feedback-timing-route1'
    board=base/'connected-board-minimal-feedback1'
    assert not physical.exists() and not batch.exists()
    audit_delta(board)
    selected=json.loads((base/'minimal-feedback-route1/selected.json').read_text())
    assert selected['wns_ns']==-1.004 and selected['whs_ns']==-0.003
    analysis=base/'minimal-feedback-path-survey1/data_arc_analysis.json'
    a=json.loads(analysis.read_text());assert len(a['families'])==44
    assert all(f['mean_net_fraction']>.73 for f in a['families'])
    origin_review=json.loads((old/'route_review.json').read_text())
    for p,h in origin_review['required_evidence'].items():assert sha(p)==h,p
    meta=json.loads((old/'physical_inputs.json').read_text())
    for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
    # Exclude only produced route/review/run products in the NEW copy. The old
    # failed route, reports, logs, and original scripts are never modified.
    exclusions={'before_route','route_only.cmd','route_review.json','result.json','stages.json',
        'ti60f225_oob.troutingtraces','ti60f225_oob.route.out','ti60f225_oob.route.rpt',
        'ti60f225_oob.route.rpt.xml','ti60f225_oob.timing.rpt','ti60f225_oob.hold_fixing.rpt'}
    shutil.copytree(old,physical,ignore=lambda directory,names:[n for n in names if n in exclusions])
    frozen_names=['outflow/ti60f225_oob'+s for s in ('.map.v','.place','.gplace','.out.placeloc','.vdb','.interface.csv')]
    frozen_names+=['ti60f225_oob.xml','ti60f225_oob.sdc','efx_pnr_settings.ini','fixed_dsp.placeloc',
        'work_pnr/ti60f225_oob.net_proto','work_pnr/ti60f225_oob.primplace','work_pnr/ti60f225_oob.io_place']
    source={str(old/n):sha(old/n) for n in frozen_names}
    copied={str(physical/n):sha(physical/n) for n in frozen_names}
    assert list(source.values())==list(copied.values())
    for row in meta['inputs']:
        row['snapshot']=str(physical/Path(row['snapshot']).relative_to(old))
        assert sha(row['snapshot'])==row['sha256']
    save(physical/'physical_inputs.json',meta)
    # Preserve and explicitly label the inherited placement, not a new place run.
    shutil.copy2(old/'before_route/result.json',physical/'result.json')
    save(physical/'stages.json',json.loads((physical/'result.json').read_text())['stages'])
    for suffix in ('.py','.json'):
        shutil.copy2(base/(old.name+'-place-driver'+suffix),base/(physical.name+'-place-driver'+suffix))
    inheritance=dict(status='EXACT_PLACEMENT_REUSED_FOR_ROUTE_PROFILE_COMPARISON',source=str(old),
        source_physical_inputs_sha256=sha(old/'physical_inputs.json'),source_files=source,copied_files=copied,
        source_placement_result_sha256=sha(old/'before_route/result.json'),
        source_route_result=str(base/'minimal-feedback-route1/selected.json'),
        source_route_result_sha256=sha(base/'minimal-feedback-route1/selected.json'),
        generator=str(Path(__file__)),generator_sha256=sha(__file__),
        placement_executed=False,rtl_changed=False,constraints_changed=False,seed_changed=False,
        route_profile_cli_override=dict(before='CONGESTION_2',after='TIMING_1'),hardware_access=False)
    save(physical/'route_inheritance.json',inheritance)
    evidence=dict(origin_review['required_evidence'])
    for f in [physical/'physical_inputs.json',physical/'route_inheritance.json',analysis,Path(__file__),
        EF/'scripts/efx_run.py',EF/'scripts/aoe/core/options_constants.py',
        EF/'doc/pdf/efinity-timing-closure.pdf']:
        evidence[str(f)]=sha(f)
    review=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(physical),
        placement_review_sha256=sha(physical/'placement_review_inputs.json'),
        rationale='The exact legal C2 seed3 placement fits56501/60800 XLR and retains all mapped/functional proofs. Actual canonical256-endpoint survey groups into44 families, each74-96% net delay. Several one-sink data nets cost3-4ns. Keep every cell and original constraint unchanged, compare the official TIMING_1 route profile once to test routing policy before adding broad RTL pipelines. This is one controlled route comparison, not a seed sweep.',
        remaining_risks=['High-utilization router may still prioritize routability; profile need not solve congestion.',
            'Current setup-1.004ns and hold-0.003ns are failures. Both must pass new original STA; no pass is inferred.',
            'Only route profile differs. Failure requires path/net comparison before any further change.'],
        required_evidence=evidence,budget_seconds=1500,route_attempts_authorized=1,
        source_and_constraints_unchanged=True,hardware_access=False,board_release=False)
    save(physical/'route_review.json',review)
    return physical,batch

def execute(physical,batch):
    bind()
    source=HERE/'route_local_decode_board.py';code=source.read_text()
    before='--pnr_opts route max_router_iterations=100 troute_options=max_iterations=100'
    assert code.count(before)==1
    code=code.replace(before,before+' optimization_level=TIMING_1')
    before="            except subprocess.TimeoutExpired:save(root/(name+'-timeout.json'),stop_launched_tree(proc));raise"
    after="            except subprocess.TimeoutExpired:\n                save(root/(name+'-timeout.json'),stop_launched_tree(proc))\n                row.update(status='timed_out_process_tree_stopped',seconds=time.monotonic()-start)\n                save(root/'stages.json',stages);raise"
    assert code.count(before)==1;code=code.replace(before,after)
    # Check original and copied placement identity again after native route.
    before="        assert re.search(r'pnr\\s*:\\s*PASS',(root/'route.log').read_text())"
    after=before+"\n        assert 'optimization_level=TIMING_1' in (root/'route.log').read_text()\n        inheritance=json.loads((physical/'route_inheritance.json').read_text())\n        for section in ('source_files','copied_files'):\n            for path,h in inheritance[section].items():assert sha(path)==h,path"
    assert code.count(before)==1;code=code.replace(before,after)
    driver=batch.parent/(batch.name+'-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    save(driver.with_suffix('.json'),dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),
        reroute_adapter=str(Path(__file__)),reroute_adapter_sha256=sha(__file__)))
    sys.argv=[str(driver),'--physical',str(physical),'--root',str(batch),
        '--review',str(physical/'route_review.json'),'--evidence-root',str(ROOT/'tmp/frame-execution/minimal-feedback-mapped1'),
        '--serial-proof',str(ROOT/'tmp/frame-execution/minimal-feedback-mapped1-serial/new'),'--budget','1500']
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='prepare':
        p,b=prepare();print(p,b)
    elif len(sys.argv)==2 and sys.argv[1]=='run':
        execute(ROOT/'tmp/frame-execution/minimal-feedback-timing-route-place1',ROOT/'tmp/frame-execution/minimal-feedback-timing-route1')
    else:raise SystemExit('Use prepare or run')
