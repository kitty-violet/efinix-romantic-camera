"""Continue a screened placement through routing once; preserve rejected prefixes."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree

def recovered_screen_review(gate,root,budget):
    assert budget<=900 and gate['status']=='PLACEMENT_RECOVERED_FOR_BOUNDED_ROUTE'
    assert gate['current']['canonical_setup_estimate_ns'] is None
    for p,h in gate['required_evidence'].items():assert sha(p)==h
    now,old=gate['current'],gate['reference']
    assert now['placement_legal'] and all(r['used']<=r['capacity'] for r in now['resources'].values())
    assert now['weighted_hpwl_per_net']<old['weighted_hpwl_per_net']
    assert all(now['congestion_peaks'][k]<v for k,v in old['congestion_peaks'].items())
    body=(root/'outflow/ti60f225_oob.place.out').read_text()
    assert 'Post-placement cluster-level checks is successful' in body and 'Post-placement primitive-level checks is successful' in body
    return dict(status='COMPLETED_PLACEMENT_RECOVERY_ROUTE_EXPERIMENT_NOT_SIGNOFF',
        rationale='Reuse exact native-legal placement after report timeout. Resource fit and all congestion scales improve; placement timing estimate unavailable. Require actual legal route and full canonical STA; no timing requirement is waived.',
        gate_sha256=sha(root/'placement_gate.json'),host_budget_s=budget+30,board_release=False)

def ownership_screen_review(gate, budget):
    """A wirelength heuristic can trigger review; never waive a silicon limit."""
    assert budget<=900
    assert gate['status']=='PLACEMENT_SCREEN_REJECT'
    assert gate['reasons']==['weighted wirelength over reference +5%']
    now,old=gate['current'],gate['reference']
    assert now['placement_legal'] and now['canonical_setup_estimate_ns']>=0
    assert all(r['used']<=r['capacity'] for r in now['resources'].values())
    assert all(now['congestion_peaks'][k]<=v*1.10 for k,v in old['congestion_peaks'].items())
    def cnn(d):return next(r['slack_ns'] for r in d['setup_clock_relationships']
                          if r['launch']==r['capture']=='process_clk_150')
    assert cnn(now)>cnn(old)
    return dict(status='BOUNDED_OFFLINE_ROUTE_EXPERIMENT_JUSTIFIED_NOT_SIGNOFF',
        original_screen_status=gate['status'],original_reasons=gate['reasons'],
        rationale='User prioritizes legal routing and original timing over optional optimization metrics. Only the wirelength heuristic rejects this legal, capacity-fitting placement; congestion remains within the existing screening bounds and the 150MHz setup estimate improves. Route once, without changing source or constraints.',
        wirelength_ratio=now['weighted_hpwl_per_net']/old['weighted_hpwl_per_net'],
        cnn_estimate_gain_ns=cnn(now)-cnn(old),host_budget_s=budget+30,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--startup-proof',type=Path,required=True);p.add_argument('--budget',type=int,default=900)
    p.add_argument('--source-kind',choices=('local','range-local','registered-ownership'),default='local');p.add_argument('--cluster-proof',type=Path)
    p.add_argument('--max-router-iterations',type=int,choices=(70,100),default=70)
    p.add_argument('--ownership-proof',type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and 60<=a.budget<=1200
    previous=json.loads((root/'result.json').read_text())
    assert previous['status'] in ('PLACEMENT_ONLY_FINISHED_NOT_RELEASE','PLACEMENT_RECOVERED_AFTER_REPORT_TIMEOUT_NOT_RELEASE')
    gate=json.loads((root/'placement_gate.json').read_text())
    review=None
    if gate['status']!='PLACEMENT_SCREEN_PASS_ROUTED_STA_REQUIRED':
        # Keep the conservative screening rejection. One wirelength tradeoff
        # can justify a bounded experiment, never a relaxed board release.
        if a.source_kind=='registered-ownership':
            review=recovered_screen_review(gate,root,a.budget) if gate['status']=='PLACEMENT_RECOVERED_FOR_BOUNDED_ROUTE' else ownership_screen_review(gate,a.budget)
            review['gate_sha256']=sha(root/'placement_gate.json')
        else:
            assert root.name=='local-controls-place1' and a.budget<=900
            assert gate['reasons']==['weighted wirelength over reference +5%']
            now,old=gate['current'],gate['reference']
            def cnn(d):return next(r['slack_ns'] for r in d['setup_clock_relationships']
                                  if r['launch']==r['capture']=='process_clk_150')
            assert now['placement_legal'] and now['canonical_setup_estimate_ns']>=0
            assert all(r['used']<=r['capacity'] for r in now['resources'].values())
            assert cnn(now)-cnn(old)>=0.3
            assert now['congestion_peaks']['4x4']<old['congestion_peaks']['4x4']*0.9
            review=dict(status='BOUNDED_OFFLINE_ROUTE_EXPERIMENT_JUSTIFIED_NOT_SIGNOFF',
              original_screen_status=gate['status'],original_reasons=gate['reasons'],
              rationale='Wirelength +7.2%, CNN estimate +0.354ns, local congestion -11.9%; legal placement, capacity and original timing estimates hold. Obtain routed evidence once; keep all acceptance thresholds.',
              gate_sha256=sha(root/'placement_gate.json'),host_budget_s=a.budget+30,board_release=False)
        assert not (root/'placement_review.json').exists()
        (root/'placement_review.json').write_text(json.dumps(review,indent=2))
    proof=a.startup_proof.resolve();passed=json.loads((proof/'result.json').read_text())
    assert passed['status']=='GLOBAL_MAPPED_STARTUP_THREE_RESETS_PASS'
    assert sha(proof/'inputs.json')==passed['inputs_sha256']
    meta=json.loads((root/'physical_inputs.json').read_text());board=Path(meta['base'])
    for rec in meta['inputs']:assert sha(rec['snapshot'])==rec['sha256']
    net=root/'outflow/ti60f225_oob.map.v'
    assert sha(net)==meta['base_map_sha256']==json.loads((proof/'inputs.json').read_text())['original_map_sha256']
    if a.source_kind in ('range-local','registered-ownership'):
        if a.source_kind=='range-local':
            from prepare_ranged_control_board import audit_delta,check_proofs
            check_proofs()
        else:
            from registered_ownership_board import audit_delta
            assert a.ownership_proof
            op=a.ownership_proof.resolve();oe=json.loads((op/'result.json').read_text())
            assert oe['status']=='REGISTERED_OWNERSHIP_FOUR_STATE_SHORT_PASS'
            assert oe['board_inputs_sha256']==sha(board/'inputs.json')
            for row in oe['results']:
                assert sha(op/row['kind']/f'simulation_{row["seed"]}.log')==row['log_sha256']
        assert a.cluster_proof
        cp=a.cluster_proof.resolve();cr=json.loads((cp/'result.json').read_text());ci=json.loads((cp/'inputs.json').read_text())
        assert cr['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS'
        assert sha(cp/'inputs.json')==cr['inputs_sha256'] and sha(cp/'simulation.log')==cr['log_sha256']
        assert ci['frozen_board']['inputs_sha256']==sha(board/'inputs.json')
        for r in ci['inputs']:assert sha(r['snapshot'])==r['sha256']
    else:
        from prepare_local_control_board import audit_delta
    audit_delta(board)
    assert not (root/'outflow/ti60f225_oob.troutingtraces').exists()
    assert not (root/'route_only.log').exists()
    snapshot=root/'before_route';snapshot.mkdir()
    for name in ('ti60f225_oob.place.rpt','ti60f225_oob.place.out','ti60f225_oob.place.timing.rpt','ti60f225_oob.out.placeloc','ti60f225_oob.place'):
        source=root/'outflow'/name
        if source.exists():shutil.copy2(source,snapshot/name)
    shutil.copy2(root/'result.json',snapshot/'result.json');shutil.copy2(__file__,root/'route_runner.py')
    (root/'route_inputs.json').write_text(json.dumps(dict(physical_inputs_sha256=sha(root/'physical_inputs.json'),
        placement_gate_sha256=sha(root/'placement_gate.json'),mapped_startup_proof=str(proof),
        engineering_screen_review=review,
        mapped_startup_result_sha256=sha(proof/'result.json'),map_sha256=sha(net),
        pre_route_files={f.name:sha(f) for f in snapshot.iterdir()},runner_sha256=sha(__file__),
        source_kind=a.source_kind,cluster_proof=str(a.cluster_proof) if a.cluster_proof else None,
        ownership_result_sha256=sha(a.ownership_proof/'result.json') if a.ownership_proof else None,
        cluster_result_sha256=sha(a.cluster_proof/'result.json') if a.cluster_proof else None,
        max_router_iterations=a.max_router_iterations,
        host_budget_s=a.budget+30,hardware_access=False),indent=2))
    script=root/'route_only.cmd';script.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts route max_router_iterations='+str(a.max_router_iterations)+' --timeout '+str(a.budget)+'\nexit /b %errorlevel%\n')
    started=time.monotonic()
    try:
        with (root/'route_only.log').open('x') as log:
            child=subprocess.Popen(['cmd.exe','/d','/c',str(script)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            (root/'route_progress.json').write_text(json.dumps(dict(stage='route',pid=child.pid,budget_s=a.budget+30)))
            try:code=child.wait(timeout=a.budget+30)
            except subprocess.TimeoutExpired:
                (root/'route_timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        assert code==0 and re.search(r'pnr\s*:\s*PASS',(root/'route_only.log').read_text())
        body=(root/'outflow/ti60f225_oob.route.out').read_text()
        rows=re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)',body)
        assert rows and int(rows[-1][1])==0
        assert (root/'outflow/ti60f225_oob.troutingtraces').exists()
        assert sha(net)==meta['base_map_sha256']
        stages=previous['stages']+[dict(flow='pnr',phase='route_after_screened_placement',exit_code=0,
                                       seconds=time.monotonic()-started,log_sha256=sha(root/'route_only.log'))]
        result=dict(status='PHYSICAL_CAPACITY_PROBE_FINISHED',stages=stages,route_iterations=int(rows[-1][0]),
                    strict_sta_pass=False,board_download_allowed=False,source_unchanged=True)
        (root/'stages.json').write_text(json.dumps(stages,indent=2));(root/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'route_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),
                    seconds=time.monotonic()-started,board_release=False),indent=2));raise
if __name__=='__main__':main()
