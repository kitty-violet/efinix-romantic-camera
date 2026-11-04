"""Full-function 150MHz candidate: bounded first seed, then two independent jobs.

No programmer access. Stop before more seeds if original-constraint STA fails.
"""
import argparse,concurrent.futures,json,subprocess,sys,threading,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from registered_ownership_board import audit_delta
from windows_process_tree import stop_launched_tree
from audit_canonical_multiseed import audit
from route_screened_board import ownership_screen_review,recovered_screen_review

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--board',type=Path,required=True)
    p.add_argument('--cluster',type=Path,required=True);p.add_argument('--short',type=Path,required=True);p.add_argument('--first-only',action='store_true')
    p.add_argument('--resume-placement',type=Path)
    p.add_argument('--max-router-iterations',type=int,choices=(70,100),default=70)
    p.add_argument('--first-placement',type=Path)
    p.add_argument('--optimization',choices=('CONGESTION_1','CONGESTION_2','CONGESTION_3'),default='CONGESTION_1')
    a=p.parse_args();root=a.root.resolve();board=a.board.resolve();cluster=a.cluster.resolve();short=a.short.resolve()
    audit_delta(board);mapping=json.loads((board/'result.json').read_text())
    assert mapping['inputs_sha256']==sha(board/'inputs.json') and mapping['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    cr=json.loads((cluster/'result.json').read_text());ci=json.loads((cluster/'inputs.json').read_text())
    assert cr['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS' and cr['inputs_sha256']==sha(cluster/'inputs.json')
    assert ci['frozen_board']['inputs_sha256']==sha(board/'inputs.json') and cr['log_sha256']==sha(cluster/'simulation.log')
    sr=json.loads((short/'result.json').read_text())
    assert sr['status']=='REGISTERED_OWNERSHIP_FOUR_STATE_SHORT_PASS' and sr['board_inputs_sha256']==sha(board/'inputs.json')
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    (root/'runner.py').write_bytes(Path(__file__).read_bytes());stages=[];lock=threading.Lock();sta_lock=threading.Lock()
    (root/'inputs.json').write_text(json.dumps(dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),
        cluster=str(cluster),cluster_result_sha256=sha(cluster/'result.json'),short=str(short),short_result_sha256=sha(short/'result.json'),
        runner_sha256=sha(__file__),max_router_iterations=a.max_router_iterations,optimization=a.optimization,hardware_access=False),indent=2))
    assert not a.first_placement or a.resume_placement
    def run(seed,name,script,args,budget):
        start=time.monotonic();logfile=root/f'{seed}_{name}.log'
        with logfile.open('x') as log:
            child=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(seed=seed,stage=name,pid=child.pid,status='running',budget_s=budget)
            with lock:stages.append(row);(root/'stages.json').write_text(json.dumps(stages,indent=2))
            try:code=child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                (root/f'{seed}_{name}_cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        with lock:
            row.update(status='finished',exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(logfile))
            (root/'stages.json').write_text(json.dumps(stages,indent=2))
        assert code==0,(seed,name,logfile)
    startup=root/'startup'
    if a.resume_placement:
        prefix=a.resume_placement.resolve();assert prefix.is_relative_to(ROOT/'tmp/frame-execution')
        prior=json.loads((prefix/'inputs.json').read_text())
        assert prior['board_inputs_sha256']==sha(board/'inputs.json')
        assert prior['cluster_result_sha256']==sha(cluster/'result.json') and prior['short_result_sha256']==sha(short/'result.json')
        startup=prefix/'startup'
        assert json.loads((startup/'result.json').read_text())['status']=='GLOBAL_MAPPED_STARTUP_THREE_RESETS_PASS'
        (root/'reused_prefix.json').write_text(json.dumps(dict(prefix=str(prefix),inputs_sha256=sha(prefix/'inputs.json'),
            failure_sha256=sha(prefix/'failure.json'),startup_result_sha256=sha(startup/'result.json')),indent=2))
    def place(seed):
        physical=root/f'place-seed{seed}'
        placement_budget=600 if a.optimization=='CONGESTION_1' else 1000
        run(seed,'place','probe_global_place.py',['--base',board,'--root',physical,'--pnr-budget',placement_budget,'--logic-limit',60800,
            '--optimization',a.optimization,'--fixed-dsp-layout','--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
            '--seed',seed,'--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],placement_budget+420)
        gate=json.loads((physical/'placement_gate.json').read_text())
        if gate['status']!='PLACEMENT_SCREEN_PASS_ROUTED_STA_REQUIRED':ownership_screen_review(gate,900)
        return physical
    def finish(seed,physical):
        canonical=root/f'canonical-seed{seed}'
        run(seed,'route','route_screened_board.py',['--root',physical,'--startup-proof',startup,
            '--source-kind','registered-ownership','--cluster-proof',cluster,'--ownership-proof',short,'--budget',900,
            '--max-router-iterations',a.max_router_iterations],990)
        with sta_lock:run(seed,'sta','canonical_snapshot_sta.py',['--base',physical,'--root',canonical],270)
        receipt=audit(canonical)
        assert receipt['current_timing_pass'],(seed,receipt['wns_ns'],receipt['whs_ns'])
        return receipt
    def one(seed):return finish(seed,place(seed))
    try:
        if a.resume_placement:
            physical=a.first_placement.resolve() if a.first_placement else prefix/'place-seed1'
            pm=json.loads((physical/'physical_inputs.json').read_text())
            assert pm['base_map_sha256']==mapping['map_sha256'] and pm['optimization_change']['value']==a.optimization
            assert pm['seed_change']['seed']==1
            assert json.loads((physical/'result.json').read_text())['status'] in ('PLACEMENT_ONLY_FINISHED_NOT_RELEASE','PLACEMENT_RECOVERED_AFTER_REPORT_TIMEOUT_NOT_RELEASE')
            gate=json.loads((physical/'placement_gate.json').read_text())
            if gate['status']=='PLACEMENT_RECOVERED_FOR_BOUNDED_ROUTE':recovered_screen_review(gate,physical,900)
            elif gate['status']!='PLACEMENT_SCREEN_PASS_ROUTED_STA_REQUIRED':ownership_screen_review(gate,900)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                start=pool.submit(run,0,'startup','validate_global_mapped_startup.py',['--board',board,'--root',startup],180)
                first_place=pool.submit(place,1)
                start.result();physical=first_place.result()
        run(0,'implementation','audit_local_control_implementation.py',['--board',board,'--physical',physical,'--out',root/'implementation.json',
            '--source-kind','registered-ownership','--cluster-proof',cluster,'--ownership-proof',short],90)
        first=finish(1,physical)
        (root/'first_seed.json').write_text(json.dumps(first,indent=2));print('FIRST_SEED_CANONICAL_PASS',flush=True)
        if a.first_only:
            (root/'result.json').write_text(json.dumps(dict(status='FIRST_SEED_PASS_NOT_BOARD_RELEASE',first=first),indent=2));return
        rows=[first];failures=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(one,seed):seed for seed in (3,5)}
            for future in concurrent.futures.as_completed(tasks):
                try:rows.append(future.result())
                except Exception as exc:failures.append(dict(seed=tasks[future],error=repr(exc),traceback=traceback.format_exc()))
        if failures:
            (root/'failure.json').write_text(json.dumps(failures,indent=2));raise RuntimeError('Seed pair failed; seed6 not started')
        rows.append(one(6));assert all(row['identity']==first['identity'] for row in rows)
        for row in rows:
            route=json.loads((Path(row['physical'])/'route_inputs.json').read_text())
            assert route['max_router_iterations']==a.max_router_iterations
        output=root/'four_seed_audit.json'
        run(0,'multiseed','audit_canonical_multiseed.py',['--roots',*[row['canonical'] for row in sorted(rows,key=lambda r:r['seed'])],'--output',output],90)
        result=dict(status='REGISTERED_OWNERSHIP_FOUR_SEED_PASS_PENDING_CDC',evidence=str(output),evidence_sha256=sha(output),hardware_access=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        if not (root/'failure.json').exists():(root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2))
        raise
if __name__=='__main__':main()
