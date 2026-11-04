"""Release evidence for one frozen implementation; retain every seed outcome.

2026-10-10 user accepted selecting the passing seed instead of all-seed PASS.
This changes selection policy only, not the actual STA/capacity/CDC requirements.
"""
import json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_canonical_multiseed import audit
from registered_ownership_board import audit_delta

def selection(board,batch,canonical):
    board=board.resolve();batch=batch.resolve();canonical=canonical.resolve()
    audit_delta(board)
    evidence={}
    def bind(p):
        evidence[str(p)]=sha(p)
        return json.loads(p.read_text())
    mapped=bind(board/'result.json');bm=bind(batch/'inputs.json')
    assert bm['board_inputs_sha256']==mapped['inputs_sha256']==sha(board/'inputs.json')
    assert Path(bm['board'])==board and bm['hardware_access'] is False
    stages=bind(batch/'stages.json')
    assert stages and all(r['status']=='finished' for r in stages),'Batch still running'
    selected=audit(canonical)
    assert selected['current_timing_pass'],'Selected route fails original timing/capacity/checks'
    assert selected['identity']['map']==mapped['map_sha256']
    assert canonical==batch/f'canonical-seed{selected["seed"]}'
    trials=[]
    for seed in sorted({r['seed'] for r in stages if r['seed']!=0}):
        cf=batch/f'canonical-seed{seed}'
        if (cf/'result.json').exists():
            row=audit(cf)
            assert row['identity']==selected['identity'],'Different inputs beyond seed'
            for name in ('result.json','inputs.json'):bind(cf/name)
            trials.append({k:row[k] for k in ('seed','physical','canonical','current_timing_pass','wns_ns','whs_ns')})
        else:
            route=batch/f'place-seed{seed}'
            pm=bind(route/'physical_inputs.json')
            assert pm['base_map_sha256']==mapped['map_sha256'] and pm['seed_change']['seed']==seed
            for item in pm['inputs']:assert sha(item['snapshot'])==item['sha256']
            failure=bind(route/'route_failure.json')
            log=route/'outflow/ti60f225_oob.route.out';body=log.read_text()
            evidence[str(log)]=sha(log)
            assert 'Routing FAILED; could not resolve congestion.' in body
            overlap=int(re.findall(r'Iter\s+\d+:\s*overlap=(\d+)',body)[-1]);assert overlap>0
            assert any(r['seed']==seed and r['stage']=='route' and r['exit_code']!=0 for r in stages)
            trials.append(dict(seed=seed,physical=str(route),current_timing_pass=False,
                status='ROUTING_FAILED_NOT_SELECTED',final_overlap=overlap,wns_ns=None,whs_ns=None))
    assert len(trials)>=2 and any(r['seed']==selected['seed'] and r['current_timing_pass'] for r in trials)
    if (batch/'failure.json').exists():bind(batch/'failure.json')
    for file in (Path(__file__),HERE/'audit_canonical_multiseed.py',ROOT/'AGENTS.md'):
        evidence[str(file)]=sha(file)
    return dict(status='SELECTED_SEED_TIMING_PASS_OTHER_RESULTS_RECORDED',
        policy='2026-10-10 selected frozen route; other seeds are diagnostic evidence, not unanimous acceptance',
        board=str(board),batch=str(batch),selected=selected,trials=trials,
        unstarted_seeds=[6] if not any(r['seed']==6 for r in stages) else [],
        historical_receipts_unchanged=True,required_evidence=evidence,board_release=False)
