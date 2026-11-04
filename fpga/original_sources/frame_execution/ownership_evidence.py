"""Evidence binding for the complete 150MHz registered-ownership system."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from registered_ownership_board import audit_delta

PROOFS={
    'short':('registered-ownership-short1','REGISTERED_OWNERSHIP_FOUR_STATE_SHORT_PASS'),
    'cluster':('registered-ownership-cluster4','FIXED_CLUSTER_REAL_RESIDENT_PASS'),
    'a_cluster':('registered-ownership-a-cluster1','REGISTERED_ACCEPTED_A_CLUSTER_PASS'),
    'a_smoke':('registered-ownership-a-smoke1','REGISTERED_ACCEPTED_A_SMOKE_PASS'),
    'a_full':('registered-ownership-a-full1','REGISTERED_ACCEPTED_A_FULL_PASS'),
}

def bind_proofs(board,evidence):
    delta=audit_delta(board);results={}
    def bind(p):
        evidence[str(p)]=sha(p);return json.loads(p.read_text())
    for key,(directory,status) in PROOFS.items():
        root=ROOT/'tmp/frame-execution'/directory
        r=bind(root/'result.json');assert r['status']==status
        results[key]=r
        if key=='short':
            assert r['board_inputs_sha256']==sha(board/'inputs.json')
            for kind in ('rq','router'):
                m=bind(root/kind/'inputs.json')
                assert m['board_inputs_sha256']==sha(board/'inputs.json')
                for item in m['inputs']:
                    f=Path(item['snapshot']);assert sha(f)==item['sha256'];evidence[str(f)]=sha(f)
                    candidate=board/'assembly'/f.name
                    if candidate.exists():assert sha(candidate)==sha(f)
            for row in r['results']:
                f=root/row['kind']/f'simulation_{row["seed"]}.log'
                assert sha(f)==row['log_sha256'];evidence[str(f)]=sha(f)
            continue
        m=bind(root/'inputs.json');assert r['inputs_sha256']==sha(root/'inputs.json')
        assert r['log_sha256']==sha(root/'simulation.log')
        evidence[str(root/'simulation.log')]=r['log_sha256']
        if key=='cluster':assert m['frozen_board']['inputs_sha256']==sha(board/'inputs.json')
        else:assert m['board_inputs_sha256']==sha(board/'inputs.json')
        for row in m['inputs']:
            f=Path(row['snapshot']);assert sha(f)==row['sha256'];evidence[str(f)]=sha(f)
        if key!='cluster':
            for f,h in m['vector_hashes'].items():assert sha(f)==h;evidence[f]=h
            assert sha(m['binary'])==m['binary_sha256'];evidence[m['binary']]=m['binary_sha256']
    assert results['a_full']['cycles']>0 and not results['a_full']['whole_board']
    for filename in ('registered_ownership_board.py','validate_registered_ownership.py',
                     'validate_ownership_accepted_a.py','ownership_evidence.py'):
        evidence[str(HERE/filename)]=sha(HERE/filename)
    return dict(delta=delta,results=results,full_board_simulated=False,board_tested=False)
