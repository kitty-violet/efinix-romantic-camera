"""Reuse real reset/pin/CDC-cell checks with the explicit CNN-local delta."""
import argparse,json
from pathlib import Path
import audit_control_state_implementation as original
from prepare_local_control_board import audit_delta,audit_branches,PROOFS,SMALL
from run_units import sha

p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
p.add_argument('--physical',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--source-kind',choices=('local','range-local','registered-ownership'),default='local');p.add_argument('--cluster-proof',type=Path)
p.add_argument('--ownership-proof',type=Path);a=p.parse_args()
board=a.board.resolve();physical=a.physical.resolve();assert not a.out.exists()
# All original structural/electrical checks remain; only the permitted source
# change audit is replaced, so a second changed RTL file is rejected.
extra_proofs=[]
if a.source_kind in ('range-local','registered-ownership'):
    if a.source_kind=='range-local':
        from prepare_ranged_control_board import audit_delta as range_delta,check_proofs
        check_proofs();original.audit_delta=range_delta
    else:
        from registered_ownership_board import audit_delta as ownership_delta
        original.audit_delta=ownership_delta
        assert a.ownership_proof
        op=a.ownership_proof.resolve();oe=json.loads((op/'result.json').read_text())
        assert oe['status']=='REGISTERED_OWNERSHIP_FOUR_STATE_SHORT_PASS' and oe['board_inputs_sha256']==sha(board/'inputs.json')
        for row in oe['results']:assert sha(op/row['kind']/f'simulation_{row["seed"]}.log')==row['log_sha256']
        extra_proofs.append(op)
    assert a.cluster_proof
    cp=a.cluster_proof.resolve();cr=json.loads((cp/'result.json').read_text());ci=json.loads((cp/'inputs.json').read_text())
    assert cr['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS' and sha(cp/'inputs.json')==cr['inputs_sha256']
    assert sha(cp/'simulation.log')==cr['log_sha256'] and ci['frozen_board']['inputs_sha256']==sha(board/'inputs.json')
    extra_proofs.append(cp)
else:original.audit_delta=audit_delta
result=original.audit(board,physical)
assert result['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS'
result['local_control_branches']=audit_branches(board/'outflow/ti60f225_oob.map.v')
for proof in (PROOFS,SMALL,*extra_proofs):
    result['required_evidence'][str(proof/'result.json')]=sha(proof/'result.json')
result['required_evidence'][str(Path(__file__))]=sha(__file__)
result['scope']='Original controller/reset/pin/observation checks, plus CNN-local zero-cycle distribution and explicitly selected source delta; not routed CDC signoff or board release.'
result['source_kind']=a.source_kind
a.out.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],reset_ff=result['reset_ff'],reset_truth_patterns=result['reset_cone_truth_patterns'],
                     local_drivers=len(result['local_control_branches']['drivers']),board_release=False)))
