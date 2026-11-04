"""Bounded existing implementation flows, explicitly rebound to accepted-A tail."""
import json,sys
from pathlib import Path
from run_units import ROOT,sha
from accepted_tail_board import audit_delta,BASE

def short_proofs(board,folder):
 r=json.loads((folder/'result.json').read_text());i=json.loads((folder/'inputs.json').read_text())
 assert r['status']=='ACCEPTED_A_TAIL_FOUR_STATE_NUMERIC_GUARD_PASS'
 assert r['inputs_sha256']==sha(folder/'inputs.json') and i['board_inputs_sha256']==sha(board/'inputs.json')
 for x in i['inputs']:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
 for n,h in i['testbenches'].items():assert sha(folder/(n+'.sv'))==h
 for x in r['runs']:assert sha(folder/(x['top']+'.log'))==x['log_sha256']
 # RQ feedback/control sources are byte-identical to the proven parent;
 # this is explicit evidence reuse, not a newly executed regression claim.
 import cached_head_flow as old
 parent_proof=old_parent_proofs(BASE,ROOT/'tmp/frame-execution/minimal-feedback-shorts1')
 for name in ('fixed_rq32_mixed.sv','fixed_partial32.sv','row_pair_aux_router.sv'):
  assert sha(board/'assembly'/name)==sha(BASE/'assembly'/name)
 return dict(tail_short_path=str(folder),tail_result_sha256=sha(folder/'result.json'),
             inherited_unchanged_rq=parent_proof,parent_inputs_sha256=sha(BASE/'inputs.json'))

def main():
 global old_parent_proofs
 mode=sys.argv[1]
 if mode=='selector-audit':
  import audit_feedback_selection_map as check
  check.audit_delta=audit_delta
  board=Path(sys.argv[2]).resolve();out=board/'feedback_selection_map_audit.json'
  assert not out.exists();out.write_text(json.dumps(check.audit(board),indent=2));print('FEEDBACK_MAP_PASS');return
 if mode=='accepted-frame':
  sys.argv.pop(1)
  import validate_ownership_accepted_a as check
  check.audit_delta=audit_delta;check.main();return
 import minimal_feedback_flow as minimal
 f,old=minimal.bind();f.audit_delta=audit_delta
 old.audit_delta=audit_delta;old.BASE=BASE
 import local_control_decode
 local_control_decode.audit_delta=audit_delta
 old_parent_proofs=old.proofs;old.proofs=short_proofs
 f.main()
if __name__=='__main__':main()
