"""Run established board gates with the independently audited no-COMB4 clone."""
import json
from run_units import HERE,sha
from split_logic_outputs_board import BASE,audit_delta
import startup_cause_flow as flow

original_proof=flow.proof
def proof(board,folder):
 record=original_proof(board,folder)
 wrapper=folder/'split_wrapper_proof.json'
 data=json.loads(wrapper.read_text())
 assert data['status']=='SPLIT_OUTPUTS_SAME_FIFTEEN_SHORT_CASES_PASS'
 assert data['original_result_sha256']==sha(folder/'result.json')
 assert data['wrapper_sha256']==sha(HERE/'validate_split_logic_outputs.py')
 assert data['implementation_sha256']==sha(HERE/'split_logic_outputs_board.py')
 record['split_wrapper_proof_sha256']=sha(wrapper)
 return record

if __name__=='__main__':
 flow.audit_delta=audit_delta;flow.BASE=BASE;flow.proof=proof
 flow.main()
