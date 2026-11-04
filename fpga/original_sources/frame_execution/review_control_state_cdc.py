"""Review the explicit control-state delta and all inherited observation/CDC checks."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_control_state_board import audit_delta,PROOFS
from prepare_snapshot_board import PROOF
from snapshot_clock_cdc import inventory
from audit_snapshot_jtag_paths import audit as audit_jtag

def review(board,batch,gray,output):
 assert not output.exists();evidence={}
 def bind(p):
  evidence[str(p)]=sha(p);return json.loads(p.read_text())
 parent=bind(ROOT/'output/deployment/board-local-reset-20261009/manifest.json')
 assert parent['status']=='OFFLINE_READY_AWAIT_USER_SUPERVISION'
 for p,h in parent['required_evidence'].items():assert sha(p)==h; evidence[p]=h
 delta=audit_delta(board)
 for proof in PROOFS:
  result=bind(proof/'result.json');inputs=bind(proof/'inputs.json')
  assert result['status'] in ('CONTROL_STATE_LOCKSTEP_PASS','ACTUAL_MAPPED_CONTROL_LOCKSTEP_PASS')
  assert result['inputs_sha256']==sha(proof/'inputs.json')
  for n,h in inputs['files'].items():assert sha(proof/n)==h;evidence[str(proof/n)]=h
  evidence[str(proof/'simulation.log')]=sha(proof/'simulation.log')
 assert bind(PROOFS[1]/'result.json')['full_a_load']
 assert (board/'assembly/EfxSapphireSoc.v').read_text()==(PROOFS[1]/'EfxSapphireSoc.v').read_text()
 impl=bind(batch/'snapshot_implementation.json')
 assert impl['status']=='CONTROL_STATE_MAPPED_RESET_AND_PERIPHERY_PASS' and impl['board']==str(board)
 assert impl['snapshot_sync']['synchronizer_ff']==20
 for p,h in impl['required_evidence'].items():assert sha(p)==h; evidence[p]=h
 short=bind(PROOF/'result.json');sm=bind(PROOF/'inputs.json')
 assert short['status']=='INTEGRATED_REAL_CONTROL_SNAPSHOT_SHORT_PASS' and short['host_decoded_scans']>=37
 assert short['evidence_sha256']==sha(PROOF/'inputs.json')
 for f,h in sm['files'].items():assert sha(PROOF/f)==h;evidence[str(PROOF/f)]=h
 for p,h in sm['original'].items():assert sha(p)==h;evidence[p]=h
 from read_control_snapshot import decode
 import re
 log=(PROOF/'simulation.log').read_text()
 decoded=[decode(int(x,16)) for x in re.findall(r'OBSRAW ([0-9a-fA-F]+)',log)]
 assert decoded==bind(PROOF/'host_decoded.json')
 mappedroot=ROOT/'tmp/frame-execution/snapshot-mapped-short1'
 mapped=bind(mappedroot/'result.json');mapped_inputs=bind(mappedroot/'inputs.json')
 assert mapped['status']=='MAPPED_OBSERVER_RTL_EQUIVALENCE_PASS' and mapped['compared']>=43602
 assert mapped['inputs_sha256']==sha(mappedroot/'inputs.json') and mapped['simulation_sha256']==sha(mappedroot/'simulation.log')
 evidence[str(mappedroot/'simulation.log')]=sha(mappedroot/'simulation.log')
 for p,h in mapped_inputs['original'].items():assert sha(p)==h;evidence[p]=h
 for f,h in mapped_inputs['files'].items():assert sha(mappedroot/f)==h;evidence[str(mappedroot/f)]=h
 evidence[str(PROOF/'simulation.log')]=sha(PROOF/'simulation.log')
 g=bind(gray/'result.json');gm=bind(gray/'inputs.json');gi=bind(gray/'clock_crossing_inventory.json')
 assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
 assert g['inputs_sha256']==sha(gray/'inputs.json') and g['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
 assert gi==json.loads(json.dumps(inventory(gray)))
 assert g['buses']==10 and g['endpoints']==70 and g['max_skew_ns']<g['limit_ns']==2.5
 j=bind(gray/'jtag_cdc_audit.json');assert j==audit_jtag(gray)
 assert g['jtag_audit_sha256']==sha(gray/'jtag_cdc_audit.json') and j['unknown_new_endpoints']==0
 for p,h in gm['original'].items():assert sha(p)==h;evidence[p]=h
 for row in gi['reports']:assert sha(row['path'])==row['sha256'];evidence[row['path']]=row['sha256']
 evidence.update(j['reports'])
 for name in ('read_control_snapshot.py','test_control_snapshot_host.py','audit_snapshot_cells.py',
              'audit_control_state_implementation.py','audit_snapshot_jtag_paths.py','snapshot_clock_cdc.py',
              'control_snapshot_integration.py','prepare_control_state_board.py','control_state_specialization.py','audit_control_state_map.py'):
  evidence[str(HERE/name)]=sha(HERE/name)
 tests=(batch/'host_tests.log').read_text();assert 'Ran 11 tests' in tests and tests.rstrip().endswith('OK')
 evidence[str(batch/'host_tests.log')]=sha(batch/'host_tests.log');evidence[str(Path(__file__))]=sha(__file__)
 result=dict(status='CONTROL_STATE_CDC_REVIEW_FOR_READ_ONLY_DIAGNOSTICS',board=str(board),gray=str(gray),
  required_evidence=evidence,delta=delta,unknown_new_crossings=0,reviewed_inherited_classes=gi['classes'],
  sync_ff=20,held_payload_paths=j['held_data_endpoints'],held_payload_max_ns=j['max_held_data_delay_ns'],
  findings=['Only fixed-control state encoding/CE inference changed; original and candidate cycle comparisons pass. Pins, clocks, reset circuits and all data-path sources retained.',
   'New raw levels and request/ack toggles enter real two-FF chains; exact full mapped connections checked.',
   'Held payload is captured only after ack synchronization; routed data settles in less than one 100ns constrained TCK.',
   'Real control/APB stall, stopped-clock and functional-reset tests pass; host decodes actual RTL scan outputs.',
   'Observation can diagnose an APB timeout; neither a good snapshot nor old module tests declare board functionality proven.'],
  conditions=['User present, current physical-B enumeration, frozen bit and evidence hashes.',
   'Read selectors 2/3 first; at most one APB identity command with bounded wait and post-failure observation.',
   'No DDR payload, no weight load, no CNN Run, no Flash write.'],
  ddr_workload_released=False,analogue_metastability_proven=False,board_tested=False)
 output.write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for name in ('board','batch','gray','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();review(**{k:v.resolve() for k,v in vars(a).items()})
