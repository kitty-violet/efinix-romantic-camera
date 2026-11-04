"""One-file observation storage delta; functional video/A/controller preserved."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from registered_boundary_board import audit_delta as parent_audit
BASE=ROOT/'tmp/frame-execution/connected-board-split-registered2'
BOARD=ROOT/'tmp/frame-execution/connected-board-snapshot-ram2'
SHORT=ROOT/'tmp/frame-execution/snapshot-ram-shorts3'
BRIDGE=ROOT/'tmp/frame-execution/snapshot-ram-bridge2'
VAR=HERE/'variants/fixed_snapshot_ram/jtag_bringup_snapshot.sv'

def proofs():
 result={}
 for path,status in ((SHORT,'OFP5_RAM_SNAPSHOT_ISOLATED_PASS'),(BRIDGE,'OFP5_RAM_REAL_BRIDGE_RESET_PROTOCOL_PASS')):
  r=json.loads((path/'result.json').read_text());m=json.loads((path/'inputs.json').read_text())
  assert r['status']==status and r['inputs_sha256']==sha(path/'inputs.json')
  assert sha(path/VAR.name)==sha(VAR)
  for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
  if path==SHORT:
   assert len(r['results'])==3
   for case in r['results']:assert sha(path/f"seed{case['seed']}.log")==case['log_sha256']
  else:assert sha(path/'simulation.log')==r['log_sha256']
  result[str(path/'result.json')]=sha(path/'result.json')
 return result

def audit_delta(board):
 assert board==BOARD;parent_audit(BASE)
 evidence=proofs()
 m=json.loads((board/'inputs.json').read_text());p=json.loads((BASE/'inputs.json').read_text())
 expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in p['inputs']}
 seen=set();changed=[]
 for row in m['inputs']:
  dest=Path(row['snapshot']);assert dest.is_relative_to(board) and sha(dest)==row['sha256']
  rel=dest.relative_to(board).as_posix();seen.add(rel)
  if sha(dest)!=sha(BASE/rel):changed.append(rel)
 assert seen==expected and changed==['assembly/jtag_bringup_snapshot.sv']
 assert sha(board/'assembly'/VAR.name)==sha(VAR)
 return dict(status='OFP5_RAM_STORAGE_ONLY_DELTA_SHORT_PASS',changed=changed,
  inputs_sha256=sha(board/'inputs.json'),parent_inputs_sha256=sha(BASE/'inputs.json'),
  evidence=evidence,all_960_payload_bits_preserved=True,header_and_host_protocol_preserved=True,
  completion_latency_changed=True,full_video_a_preserved=True,pins_clocks_sdc_ip_unchanged=True,
  hardware_access=False,board_release=False)
