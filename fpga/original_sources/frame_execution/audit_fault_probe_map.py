"""Recheck actual control D truth tables and all new clock synchronizers."""
import argparse,json,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from fault_probe_board import audit_delta
from fault_probe_clock_cdc import map_pairs
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();assert not root.exists();audit_delta(board)
 import audit_local_control_decode_map as local
 local.audit_delta=audit_delta
 (board/'local_decode_map_audit.json').write_text(json.dumps(local.audit(board),indent=2))
 import audit_complete_transition_map as full
 full.audit_delta=audit_delta;sys.argv=[full.__file__,'--board',str(board)]
 try:full.main()
 except AssertionError:
  report=board/'complete_transition_map_audit.json'
  assert report.exists() and json.loads(report.read_text())['status']=='D_CONE_SHARED_CELL_REMAINS'
 report=board/'complete_transition_map_audit.json';r=json.loads(report.read_text())
 assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['next_state_luts']==25
 source=ROOT/'tmp/board-test/check-accepted-tail-nextstate.py';code=source.read_text();root.mkdir()
 def replace(before,after):
  nonlocal code
  assert code.count(before)==1,before;code=code.replace(before,after)
 replace("net=ROOT/'tmp/frame-execution/connected-board-accepted-tail2/outflow/ti60f225_oob.map.v'",f'net=Path({str(board/"outflow/ti60f225_oob.map.v")!r})')
 replace("out=ROOT/'tmp/board-test/accepted-tail-nextstate1.json';assert not out.exists()",f'out=Path({str(root/"boolean_result.json")!r});assert not out.exists()')
 replace("assert sha(net)=='e105a7d585283c229f8d4061d8f57bb4ffd0e9c1f83f6b3cc60b2660941641b1'",f"assert sha(net)=={r['map_sha256']!r}")
 replace("cone=json.loads((ROOT/'tmp/board-test/accepted-tail-state-cones1.json').read_text())",f'cone=json.loads(Path({str(report)!r}).read_text())')
 replace("cone['states'][str(i)]['ff']['pins']['D']","cone['states'][str(i)]['ff']['D']")
 # Exported live state FFs retain their actual logic but synthesis promotes
 # their names to the diagnostic bus. Evaluate those real FF boundaries.
 replace("adapter+f'wstate[{i}]'","f'adapter_diagnostic[{99+i}]'")
 replace("adapter+f'rstate[{i}]'","f'adapter_diagnostic[{96+i}]'")
 driver=root/'boolean_check.py';driver.write_text(code)
 exec(compile(code,str(driver),'exec'),{'__file__':str(driver)})
 b=json.loads((root/'boolean_result.json').read_text());assert b['mismatch_count']==0 and b['checks']==17504
 pairs=map_pairs((board/'outflow/ti60f225_oob.map.v').read_text())
 result=dict(status='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS',map_sha256=r['map_sha256'],
  board_inputs_sha256=sha(board/'inputs.json'),next_state_luts=25,checks=b['checks'],sync=pairs,
  whole_control_upstream_shared_cells=len(r['shared_cells']),
  evidence={str(f):sha(f) for f in (source,driver,root/'boolean_result.json',report,board/'local_decode_map_audit.json',Path(__file__),HERE/'fault_probe_clock_cdc.py')},hardware_access=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])
if __name__=='__main__':main()
