"""Bind the same-map successful short gates and preserve the failed attempt."""
import argparse,json,shutil
from pathlib import Path
from run_units import HERE,sha
from split_logic_outputs_board import audit_delta
from split_logic_outputs_flow import proof

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',required=True,type=Path);p.add_argument('--root',required=True,type=Path)
 p.add_argument('--serial',required=True,type=Path);p.add_argument('--unit',required=True,type=Path);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();serial=a.serial.resolve();assert not (root/'result.json').exists()
 import startup_cause_flow as parent_flow
 parent_flow.audit_delta=audit_delta
 audit_delta(board);proof(board,a.unit.resolve());mapped=sha(board/'outflow/ti60f225_oob.map.v');evidence={}
 def bind(file):
  evidence[str(file)]=sha(file);return json.loads(file.read_text())
 def verify(items):
  for n,h in items.items():assert sha(n)==h;n=Path(n);evidence[str(n)]=h
 adapter=root.parent/(root.name+'-adapter')
 for folder,status in ((root/'startup','COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT'),
  (root/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),(serial,'MAPPED_SERIAL_APB_OFS_RESET_PASS'),
  (adapter,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
  r=bind(folder/'result.json');i=bind(folder/'inputs.json')
  assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
  assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped
  verify({str(folder/n):h for n,h in i['files'].items()})
  verify({str(folder/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
 for n in ('progress_counter_checks.json','paired_reply_result.json'):
  assert not (serial/n).exists();shutil.copy2(serial.parent/n,serial/n);bind(serial/n)
 si=json.loads((serial/'inputs.json').read_text());assert si['tck_hz']==10000000
 alias=bind(serial.parent.parent/(serial.parent.name+'-reference-binding.json'))
 assert alias['status']=='TEST_ONLY_HOST_READY_REFERENCE_BOUND' and alias['map_sha256']==mapped
 verify({str(HERE/'split_logic_outputs_serial.py'):alias['wrapper_sha256'],str(board/'assembly/EfxSapphireSoc.v'):alias['rtl_sha256']})
 r=bind(root/'rtl/result.json');i=bind(root/'rtl/inputs.json')
 assert r['status']=='CONTROL_DDR_RTL_SERIAL_EVENTS_RESET_PASS'
 assert r['inputs_sha256']==sha(root/'rtl/inputs.json') and i['board_inputs_sha256']==sha(board/'inputs.json')
 verify({str(root/'rtl'/n):h for n,h in i['files'].items()})
 verify({str(root/'rtl/simulation.log'):r['log_sha256']})
 ma=bind(root/'map-audit/result.json');assert ma['map_sha256']==mapped and ma['checks']==17504;verify(ma['evidence'])
 for name in ('new_diagnostic_cells.json','held_data_structure.json'):
  r=bind(root/'map-audit'/name);assert r['map_sha256']==mapped
 if (root/'failure.json').exists():bind(root/'failure.json');bind(root/'stages.json')
 result=dict(status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS',map_sha256=mapped,
  board_inputs_sha256=sha(board/'inputs.json'),serial=str(serial),adapter=str(adapter),
  prior_serial_failure_preserved=True,recovery='Only test reference renamed via actual mapped FF sinks; hardware and all assertions retained.',
  evidence=evidence,runner_sha256=sha(__file__),hardware_access=False,physical_timing_pass=False,
  board_release=False,ddr_phy_calibration_proven=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':main()
