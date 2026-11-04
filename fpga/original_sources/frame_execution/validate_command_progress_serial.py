"""Extend proven mapped serial regression to OFP2 and real progress counters."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from read_control_progress import decode

def main():
 p=argparse.ArgumentParser();p.add_argument('--startup',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 startup=a.startup.resolve();root=a.root.resolve()
 assert root.parent==ROOT/'tmp/frame-execution' and not root.exists()
 prior=json.loads((startup/'result.json').read_text());assert prior['status']=='COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT'
 source=ROOT/'tmp/board-test/check-mapped-jtag-regression.py';code=source.read_text()
 for before,after in [('from read_first_fault import decode','from read_control_progress import decode'),
                      ('[257:0]','[385:0]'),('258','386'),("32'h4f465331","32'h4f465032"),
                      ("raw[35:34]!==1","raw[35:34]!==1 || raw[49:42]!==2")]:
  assert before in code;code=code.replace(before,after)
 before="sources={'old':R/'tmp/frame-execution/first-fault-reference-physical1/startup','new':R/'tmp/frame-execution/shift-guard-physical2/startup'}"
 assert code.count(before)==1;code=code.replace(before,"sources={'new':Path("+repr(str(startup))+")}")
 driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code)
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),startup_result_sha256=sha(startup/'result.json')),indent=2))
 sys.argv=[str(driver),root.name,'new']
 try:runpy.run_path(str(driver),run_name='__main__')
 except SystemExit as e:
  if e.code not in (None,0):raise
 result=json.loads((root/'new/result.json').read_text())
 assert result['status']=='MAPPED_SERIAL_APB_OFS_RESET_PASS'
 records=result['snapshots'];assert [r['live']['accepted_commands'] for r in records]==[1,2,4],records
 assert [r['live']['replies'] for r in records]==[1,2,4],records
 assert all(r['live']['control_state']['legal'] for r in records)
 assert records[0]['live']['ticks']<records[-1]['live']['ticks']
 (root/'progress_counter_checks.json').write_text(json.dumps(dict(status='REAL_MAPPED_PROGRESS_COUNTERS_PASS',accepted=[1,2,4],replies=[1,2,4],source_result_sha256=sha(root/'new/result.json'),hardware_access=False),indent=2))
 print('REAL_MAPPED_PROGRESS_COUNTERS_PASS')
if __name__=='__main__':main()
