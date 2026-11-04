"""Re-audit a completed run; never rerun it or alter its frozen inputs/log."""
import argparse,json,re
from pathlib import Path
from run_units import sha

NOTE='# ** Note: (vsim-12126) Error and warning message counts have been restored: Errors=0, Warnings=553.'
def errors(body):
 # The old unanchored ERROR-space search also matched this simulator Note.
 # Ignore only that exact informational line, not arbitrary warnings/errors.
 return [line for line in body.splitlines() if line!=NOTE and re.search(r'(?i)(?:\*\*\s+(?:Error|Fatal)|ERROR:|ERROR |DDR3 Controller FAIL)',line)]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 assert not errors(NOTE)
 for message in ('# ERROR: DDR timing violation','# ** Error: failed','# ** Fatal: failed','# DDR3 Controller FAIL'):
  assert errors(NOTE+'\n'+message)
 m=json.loads((root/'inputs.json').read_text());f=json.loads((root/'failure.json').read_text())
 assert f['error']=='AssertionError()' and len(f['stages'])==3 and all(x['returncode']==0 for x in f['stages'])
 assert m['shift_guard'] and 'calibration and read/write' in m['scope']
 for n,h in m['files'].items():assert sha(root/n)==h
 body=(root/'simulation.log').read_text();assert not errors(body),errors(body)
 assert '# Errors: 0, Warnings: 553' in body
 assert re.search(r'DDR516_CAL_DONE .*pass=1',body) and '# DDR3 Controller PASS' in body
 assert 'DDR516_TRAFFIC write_beats=256 read_beats=256 writes=4 reads=4 bytes_each_direction=4096' in body
 assert 'DDR516_SCOREBOARD mismatches=0 b=4' in body
 match=re.search(r'SHIFT_AUDIT_COMPLETE requests=(\d+) setup_violations=(\d+)',body);assert match and int(match[2])==0
 result=dict(status='GUARDED_SHIFT_VENDOR_CALIBRATION_RW_PASS',requests=int(match[1]),violations=0,
  measurements=re.findall(r'^# (SHIFT_(?:SETUP|HIGH)_MEASURE.*)$',body,re.M),markers=re.findall(r'^# (DDR516_(?:CAL_DONE|TRAFFIC|SCOREBOARD).*|DDR3 Controller PASS)$',body,re.M),
  seconds=f['seconds'],stages=f['stages'],inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),board_release=False,
  post_audit=dict(reason='Exact zero-error simulator Note falsely matched old ERROR-space regex',original_failure_sha256=sha(root/'failure.json'),auditor_sha256=sha(__file__),negative_error_cases=4,rerun=False))
 assert not (root/'result.json').exists();(root/'result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='measurements'}))
if __name__=='__main__':main()
