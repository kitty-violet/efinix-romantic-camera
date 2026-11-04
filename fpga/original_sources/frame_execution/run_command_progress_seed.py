"""A second documented placement search; reuse exact-map short proofs."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    source=a.source.resolve();root=a.root.resolve();mi=json.loads((source/'inputs.json').read_text());board=Path(mi['board'])
    assert root.is_relative_to(ROOT/'tmp') and not root.exists() and a.seed==3
    startup=source/'startup';sr=json.loads((startup/'result.json').read_text());si=json.loads((startup/'inputs.json').read_text())
    assert sr['status']=='COMMAND_PROGRESS_MAPPED_STARTUP_PASS_PENDING_SERIAL_FAULT' and sr['inputs_sha256']==sha(startup/'inputs.json')
    assert si['original_map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    for n,h in si['files'].items():assert sha(startup/n)==h
    serial=source.parent/(source.name+'-serial');assert json.loads((serial/'progress_counter_checks.json').read_text())['source_result_sha256']==sha(serial/'new/result.json')
    original=HERE/'run_control_command_physical.py';code=original.read_text()
    old="  run('startup','validate_command_progress_mapped.py',['--board',board,'--root',root/'startup'],180)\n  serial=root.parent/(root.name+'-serial')\n  run('serial','validate_command_progress_serial.py',['--startup',root/'startup','--root',serial],240)"
    new="  import shutil\n  shutil.copytree(Path("+repr(str(startup))+"),root/'startup')\n  serial=Path("+repr(str(serial))+")\n  (root/'reused_short_tests.json').write_text(json.dumps("+repr(dict(startup=str(startup),startup_result_sha256=sha(startup/'result.json'),serial=str(serial),serial_result_sha256=sha(serial/'new/result.json'),map_sha256=si['original_map_sha256']))+",indent=2))"
    assert code.count(old)==1;code=code.replace(old,new)
    assert code.count("'--seed',1,")==1;code=code.replace("'--seed',1,","'--seed',3,")
    old="  import audit_control_state_implementation as impl"
    new="  physical_result=json.loads((physical/'result.json').read_text())\n  assert all(s['exit_code']==0 for s in physical_result['stages']),physical_result\n  import audit_control_state_implementation as impl"
    assert code.count(old)==1;code=code.replace(old,new)
    driver=root.parent/(root.name+'-driver.py');assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(original),source_sha256=sha(original),generated_sha256=sha(driver),wrapper_sha256=sha(__file__),reused_source=str(source),only_implementation_change='placement seed 1 -> 3'),indent=2))
    sys.argv=[str(driver),'--board',str(board),'--root',str(root)];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
