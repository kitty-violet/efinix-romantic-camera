"""Finish an unchanged diagnostic after handing off its host watchdog.

This does not promote a result to strict PASS because the original parent exit
status is unavailable. It retains completed traffic/data evidence for diagnosis.
"""
import argparse,collections,json,re,time
from pathlib import Path
from run_units import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    handoff=json.loads((root/'watchdog_handoff.json').read_text())
    active=json.loads((root/'active_process.json').read_text());assert active['pid']==handoff['simulation_pid']
    start=time.monotonic()
    while True:
        body=(root/'simulation.log').read_text(errors='replace')
        if '# End time:' in body:break
        if time.monotonic()-start>600:raise TimeoutError('Existing simulator has not ended; retained, not killed')
        time.sleep(2)
    meta=json.loads((root/'inputs.json').read_text())
    for item in meta['inputs']:assert sha(item['snapshot'])==item['sha256'],item['snapshot']
    assert 'DDR516_TRAFFIC' in body,'No completed normal traffic'
    errors=[line for line in body.splitlines() if re.search(r'\b(?:ERROR|Error):|\*\* (?:Error|Fatal)',line)]
    result=dict(status='DDR516_DIAGNOSTIC_RECOVERED_FAIL' if errors else 'DDR516_DIAGNOSTIC_RECOVERED_EXIT_UNVERIFIED',
        seconds=time.time()-active['started'],stages=[dict(stage='simulation',returncode=None,wall_timeout=False,watchdog_handoff=True)],
        error_count=len(errors),first_errors=errors[:12],
        error_categories=dict(collections.Counter(re.sub(r'.*ERROR: ?','',x) for x in errors)),
        markers=[line for line in body.splitlines() if 'DDR516_' in line or 'DDR3 Controller ' in line],
        inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),board_release=False,full_board=False,
        recovery='Only original Python watchdog was terminated; original Questa continued to normal test end. Process return code unavailable; no strict PASS inferred.')
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','seconds','first_errors','log_sha256')}),flush=True)

if __name__=='__main__':main()
