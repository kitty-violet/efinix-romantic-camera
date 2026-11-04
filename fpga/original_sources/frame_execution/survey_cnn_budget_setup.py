"""Isolated setup-only survey; original full STA remains a separate release gate.

The vendor reporter crashed in check_timing after completing a 1000-path report.
Separate the diagnostic report into its own process instead of suppressing a
release check. No placement, route, SDC, or hardware change is performed.
"""
import json,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree
from survey_camera_near_critical import parse

BASE=ROOT/'tmp/frame-execution/cnn-run-budget-convergence-route1/canonical-seed3'
OUT=ROOT/'tmp/frame-execution/cnn-run-budget-setup-survey2'

def main():
    assert not OUT.exists()
    accepted=json.loads((BASE/'result.json').read_text())
    assert accepted['status']=='CANONICAL_ROUTED_TIMING_FAIL'
    assert accepted['original_and_route_unchanged']
    assert all(v==0 for v in accepted['check_timing_actual'].values())
    files=[BASE/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.place','.vdb','.troutingtraces')]
    files += [BASE/'ti60f225_oob.sdc',BASE/'result.json',BASE/'inputs.json',Path(__file__).resolve()]
    frozen={str(f):sha(f) for f in files}
    OUT.mkdir()
    for folder in ('outflow','work_pnr'):shutil.copytree(BASE/folder,OUT/folder)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,OUT/f.name)
    tcl='report_timing_summary -file setup_only_summary.rpt\nreport_timing -setup -npaths 1000 -nworst 1 -file setup_only.rpt\nputs {SETUP_SURVEY_COMPLETE}; flush stdout\n'
    (OUT/'survey.tcl').write_text(tcl)
    (OUT/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script survey.tcl --timeout 180\nexit /b %errorlevel%\n')
    (OUT/'inputs.json').write_text(json.dumps(dict(frozen=frozen,tcl_sha256=sha(OUT/'survey.tcl'),hardware_access=False),indent=2))
    started=time.monotonic()
    with (OUT/'survey.log').open('x') as log:
        child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=210)
        except subprocess.TimeoutExpired:
            (OUT/'timeout.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
    body=(OUT/'survey.log').read_text(errors='replace')+(OUT/'outflow/ti60f225_oob.sta.out').read_text(errors='replace')
    assert code==0 and 'SETUP_SURVEY_COMPLETE' in body
    assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|^\s*(?:ERROR|FATAL)\s*:',body,re.M)
    report=OUT/'outflow/setup_only.rpt';rows=parse(report.read_text())
    assert len(rows)==1000
    before=parse((BASE/'outflow/canonical_setup.rpt').read_text())
    for p,q in zip(before,rows):
        assert (p['start'],p['end'],p['slack_ns'],p['nets'])==(q['start'],q['end'],q['slack_ns'],q['nets'])
    for f,h in frozen.items():assert sha(f)==h
    for f in files[:4]:assert sha(OUT/'outflow'/f.name)==frozen[str(f)]
    result=dict(status='ISOLATED_SETUP_SURVEY_COMPLETE_NOT_RELEASE',paths=len(rows),
        original_and_route_unchanged=True,original_gate=str(BASE/'result.json'),original_gate_sha256=sha(BASE/'result.json'),
        report=str(report),report_sha256=sha(report),inputs_sha256=sha(OUT/'inputs.json'),
        worst_ns=rows[0]['slack_ns'],last_ns=rows[-1]['slack_ns'],seconds=time.monotonic()-started,
        hardware_access=False,board_release=False)
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
