"""Read-only endpoint survey of a completed canonical candidate, never release."""
import argparse,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree
from survey_camera_near_critical import parse

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--canonical',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();base=a.canonical.resolve();out=a.root.resolve()
    assert base.is_relative_to(ROOT/'tmp') and out.is_relative_to(ROOT/'tmp') and not out.exists()
    result=json.loads((base/'result.json').read_text())
    assert result['status'] in ('CANONICAL_ROUTED_TIMING_PASS_NOT_RELEASE','CANONICAL_ROUTED_TIMING_FAIL'),result['status']
    inputs=json.loads((base/'inputs.json').read_text())
    for f,h in inputs['original'].items():assert sha(f)==h,f
    files=[base/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.place','.vdb','.troutingtraces')]
    files += [base/'ti60f225_oob.sdc',base/'inputs.json',base/'result.json',base/'outflow/canonical_setup.rpt',Path(__file__).resolve()]
    frozen={str(f):sha(f) for f in files};out.mkdir()
    for folder in ('outflow','work_pnr'):shutil.copytree(base/folder,out/folder)
    for f in base.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,out/f.name)
    (out/'survey.tcl').write_text('report_timing_summary -file survey_summary.rpt\nreport_timing -setup -npaths 1000 -nworst 1 -file survey_setup.rpt\nputs {SURVEY_COMPLETE}; flush stdout\n')
    (out/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script survey.tcl --timeout 180\nexit /b %errorlevel%\n')
    (out/'inputs.json').write_text(json.dumps(dict(canonical=str(base),frozen=frozen,tcl_sha256=sha(out/'survey.tcl'),hardware_access=False),indent=2))
    try:
        start=time.monotonic()
        with (out/'survey.log').open('x') as log:
            child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=out,stdout=log,stderr=subprocess.STDOUT)
            try:rc=child.wait(timeout=210)
            except subprocess.TimeoutExpired:
                (out/'timeout.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        log=(out/'survey.log').read_text(errors='replace')+(out/'outflow/ti60f225_oob.sta.out').read_text(errors='replace')
        assert rc==0 and 'SURVEY_COMPLETE' in log,(rc,log[-1500:])
        assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|^\s*(?:ERROR|FATAL)\s*:',log,re.M)
        for f,h in frozen.items():assert sha(f)==h,f
        for f in files[:4]:assert sha(out/'outflow'/f.name)==frozen[str(f)]
        report=out/'outflow/survey_setup.rpt';rows=parse(report.read_text())
        before=parse((base/'outflow/canonical_setup.rpt').read_text())
        assert len(rows)==1000 and len(before)>0
        for x,y in zip(before,rows):assert x==y,(x,y)
        record=dict(status='CANONICAL_GEOMETRY_ENDPOINT_SURVEY_COMPLETE_NOT_RELEASE',paths=len(rows),
                    negative_paths=sum(x['slack_ns']<0 for x in rows),worst_ns=rows[0]['slack_ns'],
                    last_ns=rows[-1]['slack_ns'],negative_endpoint_coverage_complete=rows[-1]['slack_ns']>=0,
                    seconds=time.monotonic()-start,inputs_sha256=sha(out/'inputs.json'),report=str(report),
                    report_sha256=sha(report),original_and_route_unchanged=True,hardware_access=False,board_release=False)
        (out/'result.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
