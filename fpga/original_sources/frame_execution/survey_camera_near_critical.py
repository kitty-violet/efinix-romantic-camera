"""Inspect a frozen route's near-critical endpoints; no placement/routing changes."""
import json,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/camera-checked-bound4'
OUT=ROOT/'tmp/frame-execution/camera-near-critical1'

def parse(body):
    rows=[]
    for block in body.split('++++ Path ')[1:]:
        def field(name):
            m=re.search(r'^'+name+r'\s*:\s*(.+)$',block,re.M);assert m,name
            return m[1].strip()
        data=block.split('\nData Path\n',1)[1].split('\nCapture Clock Path',1)[0]
        nets=[dict(net=n.strip(),fanout=int(fo),delay_ns=float(d)) for fo,d,n in
            re.findall(r'^\s+net \(fo=(\d+)\)\s+([\d.]+)\s+[\d.]+\s+(.+)$',data,re.M)]
        rows.append(dict(start=field('Path Start'),end=field('Path End'),
            slack_ns=float(field('Slack').split()[0]),nets=nets))
    assert rows
    return rows

def main():
    assert not OUT.exists();OUT.mkdir()
    files=[BASE/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.place','.vdb','.troutingtraces')]
    files += [BASE/'ti60f225_oob.sdc',BASE/'ti60f225_oob.peri.xml',Path(__file__).resolve()]
    frozen={str(f):sha(f) for f in files}
    for folder in ('outflow','work_pnr'):shutil.copytree(BASE/folder,OUT/folder)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,OUT/f.name)
    tcl='report_timing_summary -file initialized.rpt\nreport_timing -setup -from_clock {process_clk_150} -to_clock {process_clk_150} -npaths 200 -nworst 1 -file near_setup.rpt\n'
    (OUT/'survey.tcl').write_text(tcl)
    (OUT/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script survey.tcl --timeout 180\nexit /b %errorlevel%\n')
    (OUT/'inputs.json').write_text(json.dumps(dict(frozen=frozen,tcl_sha256=sha(OUT/'survey.tcl'),hardware_access=False),indent=2))
    start=time.monotonic()
    with (OUT/'survey.log').open('x') as log:
        child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=210)
        except subprocess.TimeoutExpired:
            (OUT/'timeout.json').write_text(json.dumps(stop_launched_tree(child)));raise
    log=(OUT/'survey.log').read_text()+(OUT/'outflow/ti60f225_oob.sta.out').read_text()
    assert code==0 and not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|^\s*(?:ERROR|FATAL)\s*:',log,re.M)
    report=OUT/'outflow/near_setup.rpt';rows=parse(report.read_text())
    for f,h in frozen.items():assert sha(f)==h
    assert min(r['slack_ns'] for r in rows)==-0.045
    result=dict(status='FROZEN_ROUTE_NEAR_CRITICAL_SURVEY_NOT_RELEASE',rows=rows,
        last_reported_slack_ns=rows[-1]['slack_ns'],seconds=time.monotonic()-start,
        inputs_sha256=sha(OUT/'inputs.json'),report_sha256=sha(report),hardware_access=False)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
