"""Frozen-route comparison of one enable tree and the original failing path."""
import argparse,json,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from native_local_drivers import graph
from survey_camera_near_critical import parse
from windows_process_tree import stop_launched_tree

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    base=a.base.resolve();out=a.root.resolve();assert not out.exists() and out.is_relative_to(ROOT/'tmp')
    net=base/'outflow/ti60f225_oob.map.v';g=graph(net.read_text())
    signal=('net','local_driver_net_261',None)
    loads=[n+'|'+pin for n,(kind,pins) in g.items() for pin,ts in pins.items() if signal in ts and not (n=='LOCAL_DRIVER_261' and pin=='O')]
    assert len(loads)==18 and all(n.endswith('|CE') for n in loads)
    target='video_processing_pipeline/pipeline/cnn/core/arena/ROW[4].HALF[1].SLICE[2].mem|WDATAB[0]'
    out.mkdir()
    frozen={str(base/n):sha(base/n) for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.place','outflow/ti60f225_oob.vdb','outflow/ti60f225_oob.troutingtraces','ti60f225_oob.sdc','ti60f225_oob.peri.xml')}
    for folder in ('outflow','work_pnr'):shutil.copytree(base/folder,out/folder)
    for f in base.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc','.rcf'):shutil.copy2(f,out/f.name)
    tcl='report_timing_summary -file initialized.rpt\n'
    for name,pins in [('enable',loads),('original',[target])]:
        tcl+='report_timing -setup -to [get_pins {'+' '.join(pins)+'}] -npaths 64 -nworst 1 -file '+name+'.rpt\n'
    (out/'survey.tcl').write_text(tcl)
    (out/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script survey.tcl --timeout 180\nexit /b %errorlevel%\n')
    inputs=dict(frozen=frozen,script_sha256=sha(__file__),tcl_sha256=sha(out/'survey.tcl'),enable_endpoints=loads,hardware_access=False)
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2));start=time.monotonic()
    with (out/'survey.log').open('x') as log:
        child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=out,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=210)
        except subprocess.TimeoutExpired:(out/'timeout.json').write_text(json.dumps(stop_launched_tree(child)));raise
    body=(out/'survey.log').read_text()+(out/'outflow/ti60f225_oob.sta.out').read_text()
    assert code==0 and not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|^\s*(?:ERROR|FATAL)\s*:',body,re.M)
    for f,h in frozen.items():assert sha(f)==h
    rows={n:parse((out/'outflow'/f'{n}.rpt').read_text()) for n in ('enable','original')}
    assert {r['end'] for r in rows['enable']}==set(loads)
    assert len(rows['original'])==1 and rows['original'][0]['end']==target
    result=dict(status='FROZEN_CONTROL_ROUTE_COMPARISON_ONLY',rows=rows,
        seconds=time.monotonic()-start,inputs_sha256=sha(out/'inputs.json'),
        reports={str(out/'outflow'/f'{n}.rpt'):sha(out/'outflow'/f'{n}.rpt') for n in rows},hardware_access=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],minimums={n:min(r['slack_ns'] for r in rs) for n,rs in rows.items()})))

if __name__=='__main__':main()
