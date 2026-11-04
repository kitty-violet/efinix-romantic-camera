"""Verify extra setup uncertainty on one unchanged routed design, privately."""
import argparse, json, re, shutil, subprocess, time
from pathlib import Path
from run_units import ROOT, EF, sha
from windows_process_tree import stop_launched_tree

p=argparse.ArgumentParser()
p.add_argument('--base',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--reserve-ns',type=float,default=0.3)
a=p.parse_args()
assert 0.3<=a.reserve_ns<=1.0 and abs(a.reserve_ns-round(a.reserve_ns,3))<1e-9
base=a.base.resolve(); root=a.root.resolve()
assert base.is_relative_to(ROOT/'tmp') and root.is_relative_to(ROOT/'tmp') and not root.exists()
route=base/'outflow/ti60f225_oob.troutingtraces'
frozen={str(f):sha(f) for f in (route,base/'outflow/ti60f225_oob.place',base/'outflow/ti60f225_oob.vdb',base/'ti60f225_oob.sdc')}
assert not re.search(r'^\s*set_clock_uncertainty\b',(base/'ti60f225_oob.sdc').read_text(),re.M)
root.mkdir(); rows=[]
(root/'runner.py').write_bytes(Path(__file__).read_bytes())
for reserve in (0.0,a.reserve_ns):
    dest=root/('canonical' if not reserve else 'reserve'); dest.mkdir()
    for folder in ('outflow','work_pnr'):
        shutil.copytree(base/folder,dest/folder)
    for f in base.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):
            shutil.copy2(f,dest/f.name)
    sdc=dest/'ti60f225_oob.sdc'
    if reserve:
        sdc.write_text(sdc.read_text()+'\n# Additional optimization setup margin; hold remains unchanged.\nset_clock_uncertainty -setup -to [get_clocks {*}] '+f'{reserve:.3f}'+'\n')
    tcl=dest/'margin.tcl'
    tcl.write_text('report_timing -setup -npaths 1 -nworst 1 -file margin_setup.rpt\nreport_timing -hold -npaths 1 -nworst 1 -file margin_hold.rpt\n')
    (dest/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script margin.tcl --timeout 120\nexit /b %errorlevel%\n')
    start=time.monotonic()
    with (dest/'sta.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=dest,stdout=log,stderr=subprocess.STDOUT)
        try: code=proc.wait(timeout=150)
        except subprocess.TimeoutExpired:
            cleanup=stop_launched_tree(proc)
            (dest/'timeout_cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
            raise
    seconds=time.monotonic()-start; body=(dest/'sta.log').read_text()
    assert code==0,(code,body[-2000:])
    assert "Unable to run" not in body,body[-3000:]
    def parse(name):
        f=dest/'outflow'/name
        if not f.exists(): f=dest/name
        text=f.read_text(); match=re.search(r'^Slack\s*:\s*([+\-\d.]+)',text,re.M)
        assert match,(name,text[-1000:])
        return dict(slack_ns=float(match[1]),report=str(f),sha256=sha(f))
    assert sha(dest/'outflow/ti60f225_oob.troutingtraces')==frozen[str(route)]
    rows.append(dict(reserve_ns=reserve,setup=parse('margin_setup.rpt'),hold=parse('margin_hold.rpt'),seconds=seconds,sdc_sha256=sha(sdc)))
assert abs(rows[0]['setup']['slack_ns']-rows[1]['setup']['slack_ns']-a.reserve_ns)<0.0011,rows
assert rows[0]['hold']['slack_ns']==rows[1]['hold']['slack_ns'],rows
for f,h in frozen.items(): assert sha(f)==h
result=dict(status='EXTRA_SETUP_MARGIN_SAME_ROUTE_VERIFIED',rows=rows,unchanged_original=frozen,
    rule=f'Only +{a.reserve_ns:.3f}ns setup uncertainty; no frequency, hold, I/O or path exception changes.',board_release=False)
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
