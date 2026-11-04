"""Prove a stricter process-only optimization target on unchanged routed inputs.

This is preparation for placement, never final STA acceptance or board release.
"""
import json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree import stop_launched_tree
from survey_camera_near_critical import parse

BASE=ROOT/'tmp/frame-execution/cnn-observer-trim-free-bound1'
ORIGINAL=ROOT/'tmp/frame-execution/cnn-sta-iso2'
PROBE=ROOT/'tmp/frame-execution/cnn-observer-margin-probe1'
RESERVE=0.900
EXTRA=('\n# Optimization only: measured process-domain routing deficit; acceptance is unchanged.\n'
       'set_clock_uncertainty -setup -from [get_clocks {process_clk_150}] '
       '-to [get_clocks {process_clk_150}] 0.900\n')

def relationships(text,kind):
    label='Setup (Max)' if kind=='setup' else 'Hold (Min)'
    part=text.split(label+' Clock Relationship\n',1)[1].split('\n\n',1)[0]
    rows={}
    for launch,capture,constraint,slack,edge in re.findall(r'^\s*(\S+)\s+(\S+)\s+([-\d.]+)\s+([-\d.]+)\s+(\([RF]-[RF]\))\s*$',part,re.M):
        key=(launch,capture,edge);assert key not in rows
        rows[key]=(float(constraint),float(slack))
    assert len(rows)==23,len(rows)
    return rows

def verify_probe():
    p=PROBE/'result.json';r=json.loads(p.read_text())
    assert r['status']=='PROCESS_ONLY_0P900_STRICTER_SAME_ROUTE_PROVED' and r['extra']==EXTRA
    for f,h in r['evidence'].items():assert sha(f)==h,f
    return r

def main():
    assert not PROBE.exists()
    baseline=json.loads((ORIGINAL/'result.json').read_text())
    assert baseline['status']=='CANONICAL_ROUTED_TIMING_FAIL'
    assert baseline['rows']['setup']['worst_ns']==-.845 and all(v==0 for v in baseline['check_timing_actual'].values())
    assert baseline['inputs_sha256']==sha(ORIGINAL/'inputs.json')
    original_inputs=json.loads((ORIGINAL/'inputs.json').read_text())
    for f,h in original_inputs['original'].items():assert sha(f)==h
    for n,h in baseline['reports'].items():assert sha(ORIGINAL/'outflow'/n)==h
    files=[BASE/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.place','.vdb','.troutingtraces')]
    files += [BASE/'ti60f225_oob.sdc',BASE/'ti60f225_oob.peri.xml',ORIGINAL/'result.json',ORIGINAL/'inputs.json',Path(__file__)]
    assert not re.search(r'^\s*set_clock_uncertainty\b',(BASE/'ti60f225_oob.sdc').read_text(),re.M)
    evidence={str(p):sha(p) for p in files}
    PROBE.mkdir()
    for folder in ('outflow','work_pnr'):shutil.copytree(BASE/folder,PROBE/folder)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,PROBE/f.name)
    (PROBE/'ti60f225_oob.sdc').write_text((BASE/'ti60f225_oob.sdc').read_text()+EXTRA,encoding='utf8')
    (PROBE/'probe.tcl').write_text('report_timing_summary -file probe_summary.rpt\nreport_timing -setup -npaths 5 -nworst 1 -file probe_setup.rpt\nreport_timing -hold -npaths 5 -nworst 1 -file probe_hold.rpt\nputs {PROBE_COMPLETE}; flush stdout\n')
    (PROBE/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script probe.tcl --timeout 180\nexit /b %errorlevel%\n')
    (PROBE/'inputs.json').write_text(json.dumps(dict(evidence=evidence,extra=EXTRA,hardware_access=False),indent=2))
    start=time.monotonic()
    try:
        with (PROBE/'probe.log').open('x') as log:
            child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=PROBE,stdout=log,stderr=subprocess.STDOUT)
            try:rc=child.wait(timeout=210)
            except subprocess.TimeoutExpired:
                (PROBE/'timeout.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
        text=(PROBE/'probe.log').read_text(errors='replace')+(PROBE/'outflow/ti60f225_oob.sta.out').read_text(errors='replace')
        assert rc==0 and 'PROBE_COMPLETE' in text
        assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|Timer cuts a combinational loop|^\s*(?:ERROR|FATAL)\s*:',text,re.M)
        for kind in ('setup','hold'):
            before=parse((ORIGINAL/'outflow'/('canonical_'+kind+'.rpt')).read_text())
            after=parse((PROBE/'outflow'/('probe_'+kind+'.rpt')).read_text())
            assert len(before)==len(after)==5
            for a,b in zip(before,after):
                assert (a['start'],a['end'],a['nets'])==(b['start'],b['end'],b['nets'])
                assert abs(a['slack_ns']-b['slack_ns']-(RESERVE if kind=='setup' else 0))<.002
        old=(ORIGINAL/'outflow/canonical_summary.rpt').read_text();new=(PROBE/'outflow/probe_summary.rpt').read_text()
        for kind in ('setup','hold'):
            before=relationships(old,kind);after=relationships(new,kind);assert set(before)==set(after)
            for key,(constraint,slack) in before.items():
                target=(kind=='setup' and key[:2]==('process_clk_150','process_clk_150'))
                assert after[key][0]==constraint
                assert abs(slack-after[key][1]-(RESERVE if target else 0))<.002,(kind,key)
        for f,h in evidence.items():assert sha(f)==h
        for f in files[:4]:assert sha(PROBE/'outflow'/f.name)==evidence[str(f)]
        for folder,prefix in ((ORIGINAL,'canonical'),(PROBE,'probe')):
            for kind in ('setup','hold','summary'):
                f=folder/'outflow'/(prefix+'_'+kind+'.rpt');evidence[str(f)]=sha(f)
        evidence[str(PROBE/'inputs.json')]=sha(PROBE/'inputs.json')
        result=dict(status='PROCESS_ONLY_0P900_STRICTER_SAME_ROUTE_PROVED',extra=EXTRA,setup_paths=5,hold_paths_unchanged=5,
            clock_setup_rows=23,clock_hold_rows=23,only_process_setup_changed=True,seconds=time.monotonic()-start,
            evidence=evidence,original_and_route_unchanged=True,hardware_access=False,board_release=False)
        (PROBE/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)
    except Exception as exc:
        (PROBE/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
