"""Bounded broad path survey on a copy of frozen, original-constraint geometry."""
import argparse,collections,json,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();root=a.root.resolve()
    assert base.is_relative_to(ROOT/'tmp') and root.is_relative_to(ROOT/'tmp') and not root.exists()
    result=json.loads((base/'result.json').read_text());assert result['original_and_route_unchanged']
    root.mkdir();shutil.copy2(__file__,root/'runner.py')
    for d in ('outflow','work_pnr'):shutil.copytree(base/d,root/d)
    for f in base.iterdir():
        if f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,root/f.name)
    frozen={str(f):sha(f) for f in [base/'ti60f225_oob.sdc']+[base/'outflow'/('ti60f225_oob'+s) for s in ('.map.v','.place','.vdb','.troutingtraces')]}
    # The installed manual lists -detail, but this 2026.1.132 binary rejects it.
    # Use the already exercised default detailed format instead.
    commands=['report_timing -setup -npaths 256 -nworst 1 -file survey_setup.rpt']
    for name,pattern in [('source','*engine/mac/source/*'),('partial','*engine/mac/partial/*'),('rq','*engine/rq/*'),('writer','*engine/writer/*')]:
        commands.append('report_timing -setup -npaths 8 -nworst 1 -from {'+pattern+'} -file survey_'+name+'.rpt')
    (root/'survey.tcl').write_text('\n'.join(commands)+'\nputs {SURVEY_COMPLETE}\n')
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script survey.tcl --timeout 180\nexit /b %errorlevel%\n')
    (root/'inputs.json').write_text(json.dumps(dict(base=str(base),base_result_sha256=sha(base/'result.json'),frozen=frozen,runner_sha256=sha(__file__),host_budget_s=210),indent=2))
    start=time.monotonic()
    with (root/'survey.log').open('x') as log:
        child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=210)
        except subprocess.TimeoutExpired:
            (root/'timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
    logtext=(root/'survey.log').read_text(errors='replace')
    assert code==0 and 'Failed to run command' not in logtext,logtext[-2000:]
    reports={};families=collections.defaultdict(list)
    for f in (root/'outflow').glob('survey_*.rpt'):
        body=f.read_text();rows=[]
        for block in re.split(r'(?=^Slack\s*:)',body,flags=re.M)[1:]:
            slack=float(re.search(r'^Slack\s*:\s*([+\-\d.]+)',block)[1])
            startpin=re.search(r'^Path Start\s*:\s*(.*)',block,re.M)
            endpin=re.search(r'^Path End\s*:\s*(.*)',block,re.M)
            assert startpin and endpin
            data=block.split('Data Path',2)[-1].split('Capture Clock Path',1)[0]
            nets=[(float(t),n.strip()) for t,n in re.findall(r'^\s*net \(fo=\d+\)\s+([\d.]+)\s+[\d.]+\s+(.*)',data,re.M)]
            row=dict(slack_ns=slack,start=startpin[1].strip(),end=endpin[1].strip(),net_delay_ns=sum(t for t,n in nets),longest_net=max(nets,default=None))
            rows.append(row)
            if f.name=='survey_setup.rpt':
                family=re.sub(r'\[\d+\]','[]',row['start'])+' -> '+re.sub(r'\[\d+\]','[]',row['end'])
                families[family].append(slack)
        reports[f.name]=dict(sha256=sha(f),paths=rows)
    assert len(reports['survey_setup.rpt']['paths'])==256
    for f,h in frozen.items():assert sha(f)==h
    output=dict(status='ROUTED_PATH_SURVEY_NOT_RELEASE',seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),reports=reports,
        families=[dict(path=k,count=len(v),worst_ns=min(v)) for k,v in families.items()],hardware_access=False)
    (root/'result.json').write_text(json.dumps(output,indent=2));print(json.dumps(dict(status=output['status'],seconds=output['seconds'],families=output['families'])),flush=True)
if __name__=='__main__':main()
