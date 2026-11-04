"""Recheck unchanged routed geometry with its frozen canonical timing constraints.

Optimization-only setup margin is removed in a private copy, never in the
physical run. This emits timing evidence, not a board or CDC release.
"""
import argparse, json, re, shutil, subprocess, time, traceback
from pathlib import Path
from run_units import ROOT, EF, sha
from windows_process_tree import stop_launched_tree
from timing_acceptance import timing_passes

p=argparse.ArgumentParser()
p.add_argument('--base',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--min-setup-slack-ns',type=float,default=0.0,
               help='Current gate is zero; use 0.300 explicitly for the deferred optimization goal')
a=p.parse_args(); base=a.base.resolve(); root=a.root.resolve()
assert 0<=a.min_setup_slack_ns<=1.0
assert base.is_relative_to(ROOT/'tmp') and root.is_relative_to(ROOT/'tmp') and not root.exists()
result=json.loads((base/'result.json').read_text())
assert any(s['flow']=='pnr' and s['exit_code']==0 for s in result['stages'])
meta=json.loads((base/'physical_inputs.json').read_text())
for rec in meta['inputs']: assert sha(rec['snapshot'])==rec['sha256']
geometry=[base/'outflow'/('ti60f225_oob'+s) for s in ('.troutingtraces','.place','.vdb','.map.v')]
assert sha(geometry[-1])==meta['base_map_sha256']
frozen={str(f):sha(f) for f in geometry}
canonical=base/'ti60f225_oob.sdc'
margin=meta.get('setup_margin')
if margin:
    canonical=Path(margin['canonical_sdc'])
    assert sha(canonical)==margin['canonical_sdc_sha256']
    assert sha(base/'ti60f225_oob.sdc')==margin['optimization_sdc_sha256']
    reserve=margin['extra_setup_ns']
    assert 0.3<=reserve<=1.0 and abs(reserve-round(reserve,3))<1e-9 and margin['extra_hold_ns']==0
    extra='\n# Reserve '+f'{reserve:.3f}'+'ns during optimization; final acceptance is unchanged.\nset_clock_uncertainty -setup -to [get_clocks {*}] '+f'{reserve:.3f}'+'\n'
    allowed=[canonical.read_text()+extra]
    if reserve==0.3:
        allowed.append(canonical.read_text()+'\n# Reserve the project-required 0.300ns during optimization; hold unchanged.\nset_clock_uncertainty -setup -to [get_clocks {*}] 0.300\n')
    assert (base/'ti60f225_oob.sdc').read_text() in allowed
frozen[str(canonical)]=sha(canonical)
frozen[str(base/'ti60f225_oob.sdc')]=sha(base/'ti60f225_oob.sdc')
root.mkdir()
(root/'runner.py').write_bytes(Path(__file__).read_bytes())
(root/'timing_acceptance.py').write_bytes(Path(__file__).with_name('timing_acceptance.py').read_bytes())
for folder in ('outflow','work_pnr'): shutil.copytree(base/folder,root/folder)
for f in base.iterdir():
    if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'): shutil.copy2(f,root/f.name)
shutil.copy2(canonical,root/'ti60f225_oob.sdc')
assert sha(root/'ti60f225_oob.sdc')==sha(canonical)
names=['canonical_setup.rpt','canonical_hold.rpt','canonical_checks.rpt','canonical_summary.rpt']
assert all(not (root/'outflow'/n).exists() for n in names)
(root/'canonical.tcl').write_text(
    'check_timing -file canonical_checks.rpt -verbose\n'
    'report_timing -setup -npaths 5 -nworst 1 -file canonical_setup.rpt\n'
    'report_timing -hold -npaths 5 -nworst 1 -file canonical_hold.rpt\n'
    'report_timing_summary -file canonical_summary.rpt\n')
(root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script canonical.tcl --timeout 180\nexit /b %errorlevel%\n')
(root/'inputs.json').write_text(json.dumps(dict(base=str(base),base_physical_sha256=sha(base/'physical_inputs.json'),original=frozen,canonical_sdc_sha256=sha(canonical),removed_setup_margin=margin,runner_sha256=sha(__file__),timing_policy_sha256=sha(root/'timing_acceptance.py'),min_setup_slack_ns=a.min_setup_slack_ns,min_hold_slack_ns=0),indent=2)+'\n')
try:
    started=time.monotonic()
    with (root/'sta.log').open('w') as log:
        proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        try: code=proc.wait(timeout=210)
        except subprocess.TimeoutExpired:
            cleanup=stop_launched_tree(proc)
            (root/'timeout_cleanup.json').write_text(json.dumps(cleanup,indent=2)+'\n')
            raise
    body=(root/'sta.log').read_text(errors='replace')
    body+='\n'+'\n'.join(f.read_text(errors='replace') for f in (root/'outflow').glob('*.sta.out'))
    assert code==0,(code,body[-3000:])
    assert not re.search(r"Unable to run|No (?:ports|pins|clocks) matched|Timer cuts a combinational loop|^\s*(?:ERROR|FATAL)\s*:",body,re.M),body[-5000:]
    rows={}
    for kind in ('setup','hold'):
        f=root/'outflow'/('canonical_'+kind+'.rpt'); text=f.read_text()
        slacks=[float(v) for v in re.findall(r'^Slack\s*:\s*([+\-\d.]+)',text,re.M)]
        assert slacks,(kind,'No timing paths')
        rows[kind]=dict(worst_ns=min(slacks),report=str(f),sha256=sha(f))
    summary=(root/'outflow/canonical_summary.rpt').read_text()
    assert 'process_clk_150' in summary and 'ddr_core_clk200' in summary
    checks=(root/'outflow/canonical_checks.rpt').read_text()
    raw={k:int(v) for k,v in re.findall(r'checking (\w+) \((\d+)\)',checks)}
    assert set(raw)=={'no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','multiple_clock'}
    # Tool headings include deliberately false-pathed asynchronous ports.
    # Preserve both raw and actual unconstrained counts, as the main gate does.
    actual=dict(raw)
    for direction in ('input','output'):
        m=re.search(r'There are (\d+) '+direction+r' ports with no '+direction+r' delay specified',checks)
        assert m
        actual['no_'+direction+'_delay']=int(m[1])
    for f,h in frozen.items(): assert sha(f)==h,('Original changed',f)
    for f in geometry: assert sha(root/'outflow'/f.name)==frozen[str(f)],('Routed geometry changed',f)
    timing_ok=timing_passes(rows['setup']['worst_ns'],rows['hold']['worst_ns'],actual,a.min_setup_slack_ns)
    output=dict(status='CANONICAL_ROUTED_TIMING_PASS_NOT_RELEASE' if timing_ok else 'CANONICAL_ROUTED_TIMING_FAIL',rows=rows,check_timing_raw=raw,check_timing_actual=actual,seconds=time.monotonic()-started,inputs_sha256=sha(root/'inputs.json'),reports={n:sha(root/'outflow'/n) for n in names},original_and_route_unchanged=True,min_setup_slack_ns=a.min_setup_slack_ns,min_hold_slack_ns=0,extra_0p300_goal_met=rows['setup']['worst_ns']>=0.3,cdc_release=False,board_release=False)
    (root/'result.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)
except Exception as e:
    (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n')
    raise
