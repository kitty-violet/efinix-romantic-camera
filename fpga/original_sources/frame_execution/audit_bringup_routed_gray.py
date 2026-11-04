"""Measure Gray skew on frozen routed geometry in an uncut diagnostic copy.

The actual acceptance SDC is unchanged. This view only reinstates paths between
independent PLLs so this Efinity version does not return an empty skew report.
Its inter-PLL setup values are not used as synchronous timing acceptance.
"""
import argparse,json,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from ddr_user_clock_profile import GROUP
from bringup_status_cdc import inventory
from windows_process_tree import stop_launched_tree

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
p.add_argument('--root',type=Path,required=True);a=p.parse_args()
b=a.base.resolve();r=a.root.resolve()
assert b.is_relative_to(ROOT/'tmp') and r.is_relative_to(ROOT/'tmp') and not r.exists()
meta=json.loads((b/'physical_inputs.json').read_text())
terminal=json.loads((b/'result.json').read_text())
assert any(s['flow']=='pnr' and s['exit_code']==0 for s in terminal['stages'])
for item in meta['inputs']:assert sha(item['snapshot'])==item['sha256']
canonical=b/'canonical_timing.sdc'
assert sha(canonical)==meta['setup_margin']['canonical_sdc_sha256']
assert sha(b/'outflow/ti60f225_oob.map.v')==meta['base_map_sha256']
geometry=[b/'outflow'/('ti60f225_oob'+s) for s in ('.troutingtraces','.place','.vdb','.map.v')]
frozen={str(f):sha(f) for f in geometry+[canonical,b/'ti60f225_oob.sdc',b/'physical_inputs.json']}
sdc=canonical.read_text();assert sdc.count(GROUP)==1
assert not re.search(r'^\s*set_clock_uncertainty\b',sdc,re.M)
uncut=sdc.replace(GROUP,'# Diagnostic only: restore independent-PLL paths for actual Gray skew reports.')
assert sum(line.startswith('set_bus_skew ') for line in uncut.splitlines())==10
r.mkdir();(r/'runner.py').write_bytes(Path(__file__).read_bytes())
for folder in ('outflow','work_pnr'):shutil.copytree(b/folder,r/folder)
for f in b.iterdir():
    if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):
        shutil.copy2(f,r/f.name)
for name in ('ddr516_reference.sv','ddr516_v4_controller.sv'):
    shutil.copy2(b/name,r/name);frozen[str(b/name)]=sha(b/name)
(r/'ti60f225_oob.sdc').write_text(uncut,encoding='utf-8')
here=Path(__file__).parent
reports=['ti60f225_oob.user_to_ddr_crossings.rpt','ti60f225_oob.ddr_to_user_crossings.rpt',
         'ti60f225_oob.gray_skew_uncut.rpt']
assert all(not (r/'outflow'/name).exists() for name in reports)
tcl=(here/'audit_ddr_clock_crossings.tcl').read_text()
tcl+='\nreport_bus_skew -npaths 10 -nworst 1 -file ti60f225_oob.gray_skew_uncut.rpt\n'
tcl+='report_timing_summary -file ti60f225_oob.timing.rpt\n'
(r/'gray.tcl').write_text(tcl)
(r/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script gray.tcl --timeout 300\nexit /b %errorlevel%\n')
(r/'inputs.json').write_text(json.dumps(dict(base=str(b),original=frozen,diagnostic_sdc_sha256=sha(r/'ti60f225_oob.sdc'),
    restored_group=GROUP,canonical_acceptance_unchanged=True,tool_budget_s=300,host_budget_s=330,
    scripts={str(f):sha(f) for f in (Path(__file__),here/'ddr_user_clock_profile.py',here/'audit_gray_skew_result.py',here/'bringup_status_cdc.py')},
    tcl_sha256=sha(r/'gray.tcl')),indent=2)+'\n')
try:
    start=time.monotonic()
    with (r/'sta.log').open('w') as log:
        proc=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=r,stdout=log,stderr=subprocess.STDOUT)
        try:code=proc.wait(timeout=330)
        except subprocess.TimeoutExpired:
            (r/'timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(proc),indent=2)+'\n');raise
    logtext=(r/'sta.log').read_text(errors='replace')+'\n'+'\n'.join(f.read_text(errors='replace') for f in (r/'outflow').glob('*.sta.out'))
    assert code==0,(code,logtext[-2000:])
    assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|Timer cuts a combinational loop|^\s*(?:ERROR|FATAL)\s*:',logtext,re.M)
    inv=inventory(r)
    (r/'clock_crossing_inventory.json').write_text(json.dumps(inv,indent=2)+'\n')
    check=subprocess.run([sys.executable,'-B',str(here/'audit_gray_skew_result.py'),'--board',str(r)],capture_output=True,text=True,timeout=30)
    assert check.returncode==0,check.stdout+check.stderr
    skew=json.loads((r/'gray_skew_audit.json').read_text())
    for name,digest in frozen.items():assert sha(name)==digest,('Original changed',name)
    for f in geometry:assert sha(r/'outflow'/f.name)==frozen[str(f)]
    for item in meta['inputs']:assert sha(item['snapshot'])==item['sha256']
    result=dict(status='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE',seconds=time.monotonic()-start,
        buses=skew['buses'],endpoints=skew['endpoints'],max_skew_ns=skew['max_skew_ns'],limit_ns=skew['limit_ns'],
        inputs_sha256=sha(r/'inputs.json'),inventory_sha256=sha(r/'clock_crossing_inventory.json'),
        skew_audit_sha256=sha(r/'gray_skew_audit.json'),original_and_route_unchanged=True,
        diagnostic_interpll_setup_not_acceptance=True,cdc_release=False,board_release=False)
    (r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
except Exception as e:
    (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n');raise
