"""Stricter routing-only target for the one measured remaining setup failure."""
import argparse,json,re,runpy,shutil,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree import stop_launched_tree

START='video_processing_pipeline/pipeline/cnn/core/dut/engine/router/held_owner[4][1]~FF|CLK'
END='video_processing_pipeline/pipeline/cnn/core/arena/ROW[4].HALF[1].SLICE[2].mem|WDATAB[0]'
EXTRA='\n# Routing-only stricter target for measured held_owner-to-RAM path.\n'+f'set_max_delay -from [get_pins {{{START}}}] -to [get_pins {{{END}}}] 6.450\n'

def probe():
    base=ROOT/'tmp/frame-execution/camera-checked-bound4';root=ROOT/'tmp/frame-execution/camera-target-margin-probe1'
    assert not root.exists();root.mkdir()
    frozen={str(base/p):sha(base/p) for p in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.place',
        'outflow/ti60f225_oob.troutingtraces','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc','ti60f225_oob.peri.xml')}
    for folder in ('outflow','work_pnr'):shutil.copytree(base/folder,root/folder)
    for f in base.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,root/f.name)
    (root/'ti60f225_oob.sdc').write_text((base/'ti60f225_oob.sdc').read_text()+EXTRA,encoding='utf8')
    (root/'probe.tcl').write_text('report_timing_summary -file probe_summary.rpt\nreport_timing -setup -npaths 5 -nworst 1 -file probe_setup.rpt\ncheck_timing -file probe_checks.rpt -verbose\n')
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script probe.tcl --timeout 180\nexit /b %errorlevel%\n')
    (root/'inputs.json').write_text(json.dumps(dict(original=frozen,extra=EXTRA,script_sha256=sha(__file__),hardware_access=False),indent=2))
    started=time.monotonic()
    with (root/'sta.log').open('w') as log:
        child=subprocess.Popen(['cmd.exe','/d','/c','run.cmd'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=210)
        except subprocess.TimeoutExpired:(root/'timeout.json').write_text(json.dumps(stop_launched_tree(child)));raise
    assert code==0
    log=(root/'sta.log').read_text()+(root/'outflow/ti60f225_oob.sta.out').read_text()
    assert not re.search(r'Unable to run|No (?:ports|pins|clocks) matched|Timer cuts a combinational loop|^\s*(?:ERROR|FATAL)\s*:',log,re.M)
    body=(root/'outflow/probe_setup.rpt').read_text();matches=[]
    for p in body.split('++++ Path ')[1:]:
        def field(n):return re.search(r'^'+n+r'\s*:\s*(.+)$',p,re.M)[1].strip()
        if field('Path Start')==START and field('Path End')==END:matches.append(float(field('Slack').split()[0]))
    assert len(matches)==1 and matches[0]<-0.045,('Target must tighten actual path',matches)
    for f,h in frozen.items():assert sha(f)==h
    result=dict(status='TARGETED_MAX_DELAY_IS_STRICTER_SAME_ROUTE_PASS',before_ns=-0.045,after_ns=matches[0],
        seconds=time.monotonic()-started,inputs_sha256=sha(root/'inputs.json'),hardware_access=False,board_release=False,
        reports={str(p):sha(p) for p in (root/'outflow').glob('probe_*.rpt')})
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

def apply(physical):
    evidence=ROOT/'tmp/frame-execution/camera-target-margin-probe1';r=json.loads((evidence/'result.json').read_text())
    assert r['status']=='TARGETED_MAX_DELAY_IS_STRICTER_SAME_ROUTE_PASS' and r['after_ns']<r['before_ns']
    i=json.loads((evidence/'inputs.json').read_text());assert r['inputs_sha256']==sha(evidence/'inputs.json') and i['script_sha256']==sha(__file__)
    for f,h in {**i['original'],**r['reports']}.items():assert sha(f)==h
    sdc=physical/'ti60f225_oob.sdc';acceptance=physical/'camera_acceptance.sdc';assert not acceptance.exists()
    original=sdc.read_text();assert EXTRA not in original;shutil.copy2(sdc,acceptance)
    sdc.write_text(original+EXTRA,encoding='utf8')
    record=dict(status='ONE_PATH_TIGHTENED_FOR_ROUTING_ONLY',canonical_sdc=str(acceptance),canonical_sdc_sha256=sha(acceptance),
        optimization_sdc_sha256=sha(sdc),extra=EXTRA,script_sha256=sha(__file__),probe=str(evidence/'result.json'),probe_sha256=sha(evidence/'result.json'),
        hardware_clocks_changed=False,new_false_paths=0,new_multicycle_paths=0)
    receipt=physical/'targeted_route_margin.json';receipt.write_text(json.dumps(record,indent=2))
    meta_file=physical/'physical_inputs.json';meta=json.loads(meta_file.read_text())
    for row in meta['inputs']:
        if Path(row['snapshot'])==sdc:row['sha256']=sha(sdc)
    meta['targeted_route_margin_sha256']=sha(receipt);meta['clock_profile']['after_sha256']=sha(sdc)
    meta_file.write_text(json.dumps(meta,indent=2))
    review_file=physical/'route_review.json';review=json.loads(review_file.read_text())
    for f in (sdc,acceptance,receipt,meta_file,evidence/'result.json',evidence/'inputs.json',Path(__file__).resolve()):review['required_evidence'][str(f)]=sha(f)
    review_file.write_text(json.dumps(review,indent=2))

def canonical():
    source=HERE/'canonical_snapshot_sta.py';code=source.read_text()
    needle="margin=meta.get('setup_margin')\n"
    assert code.count(needle)==1
    code=code.replace(needle,needle+"""assert margin is None
target=base/'targeted_route_margin.json';rule=json.loads(target.read_text())
assert meta['targeted_route_margin_sha256']==sha(target)
assert rule['script_sha256']==sha(HERE/'camera_targeted_route_margin.py')
assert sha(rule['probe'])==rule['probe_sha256']
canonical=Path(rule['canonical_sdc'])
assert sha(canonical)==rule['canonical_sdc_sha256']
assert sha(base/'ti60f225_oob.sdc')==rule['optimization_sdc_sha256']
assert (base/'ti60f225_oob.sdc').read_text()==canonical.read_text()+rule['extra']
assert rule['extra']==EXTRA
""")
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();driver=root.parent/(root.name+'-targeted-canonical.py')
    assert not driver.exists();driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),adapter_sha256=sha(__file__),generated_sha256=sha(driver)),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]]
    runpy.run_path(str(driver),run_name='__main__',init_globals=dict(HERE=HERE,EXTRA=EXTRA))

if __name__=='__main__':
    mode=sys.argv.pop(1)
    if mode=='probe':probe()
    elif mode=='canonical':canonical()
    else:raise ValueError(mode)
