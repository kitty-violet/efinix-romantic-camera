"""One reviewed same-placement route with a eight-path optimization target.

The temporary SDC is archived, used only during route, then restored byte for
byte before the original-constraint STA, Gray, and complete release gates.
No original evidence or source generator is modified.
"""
import json, runpy, shutil, sys
from pathlib import Path
from run_units import ROOT, HERE, sha
from control_command_repair import once

F = ROOT/'tmp/frame-execution'
BASE = F/'camera-factor-compact-bound1'
PROBE = F/'camera-compact-eight-path-probe1'
ANALYSIS=ROOT/'tmp/board-test/camera-compact-routed-path-analysis2.json'
analysis=json.loads(ANALYSIS.read_text())
assert analysis['status']=='FROZEN_ROUTED_FAILURE_ANALYSIS_NOT_RELEASE'
targets=[p for p in analysis['paths'] if p['slack_ns']<0]
assert len(targets)==8 and len({p['group'] for p in targets})==6
EXTRA='\n# Optimization only: the eight measured failing paths, six groups.\n'
for p in targets:
    assert '{' not in p['start']+p['end'] and '}' not in p['start']+p['end']
    EXTRA += 'set_max_delay -from [get_pins {'+p['start']+'}] -to [get_pins {'+p['end']+'}] 6.366\n'


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf8')

def probe():
    source = HERE/'canonical_snapshot_sta.py'
    code = source.read_text()
    anchor = "assert sha(root/'ti60f225_oob.sdc')==sha(canonical)"
    code = once(code, anchor, anchor+"\n(root/'ti60f225_oob.sdc').write_text(canonical.read_text()+EXTRA,encoding='utf8')")
    # This is deliberately a failing, stricter probe, never acceptance evidence.
    code = once(code, "root/'result.json'", "root/'raw_result.json'")
    code = once(code, "Path(__file__).with_name('timing_acceptance.py')", "(HERE/'timing_acceptance.py')")
    driver = PROBE.with_suffix('.py')
    assert not driver.exists() and not PROBE.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    sys.argv = [str(driver),'--base',str(BASE),'--root',str(PROBE)]
    runpy.run_path(str(driver),run_name='__main__',init_globals=dict(EXTRA=EXTRA,HERE=HERE))
    import re
    def paths(folder,kind):
        rows={}
        for body in (folder/'outflow'/('canonical_'+kind+'.rpt')).read_text().split('++++ Path ')[1:]:
            field=lambda n:re.search(r'^'+n+r'\s*:\s*(.+)$',body,re.M)[1].strip()
            rows[(field('Path Start'),field('Path End'))]=float(field('Slack').split()[0])
        return rows
    original = F/'camera-factor-compact-route1/canonical-seed3'
    before,after=paths(original,'setup'),paths(PROBE,'setup')
    assert len(before)==len(after)==5 and set(before)==set(after)
    assert all(abs(after[k]-before[k]+0.300)<0.002 for k in before)
    assert paths(original,'hold')==paths(PROBE,'hold')
    raw=json.loads((PROBE/'raw_result.json').read_text())
    assert all(v==0 for v in raw['check_timing_actual'].values())
    evidence={str(p):sha(p) for p in [source,Path(__file__),ANALYSIS,driver,PROBE/'raw_result.json',PROBE/'inputs.json',original/'result.json',original/'inputs.json']}
    evidence.update({str(p):sha(p) for folder in (PROBE,original) for p in (folder/'outflow').glob('canonical_*.rpt')})
    save(PROBE/'result.json',dict(status='EIGHT_PATH_SETUP_STRICTER_0P300_SAME_ROUTE_PASS',
        extra=EXTRA,setup_paths=5,hold_paths_unchanged=5,old_wns=min(before.values()),probe_wns=min(after.values()),
        evidence=evidence,hardware_access=False,board_release=False))
    print('PROCESS_MARGIN_PROBE_PASS',flush=True)

def verified_probe():
    r=json.loads((PROBE/'result.json').read_text())
    assert r['status']=='EIGHT_PATH_SETUP_STRICTER_0P300_SAME_ROUTE_PASS' and r['extra']==EXTRA
    for p,h in r['evidence'].items():assert sha(p)==h,p
    return r

def prepare():
    verified_probe()
    old=ROOT/'tmp/board-test/camera-compact-route-review1.json'
    review=json.loads(old.read_text())
    physical=F/'camera-factor-compact-bound3'
    review['physical']=str(physical)
    review['rationale'] += [
        'Same tested netlist, exact reviewed placement and complete hard anchors. No extra logic or shared clock/reset load; no placement rerun.',
        'The original legal route has 8 sampled failing endpoints in 6 process-domain groups, 87 to 96 percent net delay. Eight path-specific 6.366ns max-delay optimization targets jointly prioritize the measured failures without tightening every process-domain path.',
        'Frozen-route STA probe proves exactly 0.300ns tighter setup for five worst paths, unchanged hold paths and zero actual missing constraints. Original SDC restored byte-for-byte before all acceptance and release checks.'
    ]
    review['remaining_risks'] += ['Targeted stricter routing can increase congestion or move the worst endpoint; legal zero-overlap route and original global timing remain mandatory. Single bounded run, no automatic seed sweep.']
    review['reviewed_evidence'].update({str(PROBE/'result.json'):sha(PROBE/'result.json'),**verified_probe()['evidence']})
    analysis=ROOT/'tmp/board-test/camera-compact-routed-path-analysis2.json'
    review['reviewed_evidence'][str(analysis)]=sha(analysis)
    path=ROOT/'tmp/board-test/camera-compact-route-review3.json';assert not path.exists()
    save(path,review)
    import camera_factor_compact_flow as flow
    sys.argv=[str(HERE/'camera_factor_compact_flow.py'),'prepare-route','--review',str(path)]
    flow.main()
    print('SAME_PLACEMENT_PROCESS_TARGET_PREPARED',flush=True)

def route():
    verified_probe()
    original=F/'camera-factor-compact-route1-local-driver-route.py'
    code=original.read_text()
    anchor="        row=run('route',['cmd.exe','/d','/c',cmd],physical,a.budget+45)"
    change="""        from camera_compact_path_flow import EXTRA,PROBE,verified_probe
        verified_probe()
        sdc=physical/'ti60f225_oob.sdc';canonical_bytes=sdc.read_bytes()
        (root/'route_acceptance.sdc').write_bytes(canonical_bytes)
        optimized=sdc.read_text()+EXTRA
        (root/'route_optimization.sdc').write_text(optimized,encoding='utf8')
        receipt=dict(status='OPTIMIZATION_ONLY_RUNNING',canonical_sha256=sha(sdc),optimization_sha256=sha(root/'route_optimization.sdc'),
            probe=str(PROBE/'result.json'),probe_sha256=sha(PROBE/'result.json'),adapter_sha256=sha(HERE/'camera_compact_path_flow.py'),
            full_original_sta_required=True,hardware_clock_changed=False,placement_changed=False,extra=EXTRA)
        save(root/'route_optimization_inputs.json',receipt)
        sdc.write_bytes((root/'route_optimization.sdc').read_bytes())
        try:
            row=run('route',['cmd.exe','/d','/c',cmd],physical,a.budget+45)
        finally:
            sdc.write_bytes(canonical_bytes)
            assert sha(sdc)==receipt['canonical_sha256']
            receipt['status']='ORIGINAL_SDC_RESTORED_BEFORE_ACCEPTANCE'
            save(root/'route_optimization_inputs.json',receipt)
"""
    code=once(code,anchor,change.rstrip())
    compile(code,'process-margin-route','exec')
    # Existing generator remains the owner; its complete downstream gates stay
    # byte-identical. The new caller is separately bound in reviewed evidence.
    from camera_factor_compact_flow import generated
    generated(original,code,'-local-driver-route')

if __name__=='__main__':
    mode=sys.argv.pop(1)
    if mode=='probe':probe()
    elif mode=='prepare':prepare()
    elif mode=='route':route()
    else:raise ValueError(mode)
