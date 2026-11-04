"""One timing-recipe route comparison, same map/placement/original SDC."""
import argparse,inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from control_command_repair import once

BASE=ROOT/'tmp/frame-execution'
PHYSICAL=BASE/'native-split-timing-place1'
BATCH=BASE/'native-split-timing-route1'
SURVEY=BASE/'native-split-path-survey1'

def save(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf8')

def prepare():
    assert not PHYSICAL.exists() and not BATCH.exists()
    selected=json.loads((BASE/'native-split-route2/selected.json').read_text())
    assert (selected['wns_ns'],selected['whs_ns'])==(-3.715,.007)
    analysis=json.loads((SURVEY/'data_arc_analysis.json').read_text())
    assert analysis['sampled_endpoints']==1000 and analysis['sampled_negative']==982
    assert len(analysis['families'])==105
    assert analysis['source_result_sha256']==sha(SURVEY/'result.json')
    assert sha(analysis['report'])==analysis['report_sha256']
    failed=[r for r in analysis['paths'] if r['slack_ns']<0]
    assert sum(r['net_fraction'] for r in failed)/len(failed)>.89
    import rebind_native_placement as rebind
    argv=sys.argv;sys.argv=[str(Path(rebind.__file__)),'--root',str(PHYSICAL)]
    try:rebind.main()
    finally:sys.argv=argv
    # Reuse every current placement/control/clock/Gray warning gate, only
    # append actual routed analysis and one official CLI route recipe.
    import review_native_split_route as review
    code=inspect.getsource(review.main)
    code=once(code,"ROOT/'tmp/frame-execution/native-split-bound-place1'",'PHYSICAL')
    code=once(code,'budget_seconds=900','budget_seconds=1350')
    anchor="    (physical/'route_review.json').write_text"
    insertion="""    record['rationale']+=' Actual C2 routing is legal but canonical WNS=-3.715ns and WHS=+.007ns. Same-route survey1000 endpoints includes982 failures in105 families, allprocess_clk_150; mean data-net share89.06%,922/982 over80%,593 contain a data net>=2ns. Current128-pair native reimport differs from older broad split maps. Test officialTIMING_1 route only once on exact same map, locations and SDC; no added latency or functional change.'
    record['remaining_risks']+=['Older different maps failed TIMING_1; recipe names do not promise improvement. This is not a blind seed sweep.', 'Current complete capacity fits but setup fails widely; no subset of paths can release this build.']
    record['budget_reason']='One1350s route,1800s whole Job cap; no remap/place or repeated mapped shorts. Compare actual final paths on failure.'
    record['route_profile_cli_override']={'before':'CONGESTION_2','after':'TIMING_1'}
    for f in (SURVEY/'data_arc_analysis.json',SURVEY/'result.json',SURVEY/'outflow/canonical_setup.rpt',
              BASE/'native-split-route2/selected.json',BASE/'native-split-route2/failure.json',
              HERE/'reroute_native_timing.py',EF/'bin/efx_pnr_options.xml',EF/'scripts/aoe/core/options_constants.py'):
        record['required_evidence'][str(f)]=sha(f)
"""
    code=once(code,anchor,insertion+anchor)
    scope=dict(review.__dict__,__file__=__file__,PHYSICAL=PHYSICAL,SURVEY=SURVEY,BASE=BASE,EF=EF)
    exec(compile(code,__file__+':placement-review','exec'),scope);scope['main']()
    old=BASE/'native-split-bound-place2'
    names=['outflow/ti60f225_oob'+s for s in ('.map.v','.vdb','.place','.out.placeloc','.interface.csv')]
    names+=['ti60f225_oob.sdc','ti60f225_oob.xml','ti60f225_oob.peri.xml','fixed_dsp.placeloc','efx_pnr_settings.ini']
    files={str(old/n):sha(old/n) for n in names}
    copies={str(PHYSICAL/n):sha(PHYSICAL/n) for n in names}
    assert list(files.values())==list(copies.values())
    save(PHYSICAL/'route_inheritance.json',dict(source_files=files,copied_files=copies,
        map_changed=False,placement_changed=False,constraints_changed=False,seed_changed=False,
        route_profile_cli_override={'before':'CONGESTION_2','after':'TIMING_1'},hardware_access=False))
    record=json.loads((PHYSICAL/'route_review.json').read_text())
    record['required_evidence'][str(PHYSICAL/'route_inheritance.json')]=sha(PHYSICAL/'route_inheritance.json')
    save(PHYSICAL/'route_review.json',record)
    print('NATIVE_TIMING_RECIPE_COMPARISON_PREPARED',flush=True)

def driver_code():
    source=HERE/'startup_cause_route.py';code=source.read_text()
    code=once(code,'from startup_cause_board import audit_delta','from native_split_board import audit_delta')
    code=once(code,"HERE/'audit_startup_cause_gray.py'","HERE/'audit_native_split_gray.py'")
    insertion=""" changes['--pnr_opts route max_router_iterations=100 troute_options=max_iterations=100']='--pnr_opts route max_router_iterations=100 troute_options=max_iterations=100 optimization_level=TIMING_1'
 changes["        ps=old['stages']+"]="        assert 'optimization_level=TIMING_1' in (root/'route.log').read_text()\\n        inheritance=json.loads((physical/'route_inheritance.json').read_text())\\n        for section in ('source_files','copied_files'):\\n            for path,h in inheritance[section].items():assert sha(path)==h,path\\n        ps=old['stages']+"
"""
    code=once(code,' for before,after in changes.items():',insertion+' for before,after in changes.items():')
    return source,code

def execute():
    source,code=driver_code();driver=BASE/(BATCH.name+'-native-wrapper.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    save(driver.with_suffix('.json'),dict(source=str(source),source_sha256=sha(source),
        wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False))
    sys.argv=[str(driver),'--physical',str(PHYSICAL),'--root',str(BATCH),
        '--review',str(PHYSICAL/'route_review.json'),'--evidence-root',str(BASE/'native-split-evidence1'),
        '--serial-proof',str(BASE/'native-split-evidence1-serial/new'),'--budget','1350']
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.action=='prepare' else execute()
