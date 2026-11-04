"""Sparse unchanged data-route reuse, complete tiles/cascades, free remaining logic."""
import json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import locations

PLAN=ROOT/'tmp/board-test/cnn-budget-route-preservation-plan1.json'
EXPORT=ROOT/'tmp/frame-execution/cnn-budget-parent-route-export1'
PARENT=ROOT/'tmp/frame-execution/camera-factor-compact-bound3'

def plan(current):
    r=json.loads(PLAN.read_text())
    assert r['status']=='SELECTIVE_UNCHANGED_DATA_ROUTE_PRESERVATION_PLAN_NOT_APPLIED'
    for f,h in r['evidence'].items():assert sha(f)==h,f
    for k in ('all_driver_load_sets_exact','parameters_data_clock_enable_reset_exact','complete_tiles_cascades','clocks_resets_enables_routes_not_locked'):assert r[k],k
    assert r['selected_nets']==82 and r['ordinary_cells_retained']==2251 and r['ordinary_cells_free']==58130
    current=Path(current)
    assert sha(current/'outflow/ti60f225_oob.map.v')==r['evidence'][str(ROOT/'tmp/frame-execution/cnn-run-budget-place2/outflow/ti60f225_oob.map.v')]
    assert sha(current/'fixed_dsp.placeloc')==r['evidence'][str(ROOT/'tmp/frame-execution/cnn-run-budget-place2/fixed_dsp.placeloc')]
    # The exported reference trace is exactly the frozen passing parent's trace.
    assert sha(EXPORT/'outflow/ti60f225_oob.rcf_reference.troutingtraces')==sha(PARENT/'outflow/ti60f225_oob.troutingtraces')
    keep=(EXPORT/'outflow/ti60f225_oob.placer_keepout').read_text()
    available=set(re.findall(r'^(\S+)\s*\{\s*$',keep,re.M))
    nets=[x['net'] for x in r['selected']]
    assert len(set(nets))==82 and set(nets)<=available
    r['evidence'].update({str(PLAN):sha(PLAN),str(Path(__file__)):sha(__file__)})
    return r

def generate(root):
    root=Path(root);r=plan(root);target=root/'fixed_dsp.placeloc';before=sha(target)
    prior=locations(target);assert len(prior)==437
    additions={n:tuple(p) for n,p in r['assignments'].items() if n not in prior}
    for n,p in r['assignments'].items():
        if n in prior:assert prior[n]==tuple(p),(n,prior[n],p)
    combined={**prior,**additions}
    assert len(set(combined.values()))==len(combined),'Location collision'
    target.write_text(target.read_text()+'# Complete unchanged shared tiles/cascades for sparse data-route reuse.\n'+
        ''.join(f'{n} {x} {y} {z}\n' for n,(x,y,z) in sorted(additions.items())),encoding='utf8')
    after=sha(target)
    for name in ('cnn_ram_placement.json','arena_island_placement.json'):
        f=root/name;p=json.loads(f.read_text());assert p['assignments_sha256']==before
        p.update(before_preserved_assignments_sha256=before,assignments_sha256=after,preserved_generator_sha256=sha(__file__))
        f.write_text(json.dumps(p,indent=2),encoding='utf8')
    trace=root/'preserved_parent.troutingtraces';keep=root/'preserved_parent.placer_keepout'
    shutil.copy2(PARENT/'outflow/ti60f225_oob.troutingtraces',trace)
    shutil.copy2(EXPORT/'outflow/ti60f225_oob.placer_keepout',keep)
    rcf=root/'preserved_data.rcf'
    rcf.write_text('routeTraceFile '+trace.as_posix()+'\n'+''.join('restoreNetFromTraceFile '+x['net']+'\n' for x in r['selected']),encoding='utf8')
    ini=root/'efx_pnr_settings.ini';original_ini=ini.read_text()
    assert original_ini=='loc_assignment = fixed_dsp.placeloc\ndump_placeloc = on\n'
    ini.write_text(original_ini+'rcf_file = preserved_data.rcf\nplacer_keepout_file = preserved_parent.placer_keepout\n',encoding='utf8')
    r.update(status='SPARSE_82_DATA_ROUTE_PLACEMENT_PREPARED_NOT_RELEASE',
        before_assignments_sha256=before,assignments_sha256=after,added_assignments=len(additions),total_fixed_assignments=len(combined),
        original_ini=original_ini,generated_files={f.name:sha(f) for f in (trace,keep,rcf,ini,target)},
        original_parent_unchanged=True,placement_rerun=True)
    (root/'preserved_route_placement.json').write_text(json.dumps(r,indent=2),encoding='utf8')
    (root/'preserved_route_generator.py').write_bytes(Path(__file__).read_bytes())
    return r

def verify_placed(root):
    root=Path(root);r=json.loads((root/'preserved_route_placement.json').read_text())
    for f,h in r['evidence'].items():assert sha(f)==h,f
    for f,h in r['generated_files'].items():assert sha(root/f)==h,f
    assert sha(root/'preserved_route_generator.py')==sha(__file__)
    actual=locations(root/'outflow/ti60f225_oob.out.placeloc')
    for n,p in r['assignments'].items():assert actual[n]==tuple(p),(n,p,actual.get(n))
    return dict(status='SPARSE_82_NETS_COMPLETE_PLACED_ENDPOINTS_VERIFIED',nets=82,ordinary_cells=2251,
        total_verified_cells=len(r['assignments']),proof_sha256=sha(root/'preserved_route_placement.json'),
        placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)
