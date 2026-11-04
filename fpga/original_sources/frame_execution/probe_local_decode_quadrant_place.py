"""Place the whole DSP cascade groups in four regions, stop before route.

All RTL, map, frequency and original timing constraints remain frozen. DSP
cascades stay intact; RAM placement reuse is a physical preference, not logic.
"""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from local_control_decode import audit_delta,once

def main():
    p=argparse.ArgumentParser(add_help=False)
    for n in ('root','base','cnn-ram-reference'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--automatic-cnn-ram',action='store_true')
    a,rest=p.parse_known_args();root=a.root.resolve();base=a.base.resolve();reference=a.cnn_ram_reference.resolve()
    audit_delta(base);assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'probe_fourpll_reset_physical.py';code=source.read_text()
    code=once(code,'from fourpll_reset_clock_profile import apply_profile','from control_ddr_clock_profile import apply_profile')
    code=once(code,'from fixed_board_dsp_locations import generate','from fixed_quadrant_dsp_locations import generate')
    insertion='assert a.fixed_dsp_layout and not a.related_gray_placement_reference\n'
    if not a.automatic_cnn_ram:
        insertion+=('from fixed_cnn_ram_locations import generate as retain_cnn_ram\n'+
            f'cnn_ram_layout=retain_cnn_ram(root,Path({str(reference)!r}))\n'+
            "dsp_layout['assignments_sha256']=sha(root/'fixed_dsp.placeloc')\n"+
            "dsp_layout['cnn_ram_placement_sha256']=sha(root/'cnn_ram_placement.json')\n")
    insertion+=f"dsp_layout['cnn_ram_policy']={'automatic' if a.automatic_cnn_ram else 'reference'!r}\n"
    code=once(code,'related_gray_layout=None\n',insertion+'related_gray_layout=None\n')
    code=once(code,"+' --timeout '+str(budget)+'\\nexit /b %errorlevel%\\n')",
        "+(' --pnr_opts pack place max_threads=4' if flow=='pnr' else '')+' --timeout '+str(budget)+'\\nexit /b %errorlevel%\\n')")
    code=once(code,"stages.append(dict(flow=flow,","stages.append(dict(flow=('pack_place' if flow=='pnr' else flow),")
    code=once(code,"flows=['interface','pnr']","flows=['interface','pack_place']")
    code=once(code,"status='PHYSICAL_CAPACITY_PROBE_FINISHED'","status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'")
    driver=root.parent/(root.name+'-place-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),automatic_cnn_ram=a.automatic_cnn_ram,
        max_threads=4,hardware_access=False,route_started=False),indent=2))
    sys.argv=[str(driver),'--root',str(root),'--base',str(base),*rest];runpy.run_path(str(driver),run_name='__main__')
    result=json.loads((root/'result.json').read_text());assert all(s['exit_code']==0 for s in result['stages'])
    assert not (root/'outflow/ti60f225_oob.troutingtraces').exists()
    from fixed_cnn_ram_locations import memory_cells,verify_placed
    if a.automatic_cnn_ram:
        cells,_=memory_cells(root/'outflow/ti60f225_oob.map.v')
        assert memory_cells(base/'outflow/ti60f225_oob.map.v')==memory_cells(root/'outflow/ti60f225_oob.map.v')
        locations={w[0]:list(map(int,w[1:])) for line in (root/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines()
                   if len(w:=line.split())==4 and w[0] in cells}
        assert len(locations)==len(cells)==113 and len({tuple(v) for v in locations.values()})==113
        ram=dict(status='CNN_RAM_113_NATIVE_AUTOMATIC_LOCATIONS_RECORDED',count=113,locations=locations,
                 placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)
    else:ram=verify_placed(root)
    (root/'cnn_ram_actual_locations.json').write_text(json.dumps(ram,indent=2))
    from audit_global_place import metrics
    from review_existing_routability import graph_review
    current=metrics(root);ref=ROOT/'tmp/frame-execution/command-progress-seed3-1/pnr-seed1';previous=metrics(ref)
    failed=ROOT/'tmp/frame-execution/local-decode-place1';prior=metrics(failed)
    def changes(other):
        return dict(weighted_hpwl=100*(current['weighted_hpwl_per_net']/other['weighted_hpwl_per_net']-1),
            **{k:100*(current['congestion_peaks'][k]/v-1) for k,v in other['congestion_peaks'].items()})
    review=dict(status='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW',current=current,reference=previous,
        changes_pct=changes(previous),failed_route_reference=str(failed),changes_vs_failed=changes(prior),
        graph=graph_review(root),reference_path=str(ref),route_started=False,hardware_access=False,board_release=False,
        automatic_cnn_ram=a.automatic_cnn_ram)
    (root/'placement_review_inputs.json').write_text(json.dumps(review,indent=2))
    print(json.dumps({k:review[k] for k in ('status','changes_pct','changes_vs_failed')}),flush=True)

if __name__=='__main__':main()
