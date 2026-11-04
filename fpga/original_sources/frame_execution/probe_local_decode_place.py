"""Official pack/place only, with hard-column anchors and routability evidence."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from local_control_decode import audit_delta
from control_command_repair import once

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--cnn-ram-reference',type=Path,required=True)
    a,rest=p.parse_known_args();root=a.root.resolve();base=a.base.resolve();reference=a.cnn_ram_reference.resolve()
    audit_delta(base);assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'probe_fourpll_reset_physical.py';code=source.read_text()
    code=once(code,'from fourpll_reset_clock_profile import apply_profile','from control_ddr_clock_profile import apply_profile')
    code=once(code,'related_gray_layout=None\n',
        'from fixed_cnn_ram_locations import generate as retain_cnn_ram\n'
        'assert a.fixed_dsp_layout and not a.related_gray_placement_reference\n'+
        f'cnn_ram_layout=retain_cnn_ram(root,Path({str(reference)!r}))\n'+
        "dsp_layout['assignments_sha256']=sha(root/'fixed_dsp.placeloc')\n"+
        "dsp_layout['cnn_ram_placement_sha256']=sha(root/'cnn_ram_placement.json')\nrelated_gray_layout=None\n")
    code=once(code,"+' --timeout '+str(budget)+'\\nexit /b %errorlevel%\\n')",
        "+(' --pnr_opts pack place' if flow=='pnr' else '')+' --timeout '+str(budget)+'\\nexit /b %errorlevel%\\n')")
    code=once(code,"stages.append(dict(flow=flow,","stages.append(dict(flow=('pack_place' if flow=='pnr' else flow),")
    code=once(code,"flows=['interface','pnr']","flows=['interface','pack_place']")
    code=once(code,"status='PHYSICAL_CAPACITY_PROBE_FINISHED'","status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'")
    driver=root.parent/(root.name+'-place-driver.py');assert not driver.exists();driver.parent.mkdir(parents=True,exist_ok=True)
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),
        change='Official native pack/place stop; same DSP cascade and CNN RAM anchors, no general logic fixed.',hardware_access=False),indent=2))
    sys.argv=[str(driver),'--root',str(root),'--base',str(base),*rest];runpy.run_path(str(driver),run_name='__main__')
    result=json.loads((root/'result.json').read_text());assert all(s['exit_code']==0 for s in result['stages'])
    assert not (root/'outflow/ti60f225_oob.troutingtraces').exists()
    from fixed_cnn_ram_locations import verify_placed
    (root/'cnn_ram_actual_locations.json').write_text(json.dumps(verify_placed(root),indent=2))
    from audit_global_place import metrics
    current=metrics(root)
    ref=ROOT/'tmp/frame-execution/command-progress-seed3-1/pnr-seed1';previous=metrics(ref)
    from review_existing_routability import graph_review
    graph=graph_review(root)
    review=dict(status='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW',current=current,reference=previous,
        changes_pct=dict(weighted_hpwl=100*(current['weighted_hpwl_per_net']/previous['weighted_hpwl_per_net']-1),
            **{k:100*(current['congestion_peaks'][k]/v-1) for k,v in previous['congestion_peaks'].items()}),
        graph=graph,reference_path=str(ref),route_started=False,hardware_access=False,board_release=False)
    (root/'placement_review_inputs.json').write_text(json.dumps(review,indent=2))
    print(json.dumps({k:review[k] for k in ('status','changes_pct')}),flush=True)
if __name__=='__main__':main()
