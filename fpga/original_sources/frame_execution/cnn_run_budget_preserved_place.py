"""Sparse route preservation with a mandatory placement-review stop."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
from control_command_repair import once
from camera_factor_island_flow import driver_code as previous

def driver_code():
    source,code=previous()
    code=once(code,'from camera_factor_balanced_board import BOARD,audit_delta','from cnn_run_budget_board import BOARD,audit_delta')
    code=code.replace('camera_factor_island_locations','cnn_run_budget_locations')
    needle="'arena_island=retain_arena_island(root)\\n' +"
    code=once(code,needle,needle+"\n        'from cnn_run_budget_preserved_locations import generate as preserve_routes\\n' +\n        'preserved_routes=preserve_routes(root)\\n' +")
    code=once(code,"    from fixed_cnn_ram_locations import verify_placed",
        "    from cnn_run_budget_preserved_locations import verify_placed as verify_preserved\n    (root/'preserved_route_actual_locations.json').write_text(json.dumps(verify_preserved(root),indent=2))\n    from fixed_cnn_ram_locations import verify_placed")
    code=once(code,"    driver=root.parent/(root.name+'-place-driver.py');",
        "    code=once(code,'clock_profile=clock_profile,','preserved_route_placement_sha256=sha(root/\"preserved_route_placement.json\"),clock_profile=clock_profile,')\n    driver=root.parent/(root.name+'-place-driver.py');")
    code=once(code,'all remaining ordinary cells free.','Sparse 82 unchanged ordinary data routes with complete shared tiles/cascades; other ordinary cells free.')
    compile(code,'cnn-budget-preserved-place','exec');return source,code

def main():
    source,code=driver_code()
    if sys.argv[1:]==['--check-only']:
        from cnn_run_budget_preserved_locations import plan
        r=plan(ROOT/'tmp/frame-execution/cnn-run-budget-place2')
        print(json.dumps({k:r[k] for k in ('status','selected_nets','ordinary_cells_retained','ordinary_cells_free')}));return
    assert sys.argv.pop(1)=='place'
    result=ROOT/'tmp/frame-execution/cnn-run-budget-convergence-route1/selected.json'
    failed=json.loads(result.read_text());assert not failed['current_timing_pass'] and failed['wns_ns']<0
    assert (result.parent/'failure.json').exists()
    assert all(x['status']!='running' for x in json.loads((result.parent/'stages.json').read_text()))
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    driver=root.parent/(root.name+'-place-wrapper.py');assert not driver.exists()
    driver.write_text(code,encoding='utf8');driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
