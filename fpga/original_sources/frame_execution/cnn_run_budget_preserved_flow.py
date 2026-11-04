"""Bounded sparse-route candidate, unchanged canonical STA/CDC gates."""
import json,runpy,sys
from pathlib import Path
from run_units import sha
from control_command_repair import once
from cnn_run_budget_convergence_flow import prepare_code as old_prepare,route_code as old_route

def prepare_code():
    source,code=old_prepare()
    code=once(code,"    audit_delta(board);sr=", "    from cnn_run_budget_preserved_locations import verify_placed\n    assert verify_placed(source)['nets']==82\n    audit_delta(board);sr=")
    code=once(code,"'ti60f225_oob.peri.xml','efx_pnr_settings.ini')}",
        "'ti60f225_oob.peri.xml','efx_pnr_settings.ini','preserved_parent.troutingtraces','preserved_parent.placer_keepout','preserved_data.rcf','preserved_route_placement.json','preserved_route_generator.py','preserved_route_actual_locations.json')}")
    code=once(code,"    review['required_evidence']=evidence", "    assert verify_placed(root)['nets']==82\n    for name in geometry:evidence[str(root/name)]=sha(root/name)\n    review['required_evidence']=evidence")
    compile(code,'preserved-prepare','exec');return source,code

def route_code():
    source,code=old_route()
    code=once(code,"    required=review['required_evidence'];assert required",
        "    from cnn_run_budget_preserved_locations import verify_placed\n    assert verify_placed(physical)['nets']==82\n    required=review['required_evidence'];assert required")
    code=once(code,"        import cnn_run_budget_implementation as impl", "        assert verify_placed(physical)['nets']==82\n        import cnn_run_budget_implementation as impl")
    compile(code,'preserved-route','exec');return source,code

def main():
    action=sys.argv.pop(1)
    if action=='--check-only':
        prepare_code();route_code();print('PRESERVED_ROUTE_FULL_GATE_GENERATORS_COMPILE');return
    if action=='prepare':
        source,code=prepare_code();review=Path(sys.argv[sys.argv.index('--review')+1]).resolve()
        root=Path(json.loads(review.read_text())['physical']);suffix='-prepare-cnn-budget'
    else:
        assert action=='route';source,code=route_code();root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();suffix='-local-driver-route'
    driver=root.parent/(root.name+suffix+'.py');assert not driver.exists()
    driver.write_text(code,encoding='utf8');driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),driver_sha256=sha(driver),adapter_sha256=sha(__file__),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
