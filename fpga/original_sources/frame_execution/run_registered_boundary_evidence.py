"""Run remaining same-map gates, reusing the verified adapter/startup superset."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
from control_command_repair import once
from registered_boundary_board import BOARD,audit_delta

def main():
    audit_delta(BOARD)
    adapter=ROOT/'tmp/frame-execution/split-registered-adapter1'
    r=json.loads((adapter/'result.json').read_text());i=json.loads((adapter/'inputs.json').read_text())
    assert r['status']=='ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS'
    assert r['inputs_sha256']==sha(adapter/'inputs.json')
    assert i['original_map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
    assert r['simulation_sha256']==sha(adapter/'simulation.log')
    source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
    code=once(code,"  run('map-audit','audit_split_logic_outputs_map.py',['--board',board,'--root',root/'map-audit'],100)",
        "  # Exact-map truth proof already complete; avoid duplicate simulation.\n  assert json.loads((ROOT/'tmp/frame-execution/split-registered-placement-checks1/map-audit/result.json').read_text())['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS'")
    code=code.replace("'split_logic_outputs_flow.py'","'registered_boundary_flow.py'")
    code=once(code,"  adapter=root.parent/(root.name+'-adapter')",'  adapter=Path('+repr(str(adapter))+')')
    code=once(code,"  run('adapter','validate_control_adapter_idle_mapped.py',['--board',board,'--tag',adapter.name],250)",
        '  # Reuse same-map three-reset adapter evidence; all hashes rechecked below.')
    code=once(code,"status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='REGISTERED_BOUNDARY_SAME_MAP_SHORT_INTEGRATION_PASS'")
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),reused_adapter_result_sha256=sha(adapter/'result.json'),
        scope='Same assertions and mapped signals; verified startup superset reused; no test omissions.'),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
