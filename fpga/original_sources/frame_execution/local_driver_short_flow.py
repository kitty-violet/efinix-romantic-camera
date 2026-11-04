"""Short-gate wrapper preserving arguments across legacy runpy entrypoints."""
import json,runpy,sys,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from local_driver_board import audit_delta
from control_command_repair import once

def main():
    action=sys.argv.pop(1)
    if action=='audit':
        # The legacy entrypoint changes sys.argv while running its sub-gates.
        argv=list(sys.argv)
        board=Path(argv[argv.index('--board')+1]).resolve();root=Path(argv[argv.index('--root')+1]).resolve()
        completed=ROOT/'tmp/frame-execution/native-local-driver-evidence1/map-audit'
        if (completed/'result.json').exists():
            # First attempt completed the real gate before wrapper failure.
            # Reuse that exact-map evidence; do not rerun a write-once audit.
            audit_delta(board);r=json.loads((completed/'result.json').read_text())
            assert r['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS' and r['checks']==17504
            assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['board_inputs_sha256']==sha(board/'inputs.json')
            for f,h in r['evidence'].items():assert sha(f)==h,f
            assert not root.exists();shutil.copytree(completed,root)
            r['reused_from']=str(completed)
            r['evidence'][str(completed/'result.json')]=sha(completed/'result.json')
            r['evidence'][str(Path(__file__))]=sha(__file__)
            (root/'result.json').write_text(json.dumps(r,indent=2))
        else:
            import audit_fault_probe_map as prior
            prior.audit_delta=audit_delta
            try:prior.main()
            finally:sys.argv=argv
        from audit_native_split_map import extra_map
        from native_split_cdc import map_structure
        (root/'new_diagnostic_cells.json').write_text(json.dumps(extra_map(board),indent=2))
        (root/'held_data_structure.json').write_text(json.dumps(map_structure(board),indent=2))
    elif action=='flow':
        import local_driver_flow as prior
        sys.argv.insert(1,'flow');prior.main()
    elif action=='shorts':
        source=HERE/'run_split_logic_outputs_evidence.py';code=source.read_text()
        code=once(code,"'audit_split_logic_outputs_map.py',['--board'","'local_driver_short_flow.py',['audit','--board'")
        code=code.replace("'split_logic_outputs_flow.py',['", "'local_driver_short_flow.py',['flow','")
        assert 'split_logic_outputs_flow.py' not in code
        code=once(code,"status='SPLIT_OUTPUTS_SAME_MAP_SHORT_INTEGRATION_PASS'","status='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'")
        root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();driver=root.parent/(root.name+'-short-wrapper.py')
        assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
        driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
            wrapper_sha256=sha(__file__),driver_sha256=sha(driver),hardware_access=False),indent=2))
        sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
    else:raise ValueError(action)

if __name__=='__main__':main()
