"""Use exact passing parent CNN RAM geometry; reuse the same mapped short proof."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_canonical_multiseed import audit
from fixed_cnn_ram_locations import memory_cells
from first_fault_observer import audit_delta

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--startup-proof',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();board=a.board.resolve();root=a.root.resolve();proof=a.startup_proof.resolve();reference=a.reference.resolve()
    audit_delta(board);prior=audit(reference);assert prior['current_timing_pass']
    assert memory_cells(board/'outflow/ti60f225_oob.map.v')==memory_cells(Path(prior['physical'])/'outflow/ti60f225_oob.map.v')
    result=json.loads((proof/'result.json').read_text());inputs=json.loads((proof/'inputs.json').read_text())
    assert result['status']=='FIRST_FAULT_MAP_STARTUP_AND_CAPTURE_PASS' and result['inputs_sha256']==sha(proof/'inputs.json')
    assert inputs['original_map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert result['simulation_sha256']==sha(proof/'simulation.log')
    for n,h in inputs['files'].items():assert sha(proof/n)==h
    source=HERE/'run_first_fault_physical.py';code=source.read_text()
    old="        run('startup','validate_first_fault_mapped.py',['--board',board,'--root',root/'startup'],180)"
    new="""        import shutil
        proof=Path(@@PROOF@@)
        pm=json.loads((proof/'inputs.json').read_text());pr=json.loads((proof/'result.json').read_text())
        assert pr['inputs_sha256']==sha(proof/'inputs.json') and pm['original_map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
        destination=root/'startup';destination.mkdir()
        for name in set(pm['files'])|{'result.json','inputs.json','simulation.log'}:shutil.copy2(proof/name,destination/name)
        for name,h in pm['files'].items():assert sha(destination/name)==h
        (root/'reused_startup.json').write_text(json.dumps(dict(root=str(proof),result_sha256=sha(proof/'result.json'),map_sha256=pm['original_map_sha256']),indent=2))""".replace('@@PROOF@@',repr(str(proof)))
    assert code.count(old)==1;code=code.replace(old,new)
    old="run('physical','probe_fourpll_reset_physical.py',['--base',board,'--root',physical,'--pnr-budget',1500,"
    new="run('physical','probe_control_state_ram_physical.py',['--cnn-ram-reference',Path("+repr(str(reference))+"),'--base',board,'--root',physical,'--pnr-budget',1500,"
    assert code.count(old)==1;code=code.replace(old,new)
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    driver=root.parent/(root.name+'-reference-runner.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf-8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),
        wrapper_sha256=sha(__file__),reference=str(reference),reference_result_sha256=sha(reference/'result.json'),
        reuse_startup=str(proof),reuse_result_sha256=sha(proof/'result.json'),hardware_access=False),indent=2))
    sys.argv=[str(driver),'--board',str(board),'--root',str(root),'--seed','1'];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
