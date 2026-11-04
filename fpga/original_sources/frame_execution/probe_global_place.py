"""Stop officially after pack/place; do not spend router time before inspection."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a,rest=p.parse_known_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    original=HERE/'probe_fourpll_reset_physical.py';source=original.read_text()
    before="+' --timeout '+str(budget)+'\\nexit /b %errorlevel%\\n')"
    after="+(' --pnr_opts pack place' if flow=='pnr' else '')+' --timeout '+str(budget)+'\\nexit /b %errorlevel%\\n')"
    assert source.count(before)==1;source=source.replace(before,after)
    source=source.replace("status='PHYSICAL_CAPACITY_PROBE_FINISHED'","status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'")
    assert "status='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'" in source
    # A successful placement-only tool invocation is NOT routed STA evidence.
    source=source.replace('stages.append(dict(flow=flow,','stages.append(dict(flow=("pack_place" if flow=="pnr" else flow),')
    source=source.replace("flows=['interface','pnr']","flows=['interface','pack_place']")
    driver=root.parent/(root.name+'-place-probe.py')
    assert not driver.exists();driver.parent.mkdir(parents=True,exist_ok=True)
    driver.write_text(source,encoding='utf-8')
    driver.with_suffix('.json').write_text(json.dumps(dict(original_runner=str(original),
        original_sha256=sha(original),generated_sha256=sha(driver),wrapper_sha256=sha(__file__),
        change='Official efx_run --pnr_opts pack place; explicitly no routed PASS',hardware_access=False),indent=2))
    sys.argv=[str(driver),'--root',str(root),*rest];runpy.run_path(str(driver),run_name='__main__')
    assert not (root/'outflow/ti60f225_oob.troutingtraces').exists(),'Unexpected routing'
    from audit_global_place import audit
    result=audit(root,a.reference.resolve()) if a.reference else audit(root)
    (root/'placement_gate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
