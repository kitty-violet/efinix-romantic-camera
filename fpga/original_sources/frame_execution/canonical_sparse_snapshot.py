"""Original strict STA with complete sparse-RCF collateral copied verbatim."""
import json,runpy,shutil,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once

def main():
    base=Path(sys.argv[sys.argv.index('--base')+1]).resolve()
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    from audit_camera_sparse_route import audit
    evidence={}
    def verify(rows):
        for f,h in rows.items():assert sha(f)==h,f
        evidence.update(rows)
    def bind(p):verify({str(p):sha(p)});return json.loads(p.read_text())
    audit(base,bind,verify)
    source=HERE/'canonical_snapshot_sta.py';code=source.read_text()
    code=once(code,"('.xml','.sdc','.ini','.placeloc')","('.xml','.sdc','.ini','.placeloc','.rcf','.troutingtraces')")
    code=once(code,"frozen[str(canonical)]=sha(canonical)",
        "frozen.update("+repr(evidence)+")\nfrozen[str(canonical)]=sha(canonical)")
    driver=root.parent/(root.name+'-sparse-canonical.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    policy=driver.with_name('timing_acceptance.py')
    if policy.exists():assert sha(policy)==sha(HERE/'timing_acceptance.py')
    else:shutil.copy2(HERE/'timing_acceptance.py',policy)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        adapter_sha256=sha(__file__),generated_sha256=sha(driver),sparse_evidence=evidence,hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
