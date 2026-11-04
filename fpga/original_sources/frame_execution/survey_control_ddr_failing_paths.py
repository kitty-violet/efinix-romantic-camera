"""Read-only expanded STA on an already routed design, to group all failures."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_ddr_combined import once

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'canonical_snapshot_sta.py'
    code=once(source.read_text(),'report_timing -setup -npaths 5 -nworst 1','report_timing -setup -npaths 1000 -nworst 1')
    code=once(code,"Path(__file__).with_name('timing_acceptance.py')",repr(str(HERE/'timing_acceptance.py')))
    code=code.replace(repr(str(HERE/'timing_acceptance.py'))+'.read_bytes()',"Path("+repr(str(HERE/'timing_acceptance.py'))+").read_bytes()")
    driver=root.parent/(root.name+'-runner.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__),change='Setup report 5 -> 1000 paths only; no geometry or constraints changed'),indent=2))
    sys.argv=[str(driver),'--base',str(a.base.resolve()),'--root',str(root)];runpy.run_path(str(driver),run_name='__main__')
