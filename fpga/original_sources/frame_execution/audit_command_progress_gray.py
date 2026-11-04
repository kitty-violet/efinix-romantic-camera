"""Reuse routed Gray/observation checks with zero extra optimization margin."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 base=a.base.resolve();root=a.root.resolve();meta=json.loads((base/'physical_inputs.json').read_text())
 assert meta['setup_margin'] is None and root.is_relative_to(ROOT/'tmp') and not root.exists()
 source=HERE/'audit_snapshot_routed_gray.py';code=source.read_text()
 before="canonical=b/'canonical_timing.sdc'\nassert sha(canonical)==meta['setup_margin']['canonical_sdc_sha256']"
 after="canonical=b/'ti60f225_oob.sdc'\nassert meta['setup_margin'] is None\nassert sha(canonical)==next(item['sha256'] for item in meta['inputs'] if Path(item['snapshot'])==canonical)"
 assert code.count(before)==1;code=code.replace(before,after)
 # Referenced Tcl/check scripts stay in their existing project directory.
 code=code.replace("here=Path(__file__).parent","here=Path("+repr(str(HERE))+")")
 driver=root.parent/(root.name+'-runner.py');assert not driver.exists();driver.write_text(code)
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),wrapper_sha256=sha(__file__),acceptance_constraints_unchanged=True),indent=2))
 sys.argv=[str(driver),'--base',str(base),'--root',str(root)];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
