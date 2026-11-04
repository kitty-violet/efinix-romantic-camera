"""Use the same test gates with one explicit register/driver placement delta."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_ddr_combined import once

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);a,rest=p.parse_known_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'run_control_ddr_reuse_physical.py';code=source.read_text()
    code=code.replace('probe_control_ddr_reuse_physical.py','probe_control_ddr_control_physical.py')
    code=code.replace('fixed_control_ddr_reuse_locations.py','fixed_control_ddr_control_locations.py')
    wrapper=root.parent/(root.name+'-control-dispatch-wrapper.py');assert not wrapper.exists()
    compile(code,str(wrapper),'exec');wrapper.write_text(code)
    wrapper.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(wrapper),adapter_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(wrapper),*sys.argv[1:]];runpy.run_path(str(wrapper),run_name='__main__')
if __name__=='__main__':main()
