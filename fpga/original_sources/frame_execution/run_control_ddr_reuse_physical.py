"""Reuse proven functional gates, implement the corrected physical partition."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
from control_ddr_combined import once

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True);a,rest=p.parse_known_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'run_control_ddr_physical.py';code=source.read_text()
    code=once(code,"'probe_control_ddr_physical.py'","'probe_control_ddr_reuse_physical.py'")
    anchor="source=HERE/'run_control_command_physical.py';code=source.read_text()"
    code=once(code,anchor,anchor+"\n    code=once(code,\"'--pnr-budget',1500\",\"'--pnr-budget',2100\")\n    code=once(code,\"inventory.json'],1950)\",\"inventory.json'],2550)\")")
    driver=root.parent/(root.name+'-reuse-dispatch.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),
        wrapper_sha256=sha(__file__),probe_sha256=sha(HERE/'probe_control_ddr_reuse_physical.py'),placement_sha256=sha(HERE/'fixed_control_ddr_reuse_locations.py'),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
