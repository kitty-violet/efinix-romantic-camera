"""Change the documented placement/routing recipe, preserving all RTL proofs."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from control_ddr_combined import once

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,required=True)
    args,unused=parser.parse_known_args()
    root=args.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'run_control_ddr_physical.py'
    code=once(source.read_text(),"source=HERE/'run_control_command_physical.py';code=source.read_text()",
        "source=HERE/'run_control_command_physical.py';code=source.read_text()\n    code=once(code,\"'CONGESTION_2'\",\"'CONGESTION_3'\")")
    driver=root.parent/(root.name+'-recipe-runner.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    doc=EF/'doc/topics/tim-pnr.html';assert 'CONGESTION_3' in doc.read_text()
    evidence={str(p):sha(p) for p in (source,driver,doc,Path(__file__))}
    driver.with_suffix('.json').write_text(json.dumps(dict(
        change='CONGESTION_2 -> CONGESTION_3 only; same mapped design, placement anchors, seed, clocks and original SDC',
        required_evidence=evidence,hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
