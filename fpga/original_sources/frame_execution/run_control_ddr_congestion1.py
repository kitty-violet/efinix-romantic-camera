"""Independent congestion-recipe-1 comparison on the exact same mapped design."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,required=True)
    args,unused=parser.parse_known_args()
    root=args.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    source=HERE/'run_control_ddr_congestion.py';code=source.read_text()
    assert code.count('CONGESTION_3')==3
    code=code.replace('CONGESTION_3','CONGESTION_1')
    driver=root.parent/(root.name+'-recipe-dispatch.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),adapter_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
