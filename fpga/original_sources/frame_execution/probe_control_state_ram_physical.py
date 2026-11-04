"""Independent physical probe: old tested runner plus audited CNN RAM locations."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from run_units import ROOT, HERE, sha

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cnn-ram-reference',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args,rest=parser.parse_known_args()
    root=args.root.resolve();reference=args.cnn_ram_reference.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    original=HERE/'probe_fourpll_reset_physical.py'
    source=original.read_text()
    anchor="related_gray_layout=None\n"
    assert source.count(anchor)==1
    addition=("from fixed_cnn_ram_locations import generate as retain_cnn_ram\n"
              "assert a.fixed_dsp_layout and not a.related_gray_placement_reference\n"
              f"cnn_ram_layout=retain_cnn_ram(root,Path({str(reference)!r}))\n"
              "dsp_layout['assignments_sha256']=sha(root/'fixed_dsp.placeloc')\n"
              "dsp_layout['cnn_ram_placement_sha256']=sha(root/'cnn_ram_placement.json')\n")
    source=source.replace(anchor,addition+anchor)
    driver=root.parent/(root.name+'-ram-probe.py')
    assert not driver.exists();driver.parent.mkdir(parents=True,exist_ok=True)
    driver.write_text(source,encoding='utf-8')
    receipt=driver.with_suffix('.json')
    receipt.write_text(json.dumps(dict(original_runner=str(original),original_sha256=sha(original),
        generated_runner=str(driver),generated_sha256=sha(driver),wrapper_sha256=sha(__file__),
        change='Add 113 audited unchanged CNN RAM physical locations before interface/P&R',
        hardware_access=False),indent=2))
    sys.argv=[str(driver),'--root',str(root),*rest]
    runpy.run_path(str(driver),run_name='__main__')
    from fixed_cnn_ram_locations import verify_placed
    result=verify_placed(root)
    (root/'cnn_ram_actual_locations.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
