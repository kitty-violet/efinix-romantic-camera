"""Adapt the existing physical runner only for the bound PLL wrapper repair."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--cnn-ram-reference',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a,rest=p.parse_known_args()
 root=a.root.resolve();reference=a.cnn_ram_reference.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 original=HERE/'probe_fourpll_reset_physical.py';source=original.read_text()
 anchor='from fourpll_reset_clock_profile import apply_profile';assert source.count(anchor)==1
 source=source.replace(anchor,'from shift_guard_clock_profile import apply_profile')
 anchor='related_gray_layout=None\n';assert source.count(anchor)==1
 addition=('from fixed_cnn_ram_locations import generate as retain_cnn_ram\nassert a.fixed_dsp_layout and not a.related_gray_placement_reference\n'+f'cnn_ram_layout=retain_cnn_ram(root,Path({str(reference)!r}))\n'+"dsp_layout['assignments_sha256']=sha(root/'fixed_dsp.placeloc')\ndsp_layout['cnn_ram_placement_sha256']=sha(root/'cnn_ram_placement.json')\n")
 source=source.replace(anchor,addition+anchor)
 driver=root.parent/(root.name+'-ram-probe.py');assert not driver.exists();driver.parent.mkdir(parents=True,exist_ok=True);driver.write_text(source)
 driver.with_suffix('.json').write_text(json.dumps(dict(original_runner=str(original),original_sha256=sha(original),generated_runner=str(driver),generated_sha256=sha(driver),wrapper_sha256=sha(__file__),clock_profile_sha256=sha(HERE/'shift_guard_clock_profile.py'),change='Exact new PLL shift wrapper evidence; same clock constraints and 113 CNN RAM locations',hardware_access=False),indent=2))
 sys.argv=[str(driver),'--root',str(root),*rest];runpy.run_path(str(driver),run_name='__main__')
 from fixed_cnn_ram_locations import verify_placed
 (root/'cnn_ram_actual_locations.json').write_text(json.dumps(verify_placed(root),indent=2))
if __name__=='__main__':main()
