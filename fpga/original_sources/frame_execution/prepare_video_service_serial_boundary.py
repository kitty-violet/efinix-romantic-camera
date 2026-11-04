"""Bound a transport test with a directed near-wrap counter state.

Do not call this 272 natural completions: it makes 22 real requests and seeds
the unchanged 8-bit sequence counter to250 while both transports are idle.
All other state, full1024-bit scans, pauses, abort and stopped-clock tests stay.
"""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 src=a.source.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 meta=json.loads((src/'inputs.json').read_text())
 for f,h in meta['files'].items():assert sha(f)==h,f
 root.mkdir()
 for f in src.iterdir():
  if f.suffix in ('.v','.sv'):shutil.copy2(f,root/f.name)
 tb=root/'tb.sv';s=tb.read_text()
 needle='  for(integer j=0;j<270;j=j+1)begin';assert s.count(needle)==1
 seed='  // Directed counter-wrap state; all handshakes remain idle/even.\n'
 for dut in ('old_dut','new_dut'):
  for i in range(1,8):seed+=f"  {dut}.\\cpu/system_hardJtag_debug_logic_jtagBridge/observation/sequence_number[{i}]~FF .Q=1'b{250>>i&1};\n"
 seed+='  #1;if(old_sequence!==250 || new_sequence!==250 || old_busy || new_busy)$fatal(1,"bad directed wrap state");\n'
 s=s.replace(needle,seed+'  for(integer j=0;j<20;j=j+1)begin')
 s=s.replace('expected 272 completions with wrap','expected22 completions after directed seed250')
 s=s.replace('completed=272 sequence_wrap=1','completed=22 directed_sequence_seed=250 sequence_wrap=1')
 s=s.replace('   randomize();settled();start(0);', '   $display("SERIAL_PROGRESS request=%0d",j);randomize();settled();start(0);')
 tb.write_text(s)
 files={str(f):sha(f) for f in root.iterdir()};files[str(Path(__file__))]=sha(__file__)
 (root/'inputs.json').write_text(json.dumps(dict(files=files,parent_inputs=str(src/'inputs.json'),parent_inputs_sha256=sha(src/'inputs.json'),stimulus_change='22actual requests; counter seeded250 after initial idle scan; no DUT source change',hardware_access=False),indent=2))
 print('SERIAL_BOUNDARY_FIXTURE_PREPARED',root)
if __name__=='__main__':main()
