"""Four-state timing/reset checks of the small public PLL interface adapter."""
import argparse,json,os,shutil,time,re
from pathlib import Path
from run_units import ROOT,HERE,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 files=['ddr516_pll_shift_guard.sv','tb_ddr516_shift_min_gap.sv']
 for f in files:shutil.copy2(HERE/f,root/f)
 hashes={f:sha(root/f) for f in files};start=time.monotonic()
 sim=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 try:
  run([sim/'vlib.exe','work'],root,'vlib.log',30)
  run([sim/'vlog.exe','-sv',*files],root,'compile.log',30)
  body,_=run([sim/'vsim.exe','-c','work.tb_ddr516_shift_min_gap','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',30)
  match=re.search(r'PASS PLL_SHIFT_MIN_GAP pulses=64 gaps=63 low_ns=([0-9.]+) high_ns=([0-9.]+)',body);assert match
  low,high=map(float,match.groups());assert low>=13.331 and high>=33.329
  for f,digest in hashes.items():assert sha(root/f)==sha(HERE/f)==digest
  result=dict(status='PLL_SHIFT_MINIMUM_GAP_PASS',low_ns=low,high_ns=high,inputs=hashes,log_sha256=sha(root/'simulation.log'),seconds=time.monotonic()-start,hardware_access=False,board_release=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e)),indent=2));raise

if __name__=='__main__':main()
