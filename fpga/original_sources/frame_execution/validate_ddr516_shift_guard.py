"""Four-state timing/reset checks of the small public PLL interface adapter."""
import argparse,json,os,shutil,time
from pathlib import Path
from run_units import ROOT,HERE,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 files=['ddr516_pll_shift_guard.sv','tb_ddr516_pll_shift_guard.sv']
 for f in files:shutil.copy2(HERE/f,root/f)
 hashes={f:sha(root/f) for f in files};start=time.monotonic()
 sim=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 try:
  run([sim/'vlib.exe','work'],root,'vlib.log',30)
  run([sim/'vlog.exe','-sv',*files],root,'compile.log',30)
  body,_=run([sim/'vsim.exe','-c','work.tb_ddr516_pll_shift_guard','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',30)
  assert 'PASS PLL_SHIFT_GUARD complete=37' in body
  for f,digest in hashes.items():assert sha(root/f)==sha(HERE/f)==digest
  result=dict(status='PLL_SHIFT_GUARD_FOURSTATE_PASS',inputs=hashes,log_sha256=sha(root/'simulation.log'),seconds=time.monotonic()-start,hardware_access=False,board_release=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e)),indent=2));raise

if __name__=='__main__':main()
