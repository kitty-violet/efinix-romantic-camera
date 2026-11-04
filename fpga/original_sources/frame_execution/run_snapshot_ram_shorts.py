"""Official mixed-width RAM model and unchanged OFP5 host bridge short tests."""
import argparse,json,os,re,shutil,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
MS=Path('E:/quatra/questa_fse/win64')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 sources=[HERE/'variants/fixed_snapshot_ram/jtag_bringup_snapshot.sv',HERE/'tb_snapshot_ram.sv',EF/'sim_models/verilog/efx_ram10.v']
 inputs=[]
 for src in sources:
  dst=root/src.name;shutil.copy2(src,dst);inputs.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
 manifest=dict(inputs=inputs,runner_sha256=sha(__file__),scope='OFP5 snapshot isolated test; no board access')
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log',60)
  run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in inputs]],root,'compile.log',60)
  results=[]
  for seed in (97,391,1027):
   body,seconds=run([MS/'vsim.exe','-c','work.tb_snapshot_ram','-gSEED='+str(seed),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log',160)
   lines=re.findall(r'^# (PASS SNAPSHOT_RAM .*)$',body,re.M);assert len(lines)==1
   results.append(dict(seed=seed,seconds=seconds,line=lines[0],log_sha256=sha(root/f'seed{seed}.log')))
  for row in inputs:assert sha(row['source'])==sha(row['snapshot'])==row['sha256']
  result=dict(status='OFP5_RAM_SNAPSHOT_ISOLATED_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),pnr=False,hardware_access=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as exc:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
