"""Real RAM timing/numerics for the bounded two-group parameter cache."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',type=Path);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 paths=[EF/'sim_models/verilog/efx_ram10.v',HERE/'variants/header_ram/header_group_cache3.sv',HERE/'variants/header_ram/tb_header_group_cache3.sv']
 if a.variant:
  variant=a.variant.resolve();assert variant.is_relative_to(HERE/'variants')
  paths=[paths[0],variant/'header_group_cache3.sv',variant/'tb_header_group_cache3.sv']
 records=[]
 for f in paths:
  dst=root/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_header_group_cache3','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=120)
  assert 'Errors: 0, Warnings: 0' in body
  lines=re.findall(r'^# (PASS HEADER_GROUP_CACHE .*)$',body,re.M);assert len(lines)==1
  result=dict(status='HEADER_GROUP_CACHE_REAL_RAM_FOUR_STATE_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),board_integrated=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
