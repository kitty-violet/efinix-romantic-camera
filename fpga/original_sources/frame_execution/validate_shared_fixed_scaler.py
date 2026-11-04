"""Exact fixed-ratio scaler sharing: exhaustive arithmetic and cycle A/B."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
base=ROOT/'tmp/frame-execution/connected-board-header-ram1/rtl'
variant=HERE/'variants/shared_fixed_scaler';records=[]
try:
 for source in [base/'rgb_bilinear_scaler_prototype.sv',variant/'rgb_bilinear_scaler_prototype.sv',variant/'tb_shared_fixed_scaler.sv',base/'resize_axis_up_2_to_3.sv',base/'resize_axis_down_3_to_2.sv']:
  original=source==base/'rgb_bilinear_scaler_prototype.sv'
  dst=root/('reference.sv' if original else source.name)
  if original:dst.write_text(source.read_text().replace('module rgb_bilinear_scaler_prototype','module rgb_bilinear_scaler_reference'))
  else:shutil.copy2(source,dst)
  records.append(dict(path=str(source),source_sha256=sha(source),snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
 body,seconds=run([MS/'vsim.exe','-c','work.tb_shared_fixed_scaler','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
 lines=re.findall(r'^# (PASS SHARED_FIXED_SCALER.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 for r in records:assert sha(r['path'])==r['source_sha256'] and sha(r['snapshot'])==r['sha256']
 result=dict(status='FIXED_SCALER_EXHAUSTIVE_AND_CYCLE_AB_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),board_test=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
