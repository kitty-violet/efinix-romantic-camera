"""Four-state full pipeline A/B and exhaustive17-bit fractional edge checks."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,sha,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--overrides-dir',type=Path);p.add_argument('--dsp-model',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  files=[HERE/'variants/compact_quant/requantize_i32_mul18_q17_pipeline.sv',HERE/'variants/narrow_round/tb_narrow_round.sv']
  if a.overrides_dir:
   override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
   files=[override/f.name if (override/f.name).is_file() else f for f in files]
  old=HERE.parent/'pw_array_feed_compact/requantize_i32_mul18_q17_pipeline.sv'
  if a.dsp_model:files.append(EF/'sim_models/verilog/efx_dsp48.v')
  for f in files:shutil.copy2(f,root/f.name)
  (root/'reference_pipeline.sv').write_text(old.read_text().replace('module requantize_i32_mul18_q17_pipeline','module reference_pipeline'))
  manifest={str(f):sha(f) for f in [*files,old,Path(__file__)]}
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  models=['efx_dsp48.v'] if a.dsp_model else []
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*models,'requantize_i32_mul18_q17_pipeline.sv','reference_pipeline.sv','tb_narrow_round.sv'],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_narrow_round','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
  lines=re.findall(r'^# (PASS NARROW_ROUND.*)$',body,re.M);assert len(lines)==1
  for f,h in manifest.items():assert sha(f)==h
  result=dict(status='COMPACT_QUANT_FOUR_STATE_EQUIVALENCE_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),same_four_edge_latency=True,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
