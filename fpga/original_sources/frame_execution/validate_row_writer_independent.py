"""Four-state targeted terminal RQ queue/writer isolation, cancel and exact data."""
import argparse,json,re,shutil,traceback
from run_units import ROOT,HERE,EF,MS,sha,run
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 variant=HERE/'variants/fixed_row_writer';records=[]
 for src in [EF/'sim_models/verilog/efx_dsp48.v',EF/'sim_models/verilog/efx_srl8.v',*[variant/n for n in ('fixed_rq32_mixed.sv','fixed_column_store4.sv','tests/tb_row_writer_independent.sv')]]:
  dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=60)
  lines=re.findall(r'^# (PASS ROW_WRITER_INDEPENDENT.*)$',body,re.M);assert len(lines)==1
  result=dict(status='ROW_WRITER_INDEPENDENT_FOUR_STATE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'))
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception:
  (root/'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
