"""Four-state private schedule sharing against frozen independent MAC fronts."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--reference-board',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 base=a.reference_board.resolve();records=[]
 source=json.loads((base/'inputs.json').read_text())
 def frozen(name):
  rows=[x for x in source['inputs'] if Path(x['snapshot']).name==name];assert len(rows)==1
  row=rows[0];assert sha(row['snapshot'])==row['sha256'];return Path(row['snapshot'])
 paths=[EF/'sim_models/verilog/efx_dsp24.v',frozen('int8_dual_mac_lane_dspacc.sv'),
  frozen('int8_dual_mac_lane_prefix_stream.sv'),HERE/'variants/paired_context/shared_mac_stream_front.sv',HERE/'variants/paired_context/tb_paired_context.sv']
 try:
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst)))
  ref=frozen('shared_mac_stream_front.sv');dst=root/'shared_mac_stream_front_ref.sv'
  dst.write_text(ref.read_text().replace('module shared_mac_stream_front #','module shared_mac_stream_front_ref #'))
  records.append(dict(source=str(ref),source_sha256=sha(ref),snapshot=str(dst),sha256=sha(dst),transform='rename module only'))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in (71,391,1027):
   name=f'seed{seed}.log'
   body,seconds=run([MS/'vsim.exe','-c','work.tb_paired_context',f'-gSEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,name,timeout=120)
   lines=re.findall(r'^# (PASS PAIRED_CONTEXT.*)$',body,re.M)
   assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(root/name)))
  for x in records:assert sha(x['snapshot'])==x['sha256']
  result=dict(status='PAIRED_CONTEXT_FOUR_STATE_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),whole_board=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
