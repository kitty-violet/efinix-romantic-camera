"""Actual dual RAM cycle A/B for flattened four-source plus DMA payload selection."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
from frozen_sources import frozen_paths

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir();records=[]
 variant=HERE/'variants/fused_write_payload'
 try:
  original,arena=frozen_paths([HERE/'arena_client_router2.sv',HERE/'tensor_arena12.sv'],a.base)
  for src in [EF/'sim_models/verilog/efx_dpram10.v',variant/'tb_write_payload_route.sv',variant/'arena_client_router2.sv',arena,original]:
   body=src.read_text();dst=root/src.name
   if src==original:
    body=body.replace('module arena_client_router2 #','module arena_client_router2_ref #',1);dst=root/'arena_client_router2_ref.sv'
   dst.write_text(body);records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
  body=arena.read_text()
  for m in ['tensor_arena12','tensor_dual_bank']:body=re.sub(r'\b'+m+r'\b',m+'_ref',body)
  dst=root/'tensor_arena12_ref.sv';dst.write_text(body)
  records.append(dict(source=str(arena),source_sha256=sha(arena),snapshot=str(dst),sha256=sha(dst)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  runs=[]
  for mode in (0,1):
   for seed in (71,391,1027):
    log=f'mode{mode}_seed{seed}.log'
    body,secs=run([MS/'vsim.exe','-c','work.tb_write_payload_route',f'-gSEED={seed}',f'-gPAIR_MODE={mode}','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,log,timeout=120)
    lines=re.findall(r'^# (PASS FUSED_PAYLOAD_ROUTE.*)$',body,re.M)
    assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body and not re.search(r'\bFAIL\b|\*\* (Error|Fatal)',body)
    runs.append(dict(mode=mode,seed=seed,line=lines[0],seconds=secs,log_sha256=sha(root/log)))
  for r in records:assert sha(r['source'])==r['source_sha256'] and sha(r['snapshot'])==r['sha256']
  result=dict(status='FUSED_PAYLOAD_ROUTE_FOUR_STATE_PASS',runs=runs,dual_real_ram=True,inputs_sha256=sha(root/'inputs.json'),full_frame=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
