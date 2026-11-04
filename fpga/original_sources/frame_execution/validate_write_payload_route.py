"""Four-state connected router + real dual-port RAM comparison."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference-board',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 meta=json.loads((a.reference_board/'inputs.json').read_text());records=[]
 try:
  paths=[EF/'sim_models/verilog/efx_dpram10.v',HERE/'tb_write_payload_route.sv']+[HERE/'variants/write_payload_route'/n for n in ('arena_client_router2.sv','tensor_arena12.sv')]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst)))
  for name in ('arena_client_router2.sv','tensor_arena12.sv'):
   row=next(x for x in meta['inputs'] if Path(x['snapshot']).name==name);f=Path(row['snapshot']);assert sha(f)==row['sha256']
   body=f.read_text()
   for module in ('arena_client_router2','tensor_dual_bank','tensor_arena12'):
    body=re.sub(r'\b'+module+r'\b',module+'_ref',body)
   dst=root/(f.stem+'_ref.sv');dst.write_text(body)
   records.append(dict(source=str(f),source_sha256=sha(f),snapshot=str(dst),sha256=sha(dst),transform='rename modules only'))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in (71,391,1027):
   name=f'seed{seed}.log'
   body,secs=run([MS/'vsim.exe','-c','work.tb_write_payload_route',f'-gSEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,name,timeout=180)
   lines=re.findall(r'^# (PASS WRITE_PAYLOAD_ROUTE.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=lines[0],seconds=secs,log_sha256=sha(root/name)))
  for x in records:assert sha(x['snapshot'])==x['sha256']
  result=dict(status='WRITE_PAYLOAD_ROUTE_FOUR_STATE_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),whole_board=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
