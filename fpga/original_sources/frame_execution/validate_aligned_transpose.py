"""Prove fixed-lane transpose keeps old handshakes and independent byte order."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 fw=HERE.parent/'pw_array_feed32';original=fw/'stream_group_transpose.sv';new=fw/'variants/aligned_transpose/stream_group_transpose.sv'
 paths=[original,new,HERE/'tb_aligned_transpose.sv',Path(__file__)];records=[]
 for i,f in enumerate(paths):
  dst=root/(str(i)+'-'+f.name);shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
 old=(root/'0-stream_group_transpose.sv').read_text().replace('module stream_group_transpose #','module stream_group_transpose_reference #',1)
 (root/'reference.sv').write_text(old)
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,reference_generated_sha256=sha(root/'reference.sv'),rename_only=True),indent=2))
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv',root/'reference.sv',root/'1-stream_group_transpose.sv',root/'2-tb_aligned_transpose.sv'],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  rows=[]
  for pixels in (1,2,4):
   for seed in (71,391,1027):
    name=f'w{pixels}-seed{seed}.log'
    body,seconds=run([MS/'vsim.exe','-c','work.tb_aligned_transpose',f'-gWRITE_PIXELS={pixels}',f'-gRANDOM_SEED={seed}',
      '-do','onerror {quit -code 1}; run -all; quit -code 0'],root,name)
    lines=re.findall(r'^# (PASS ALIGNED_TRANSPOSE.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
    rows.append(dict(width=pixels*64,seed=seed,line=lines[0],seconds=seconds,log=str(root/name),log_sha256=sha(root/name)))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_TRANSPOSE_CYCLE_EQUIVALENCE_AND_NUMERIC_PASS',results=rows,inputs_sha256=sha(root/'inputs.json'),full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
