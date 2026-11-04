"""Next header prefetch and active cyclic weights share five real RAM read ports."""
import argparse,json,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--variant',type=Path,default=HERE/'variants/compact_header');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  original=ROOT/'tmp/full-frame-closure/resident-feed-source-attempt6';vectors=root/'vectors';shutil.copytree(original/'vectors',vectors)
  old=json.loads((original/'inputs.json').read_text())
  for n,h in old['vectors'].items():assert sha(vectors/n)==h
  paths=[EF/'sim_models/verilog/efx_ram10.v',HERE/'resident_weight_bank64.sv',HERE.parent/'full_frame_closure/coefficient_bank_store_fast.sv']+[HERE/n for n in ('phase_resident_reader.sv','next_header_prefetch.sv','resident_read_mux2.sv','tb_header_prefetch.sv')]
  from frozen_sources import frozen_paths
  paths=frozen_paths(paths,a.base)
  variants=a.variant.resolve();assert variants.is_relative_to(HERE/'variants')
  paths=[variants/f.name if (variants/f.name).is_file() else f for f in paths]
  records=[]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,vectors=old['vectors'],runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  checker=HERE.parent/'coefficient_store/validate_readback.py';dump=root/'readback.hex'
  do='onerror {quit -code 1}\nrun 100 us\nif {[examine -radix unsigned sim:/tb_header_prefetch/verify_checkpoint] != 1} {error "missing real readback"}\n'
  do+='puts [exec {'+Path(sys.executable).as_posix()+'} -B {'+checker.as_posix()+'} --image-dir {'+vectors.as_posix()+'} --readback {'+dump.as_posix()+'}]\n'
  do+='force -deposit sim:/tb_header_prefetch/software_verified 1\nrun -all\nquit -code 0\n';(root/'run.do').write_text(do)
  body,secs=run([MS/'vsim.exe','-c','work.tb_header_prefetch','+VECTORS='+vectors.as_posix(),'+DUMP='+dump.as_posix(),'-do','do run.do'],root,'simulation.log')
  lines=re.findall(r'^# (PASS HEADER_PREFETCH.*)$',body,re.M);assert len(lines)==1
  assert 'PASS COMPACT_SCALE_FIELDS cases=4 positive_negative_valid_invalid=1' in body
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='COMPACT_HEADER_PREFETCH_SHARED_REAL_RESIDENT_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_layer_scheduler=False,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
