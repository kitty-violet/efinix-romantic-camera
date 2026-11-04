"""Actual five-bank official-RAM cyclic reader and verified-image boundary."""
import argparse,json,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,run,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed'))
from feed_reference import generate
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[71,391,1027]);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  generate(root/'vectors','dense')
  paths=[EF/'sim_models/verilog/efx_ram10.v',HERE/'resident_weight_bank64.sv',
    HERE.parent/'full_frame_closure/coefficient_bank_store_fast.sv',HERE/'phase_resident_reader.sv',HERE/'tb_phase_resident_reader.sv']
  records=[]
  checker=HERE.parent/'coefficient_store/validate_readback.py'
  deps=[Path(__file__),checker,HERE.parent/'pw_array_feed/feed_reference.py']+[ROOT/'network_ab'/n for n in ('coefficient_layout.py','model_b_contract.json')]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  manifest=dict(inputs=records,dependencies={str(f):sha(f) for f in deps},vectors={f.name:sha(f) for f in (root/'vectors').iterdir() if f.is_file()})
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in a.seeds:
   dump=root/f'readback-{seed}.hex'
   do='onerror {quit -code 1}\nrun 100 us\n'
   do+='if {[examine -radix unsigned sim:/tb_phase_resident_reader/verify_checkpoint] != 1} {error "Missing actual readback"}\n'
   do+='puts [exec {'+Path(sys.executable).as_posix()+'} -B {'+checker.as_posix()+'} --image-dir {'+(root/'vectors').as_posix()+'} --readback {'+dump.as_posix()+'}]\n'
   do+='force -deposit sim:/tb_phase_resident_reader/software_verified 1\nrun -all\nquit -code 0\n'
   (root/f'run-{seed}.do').write_text(do,encoding='ascii')
   body,secs=run([MS/'vsim.exe','-c','work.tb_phase_resident_reader','-gSEED='+str(seed),'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'-do',f'do run-{seed}.do'],root,f'seed{seed}.log')
   lines=re.findall(r'^# (PASS PHASE_RESIDENT.*)$',body,re.M)
   assert len(lines)==1 and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=lines[0],seconds=secs,log_sha256=sha(root/f'seed{seed}.log'),readback_sha256=sha(dump)))
   print(lines[0],flush=True)
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  for f,h in manifest['dependencies'].items():assert sha(f)==h
  result=dict(status='LOCAL_PHASE_RESIDENT_OFFICIAL_RAM_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),official_ram_count=20,
    periods=[8,9,16,27,32,64,128],layer_weight_copy=False,full_frame=False,pnr=False,production_trained_image=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

