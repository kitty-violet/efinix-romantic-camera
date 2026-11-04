"""Four-state replay of an exact frozen continuous-frame RTL snapshot."""
import argparse,json,re,shutil,subprocess,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--limit',type=int,default=160);p.add_argument('--profile',type=int,default=1);p.add_argument('--simulation-timeout',type=int,default=600);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
try:
 prior=json.loads((a.snapshot/'inputs.json').read_text());vec=Path(prior['vectors']);vm=json.loads((vec/'frame_manifest.json').read_text());records=[]
 for item in prior['inputs']:
  f=Path(item['snapshot']);assert sha(f)==item['sha256'];dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=item['path'],snapshot=str(dst),sha256=sha(dst)))
 manifest=dict(inputs=records,original_inputs_sha256=sha(a.snapshot/'inputs.json'),vector_manifest_sha256=sha(vec/'frame_manifest.json'),limit=a.limit,profile=a.profile)
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
 checker=HERE.parent/'coefficient_store/validate_readback.py';dump=root/'readback.hex'
 command=subprocess.list2cmdline([sys.executable,'-B',str(checker),'--image-dir',str(vec),'--readback',str(dump)])
 final=next(n['offset'] for n in vm['nodes'] if n['id']=='rgb_head');con=vm.get('concurrency',{})
 args=[MS/'vsim.exe','-c','work.tb_fourway_full','-gOPTIMIZED='+str(int(prior['optimized'])),'-gCONCURRENT='+str(int(prior.get('concurrent',False))),
  '-gCOMMANDS='+str(vm['commands']),'-gEXPECTED_WORDS='+str(vm['expected_compute_words']),'-gCOMPUTE_COMMANDS='+str(con.get('compute_commands',0)),'-gDMA_COMMANDS='+str(con.get('dma_commands',0)),
  '+VECTORS='+vec.as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+command,'+FINAL_GOLD_OFFSET='+str(final),'+LIMIT='+str(a.limit),'+PROFILE='+str(a.profile),
  '-do','onerror {quit -code 1}; run -all; quit -code 0']
 body,seconds=run(args,root,'simulation.log',timeout=a.simulation_timeout);matches=re.findall(r'^# SMOKE FULL_CHAIN commands=(\d+) cycles=(\d+) checks=(\d+)$',body,re.M)
 assert len(matches)==1 and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and 'Errors: 0, Warnings: 0' in body
 result=dict(status='CONTINUOUS_FRAME_FOUR_STATE_PREFIX_PASS',commands=int(matches[0][0]),cycles=int(matches[0][1]),byte_checks=int(matches[0][2]),profile=a.profile,seconds=seconds,
  inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_frame=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
