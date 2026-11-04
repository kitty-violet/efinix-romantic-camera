"""Verify one mode-switching DW/PW cluster with real arena/resident RAMs."""
import argparse,json,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,run,sha
from validate_banked_pool import sources
from fourway_vectors import vectors
def selected(fast_skip=False,flex_skip=False,production=False):
 from validate_compute_mailbox import selected as inherited
 paths=[HERE/('variants/production_banked' if production else 'variants/fourway')/f.name if f.name=='compute_column_cluster.sv' else HERE/'variants/mixed_fifo80'/f.name if f.name=='shared_pool_banked_fifo.sv' else HERE/'variants/fifo80'/f.name if f.name=='banked_result_fifo.sv' else f for f in inherited()]+[HERE/n for n in ('pair_column_reader.sv','pair_column_dispatch32.sv','result_pair_expand4.sv','column_write8.sv','variants/flex_skip/pw_skip_fuse4.sv' if flex_skip else 'variants/throughput/pw_skip_fuse4.sv' if fast_skip else 'pw_skip_fuse4.sv')]
 if production:
  paths=[HERE/'variants/production_banked'/f.name if f.name in ('shared_pool_banked_fifo.sv','pw_skip_fuse4.sv') else f for f in paths if f.name not in ('compute_column_reader.sv','pair_column_reader.sv','dw_column_dispatch32.sv','pw_column_dispatch32.sv','pair_column_dispatch32.sv')]+[HERE/'unified_column_reader.sv',HERE/'unified_column_dispatch32.sv',HERE/'shallow_result_fifo.sv',EF/'sim_models/verilog/efx_srl8.v']
 return paths

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[71]);p.add_argument('--full',action='store_true');p.add_argument('--fast-skip',action='store_true');p.add_argument('--flex-skip',action='store_true');p.add_argument('--production',action='store_true');p.add_argument('--factored-dispatch',action='store_true');p.add_argument('--snapshot-only',action='store_true');p.add_argument('--selected-cases',type=int,nargs='+');p.add_argument('--simulation-timeout',type=int,default=600);p.add_argument('--specialized',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  oracle=vectors(root/'vectors',a.full,specialized=a.specialized)
  if a.selected_cases is not None:
   lines=(root/'vectors/cases.txt').read_text().splitlines()[1:]
   assert len(set(a.selected_cases))==len(a.selected_cases) and all(0<=i<len(lines) for i in a.selected_cases)
   chosen=[lines[i] for i in a.selected_cases]
   kinds=[int(x.split()[2]) for x in chosen]
   assert sum(x!=y for x,y in zip(kinds,kinds[1:]))>=4,'Selection must cover four mode switches before starting a long run'
   assert any([int(x.split()[j]) for j in (2,3,4)]==[1,80,2] for x in chosen),'Selection must include explicit 512-cycle result backpressure fixture'
   (root/'vectors/cases.txt').write_text(str(len(chosen))+'\n'+'\n'.join(chosen)+'\n')
   oracle['full_cases']=oracle['cases'];oracle['cases']=len(chosen);oracle['selected_cases']=a.selected_cases
  paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+selected(a.fast_skip,a.flex_skip,a.production)+[
   HERE/'tensor_arena12.sv',HERE/'resident_weight_bank64.sv',HERE.parent/'full_frame_closure/coefficient_bank_store_fast.sv',HERE/('variants/production_banked' if a.production else 'variants/fourway')/'tb_compute_column_cluster.sv']
  records=[]
  if a.factored_dispatch:
   assert a.production
   paths=[HERE/'variants/factored_dispatch'/f.name if f.name=='unified_column_dispatch32.sv' else f for f in paths]
  checker=HERE.parent/'coefficient_store/validate_readback.py'
  deps=[Path(__file__),checker,HERE/'fourway_vectors.py',HERE/'residual_q17_align.py',HERE/'cluster_stride_vectors.py',HERE.parent/'pw_array_feed/feed_reference.py',HERE/'validate_dw_block.py']+[
   ROOT/'network_ab'/n for n in ('coefficient_layout.py','contract_reference.py','model_b_contract.json')]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  manifest=dict(inputs=records,dependencies={str(f):sha(f) for f in deps},oracle=oracle,
   vectors={f.name:sha(f) for f in (root/'vectors').iterdir() if f.is_file()})
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  if a.snapshot_only:print('SNAPSHOT_ONLY_NOT_PASS',flush=True);return
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in a.seeds:
   dump=root/f'readback-{seed}.hex'
   do='onerror {quit -code 1}\nrun 120 us\n'
   do+='if {[examine -radix unsigned sim:/tb_compute_column_cluster/verify_checkpoint] != 1} {error "Missing actual readback"}\n'
   do+='puts [exec {'+Path(sys.executable).as_posix()+'} -B {'+checker.as_posix()+'} --image-dir {'+(root/'vectors').as_posix()+'} --readback {'+dump.as_posix()+'}]\n'
   do+='force -deposit sim:/tb_compute_column_cluster/software_verified 1\nrun -all\nquit -code 0\n'
   (root/f'run-{seed}.do').write_text(do,encoding='ascii')
   body,secs=run([MS/'vsim.exe','-c','work.tb_compute_column_cluster','-gSEED='+str(seed),'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'-do',f'do run-{seed}.do'],root,f'seed{seed}.log',timeout=a.simulation_timeout)
   lines=re.findall(r'^# (PASS COMPUTE_CLUSTER.*)$',body,re.M);rows=re.findall(r'^# BLOCK (.*)$',body,re.M)
   assert len(lines)==1 and len(rows)==oracle['cases']*2 and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=lines[0],rows=rows,seconds=secs,log_sha256=sha(root/f'seed{seed}.log'),readback_sha256=sha(dump)))
   print(lines[0],flush=True)
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  for f,h in manifest['dependencies'].items():assert sha(f)==h
  result=dict(status='LOCAL_FOURWAY_CLUSTER_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),official_ram_count=None,ram_count_source='Use the matched mapped report; no inherited constant',
    unique_mac_pools=1,unique_arena_readers=1 if a.production else 2,physical_arena_banks=12,physical_ports_per_bank=2,
    unique_resident_readers=1,layer_weight_copy=False,header_external=True,
    full_frame=False,pnr=False,production_trained_image=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
