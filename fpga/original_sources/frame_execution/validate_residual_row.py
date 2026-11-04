"""Real finite arena read -> same 32/4 pool twice -> acknowledged arena write.
No arithmetic/RAM surrogate. Freeze all source/vector hashes; numeric oracle is
the frozen contract_reference residual function with explicit scale fixtures.
"""
import argparse,json,re,shutil,traceback,random
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
from validate_banked_pool import sources
from residual_q17_align import encode_alignment,fixtures,residual_value,requantize
def row_sources():
 return sources()+[HERE/n for n in ('tensor_arena12.sv','residual_packet_dispatch32.sv','residual_row_shared.sv')]
def vectors(root):
 root.mkdir()
 records=[encode_alignment(x) for x in fixtures()]
 rows=[];data={k:[] for k in ('main','skip','align','golden')};rng=random.Random(301927);off=0
 shapes=[(w,4,f) for w in (80,1,2,3,4,7,31,32,33,79,320) for f in range(3)]
 shapes += [(w,g,f) for w in (7,80) for g in (1,2) for f in range(3)]
 for ci,(width,g,f) in enumerate(shapes):
  r=records[f];mp=ci%3;sp=(mp+1)%3;op=(mp+2)%3
  mb=3+(ci*17)%101;sb=151+(ci*11)%71;ob=500
  md=ci%8;ds=(ci*3+1)%8
  rows.append([ci,off,width,g,mp,sp,op,mb,sb,ob,md,ds,int(r['main_scaled']),r['multiplier'],r['post_multiplier']])
  for x in range(width):
   for ch in range(g):
    a=[rng.randrange(-128,128) for _ in range(8)];b=[rng.randrange(-128,128) for _ in range(8)]
    if (x+ch)%7==0:a[0]=-128;b[0]=127;a[1]=127;b[1]=-128
    al=[requantize(v,0,r['multiplier'])[0] for v in (a if r['main_scaled'] else b)]
    gold=[residual_value(aa,bb,r['parameters']) for aa,bb in zip(a,b)]
    for k,v in zip(data,(a,b,al,gold)):data[k].append(sum((z&255)<<(8*l) for l,z in enumerate(v)))
  off+=width*g
 assert off<65536
 for k,v in data.items():(root/(k+'.hex')).write_text(''.join(f'{w:016x}\n' for w in v))
 (root/'cases.txt').write_text(f'{len(rows)} {off}\n'+''.join(' '.join(map(str,row))+'\n' for row in rows))
 return dict(cases=len(rows),words=off,scalar_checks=(off+320)*24,parameters=records)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[71]);p.add_argument('--testbench',type=Path,default=HERE/'tb_residual_row_parallel.sv');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+row_sources()+[a.testbench.resolve()]
 records=[]
 try:
  for f in paths+[Path(__file__),HERE/'run_units.py',HERE/'residual_q17_align.py',HERE/'residual_tables.py',
      ROOT/'network_ab/contract_reference.py',ROOT/'network_ab/model_b_contract.json']:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  vec=vectors(root/'vectors')
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,vectors={f.name:sha(f) for f in (root/'vectors').iterdir()},oracle=vec),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[root/f.name for f in paths]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in a.seeds:
   body,secs=run([MS/'vsim.exe','-c','work.tb_residual_row_shared','-gSEED='+str(seed),
    '+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log')
   lines=re.findall(r'^# (PASS RESIDUAL_ROW.*)$',body,re.M);rows=re.findall(r'^# ROW (.*)$',body,re.M)
   assert len(lines)==1 and len(rows)==vec['cases']+1 and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=lines[0],rows=rows,seconds=secs,log_sha256=sha(root/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_RESIDUAL_REAL_ARENA_SHARED_POOL_ROW_PASS',results=results,oracle=vec,
   inputs_sha256=sha(root/'inputs.json'),actual_primitive_ram=108,unique_shared_pools=1,full_frame=False,pnr=False,production_parameters=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','oracle','inputs_sha256')}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

