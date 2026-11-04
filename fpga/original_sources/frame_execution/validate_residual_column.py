"""Independent original signed-byte residual oracle; no in-place aligned copy."""
import argparse,json,sys,shutil,re,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,run,sha
from validate_compute_residual_shared import selected
from residual_q17_align import encode_alignment,fixtures,residual_value
from validate_dw_block import byte
def vectors(root,full=False):
 root.mkdir()
 shapes=[(w,2,g,f) for w,g,f in [(1,1,0),(3,4,1),(7,2,2),(32,4,0),(80,4,1),(82,4,0),(84,4,2),(88,4,0),(160,2,1),(320,4,0)]]
 if full:shapes += [(w,r,g,f) for w in (2,4,31,33,79,80,320) for r in (1,2) for g in (1,2,4) for f in range(3)]
 rows=[];gold=[];detail=[];records=[encode_alignment(p) for p in fixtures()]
 for ci,(w,h,g,f) in enumerate(shapes):
  p=records[f];span=((w+10)//4)*g
  mp=ci%3;sp=(mp+1)%3;op=(mp+2)%3
  planes=[];bases=[];deltas=[];obs=[];offs=len(gold)
  for y in range(h):
   md=(ci+y)%8;sd=(ci*3+y+1)%8;mb=10+y*span;sb=200+y*span
   assert max(mb,sb)+span<=1024
   src=[(mp,mb,md),(sp,sb,sd)]
   for pp,bb,dd in (src if p['main_scaled'] else src[::-1]):planes.append(pp);bases.append(bb);deltas.append(dd)
   obs.append(y*((w+3)//4)*g)
   for x in range(w):
    for ch in range(g):
     value=[]
     for lane in range(8):
      aa=byte(mp*4+(x+md)%4,mb+(x+md)//4*g+ch,lane,ci)
      bb=byte(sp*4+(x+sd)%4,sb+(x+sd)//4*g+ch,lane,ci)
      value.append(residual_value(aa,bb,p['parameters']))
     gold.append(sum((v&255)<<(l*8) for l,v in enumerate(value)))
  pack=lambda arr,b:sum(v<<(i*b) for i,v in enumerate(arr))
  rows.append([ci,offs,w,h,g,p['multiplier'],p['post_multiplier'],pack(planes,2),pack(bases,10),pack(deltas,11),pack([op]*h,2),pack(obs,10)])
  detail.append(dict(case=ci,width=w,rows=h,groups=g,parameters=p))
 assert len(gold)<524288
 (root/'golden.hex').write_text(''.join(f'{v:016x}\n' for v in gold))
 (root/'cases.txt').write_text(f'{len(rows)} {len(gold)}\n'+''.join(' '.join(str(v) if i<7 else f'{v:x}' for i,v in enumerate(row))+'\n' for row in rows))
 return dict(cases=len(rows),gold_words=len(gold),parameters=records,detail=detail)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--seeds',nargs='+',type=int,default=[71]);p.add_argument('--full',action='store_true');p.add_argument('--snapshot-only',action='store_true');p.add_argument('--testbench',type=Path,default=HERE/'tb_residual_column.sv');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+selected()+[HERE/'tensor_arena12.sv',a.testbench.resolve()]
  records=[]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  oracle=vectors(root/'vectors',a.full)
  deps=[Path(__file__),HERE/'validate_compute_residual_shared.py',HERE/'residual_q17_align.py',HERE/'residual_tables.py',HERE/'validate_dw_block.py',ROOT/'network_ab/contract_reference.py',ROOT/'network_ab/model_b_contract.json']
  manifest=dict(inputs=records,dependencies={str(f):sha(f) for f in deps},oracle=oracle,vectors={f.name:sha(f) for f in (root/'vectors').iterdir()})
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  if a.snapshot_only:print('SNAPSHOT_ONLY_NOT_PASS');return
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in a.seeds:
   body,secs=run([MS/'vsim.exe','-c','work.tb_residual_column','-gSEED='+str(seed),'+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log')
   lines=re.findall(r'^# (PASS RESIDUAL_COLUMN.*)$',body,re.M);rows=re.findall(r'^# RESCOL (.*)$',body,re.M)
   assert len(lines)==1 and len(rows)==oracle['cases']*2+1 and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=lines[0],rows=rows,seconds=secs,log_sha256=sha(root/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  for f,h in manifest['dependencies'].items():assert sha(f)==h
  result=dict(status='LOCAL_SHARED_COLUMN_FUSED_RESIDUAL_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),shared_reader=True,shared_writer=True,shared_pool=True,intermediate_arena_write=False,arena_ram=84,pool_ram=36,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
