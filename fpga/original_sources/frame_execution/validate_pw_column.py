"""PW column engine actual arena/weight RAM/pool/write-ACK numeric suite."""
import argparse,json,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
from validate_banked_pool import sources
sys.path.insert(0,str(ROOT/'network_ab'))
from contract_reference import requantize
from validate_dw_block import byte
def vectors(root,limit=None,quick=False,shapes=None):
 root.mkdir();cases=[];gold=[]
 if shapes is None:shapes=[(80,4,4,4,255,0,0),(80,4,4,4,255,80,3),
 (33,3,1,2,255,79,6),(84,4,2,4,255,78,9),(88,4,4,2,255,76,12),
 (32,4,2,1,255,0,0),(33,3,1,1,7,287,177),(80,1,4,4,255,240,179),
 (320,4,1,2,255,0,0),(80,4,4,4,255,160,176),(84,2,2,1,255,236,178),
 (1,1,1,1,7,0,0)]
 if quick:shapes=[shapes[i] for i in (2,6,11)]
 if limit:shapes=shapes[:limit]
 for c,(w,h,gi,go,mask,ox,oy) in enumerate(shapes):
  origin=ox;span=((w+3)//4)*gi;outspan=((w+3)//4)*go
  pcount=[0]*3;planes=[];bases=[];op=[];ob=[];outcount=[0]*3
  for y in range(6):
   plane=(oy+y+c)%3
   planes.append(plane);bases.append(pcount[plane]*span);pcount[plane]+=1
  for y in range(4):
   plane=(oy+y+c+1)%3;op.append(plane);ob.append(2*span+16+outcount[plane]*outspan);outcount[plane]+=1
  assert max(ob)+outspan<=1024
  cases.append([c,len(gold),w,h,go,ox,origin,320,span,gi,mask,sum(p<<(2*y) for y,p in enumerate(planes)),
   sum(v<<(10*y) for y,v in enumerate(bases)),sum(p<<(2*y) for y,p in enumerate(op)),sum(v<<(10*y) for y,v in enumerate(ob))])
  for y in range(h):
   for x in range(w):
    for group in range(go):
     values=[]
     for lane in range(8):
      dot=0
      for ch in range(gi*8):
       b=planes[y]*4+x%4;address=bases[y]+(x//4)*gi+ch//8
       dot+=byte(b,address,ch%8,c)*(((group*5+ch*3+lane)%7)-3)
      value,ov=requantize(dot,(group*8+lane)*5-73,16384,relu=bool((group+lane)%2));assert not ov
      values.append(value if mask>>lane&1 else 0)
     gold.append(sum((v&255)<<(l*8) for l,v in enumerate(values)))
 (root/'golden.hex').write_text(''.join(f'{x:016x}\n' for x in gold))
 (root/'cases.txt').write_text(str(len(cases))+'\n'+''.join(' '.join(str(x) if n<11 else f'{x:x}' for n,x in enumerate(c))+'\n' for c in cases))
 return dict(cases=len(cases),output_words=len(gold),scalar_checks=len(gold)*8,shapes=shapes)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--seeds',nargs='+',type=int,default=[71]);p.add_argument('--bounded',action='store_true');p.add_argument('--snapshot-only',action='store_true');p.add_argument('--quick',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+sources()+[HERE/n for n in
 ('tensor_arena12.sv','pw_column_reader.sv','pw_column_dispatch32.sv','column_write4.sv','phase_weight_cache.sv','tb_pw_column_block.sv')]
 paths=[HERE/'variants/registered_credit'/f.name if f.name=='shared_pool_banked_fifo.sv' else f for f in paths]
 records=[]
 if a.bounded:paths=[HERE/'variants/bounded_columns'/f.name if f.name in ('pw_column_reader.sv','tb_pw_column_block.sv') else f for f in paths]
 try:
  for f in paths+[Path(__file__),ROOT/'network_ab/contract_reference.py']:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  oracle=vectors(root/'vectors',a.limit,a.quick)
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,vectors={f.name:sha(f) for f in (root/'vectors').iterdir()},oracle=oracle),indent=2))
  if a.snapshot_only:print(json.dumps(dict(status='SNAPSHOT_ONLY_NOT_PASS',oracle=oracle)),flush=True);return
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[root/f.name for f in paths]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in a.seeds:
   body,seconds=run([MS/'vsim.exe','-c','work.tb_dw_block','-gSEED='+str(seed),'+VECTORS='+(root/'vectors').as_posix(),
   '-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log')
   passed=re.findall(r'^# (PASS PW_COLUMN.*)$',body,re.M);rows=re.findall(r'^# BLOCK (.*)$',body,re.M)
   assert len(passed)==1 and len(rows)==oracle['cases']*2 and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=passed[0],rows=rows,seconds=seconds,log_sha256=sha(root/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_PW_COLUMN_REAL_PORT_SOURCE_PASS',results=results,oracle=oracle,inputs_sha256=sha(root/'inputs.json'),actual_primitive_ram=112,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
