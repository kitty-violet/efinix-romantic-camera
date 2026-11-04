"""DW block: real arena ports -> actual single pool -> real write ACK/readback.
Independent Python convolution oracle, finite source/output allocations. Numeric
coefficients are fixtures; this does not validate training quality or DDR/fullframe.
"""
import argparse,json,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
from validate_banked_pool import sources
sys.path.insert(0,str(ROOT/'network_ab'))
from contract_reference import requantize

def byte(b,a,l,c):return ((b*37+a*13+l*29+c*11)&255)-128
def vectors(root,limit=None,shapes=None,quick=False):
 root.mkdir();cases=[];gold=[]
 if shapes is None:shapes=[(80,4,4,0,0),(80,4,4,80,3),(82,4,4,79,6),(84,4,4,78,9),(88,4,4,76,12),
  (32,4,1,0,0),(33,3,2,287,177),(80,1,4,240,179),(320,4,1,0,0),
  (80,4,4,160,176),(84,2,2,236,178)]
 if quick:shapes=[(9,2,1,79,6),(1,1,1,0,0),(33,3,2,287,177)]
 if limit:shapes=shapes[:limit]
 for c,(w,h,g,ox,oy) in enumerate(shapes):
  origin=max(0,ox-1);span=((w+5)//4)*g;outspan=((w+3)//4)*g
  pcount=[0]*3;alloc={};planes=[];bases=[];op=[];ob=[];outcount=[0]*3
  for y in range(6):
   sy=max(0,min(179,oy+y-1));plane=(sy+c)%3
   if sy not in alloc:alloc[sy]=pcount[plane]*span;pcount[plane]+=1
   planes.append(plane);bases.append(alloc[sy])
  for y in range(4):
   plane=(oy+y+c+1)%3;op.append(plane);ob.append(2*span+16+outcount[plane]*outspan);outcount[plane]+=1
  assert max(ob)+outspan<=1024
  cases.append([c,len(gold),w,h,g,ox,origin,320,span,sum(p<<(2*y) for y,p in enumerate(planes)),
   sum(v<<(10*y) for y,v in enumerate(bases)),sum(p<<(2*y) for y,p in enumerate(op)),sum(v<<(10*y) for y,v in enumerate(ob))])
  for y in range(h):
   for x in range(w):
    for group in range(g):
     values=[]
     for lane in range(8):
      dot=0
      for ky in range(3):
       for kx in range(3):
        sx=max(0,min(319,ox+x+kx-1));lx=sx-origin;b=planes[y+ky]*4+lx%4;a=bases[y+ky]+(lx//4)*g+group
        weight=((group*5+(ky*3+kx)*3+lane)%7)-3
        dot+=byte(b,a,lane,c)*weight
      value,ov=requantize(dot,(group*8+lane)*5-73,16384,relu=bool((group+lane)%2));assert not ov;values.append(value)
     gold.append(sum((v&255)<<(l*8) for l,v in enumerate(values)))
 (root/'golden.hex').write_text(''.join(f'{x:016x}\n' for x in gold))
 (root/'cases.txt').write_text(str(len(cases))+'\n'+''.join(' '.join(str(x) if n<9 else f'{x:x}' for n,x in enumerate(c))+'\n' for c in cases))
 return dict(cases=len(cases),output_words=len(gold),scalar_checks=len(gold)*8,shapes=shapes)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[71]);p.add_argument('--limit',type=int);p.add_argument('--registered-credit',action='store_true');p.add_argument('--striped',action='store_true');p.add_argument('--window',action='store_true');p.add_argument('--paired-window',action='store_true');p.add_argument('--column',action='store_true');p.add_argument('--cluster',action='store_true');p.add_argument('--bounded',action='store_true');p.add_argument('--quick',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+sources()+[HERE/n for n in
  ('tensor_arena12.sv','dw_block_plan32.sv','dw_bank_read32.sv','dw_block_write4.sv','dw_coefficient_registers.sv','tb_dw_block.sv')]
 if a.registered_credit:paths=[HERE/'variants/registered_credit'/f.name if f.name=='shared_pool_banked_fifo.sv' else f for f in paths]
 if a.striped:paths=[HERE/'variants/striped_words'/f.name if f.name in ('dw_block_plan32.sv','dw_bank_read32.sv','dw_coefficient_registers.sv') else f for f in paths]
 if a.window:
  assert not a.striped
  paths=[f for f in paths if f.name not in ('dw_block_plan32.sv','dw_bank_read32.sv','tb_dw_block.sv')]+[HERE/n for n in ('dw_window_prefetch.sv','dw_window_dispatch32.sv','tb_dw_window_block.sv')]
 if a.paired_window:
  assert a.window
  paths=[HERE/'variants/window_paired'/f.name if f.name=='dw_window_prefetch.sv' else f for f in paths]
 if a.column:
  assert not a.striped and not a.window
  paths=[f for f in paths if f.name not in ('dw_block_plan32.sv','dw_bank_read32.sv','tb_dw_block.sv')]+[HERE/n for n in ('dw_column_reader.sv','dw_column_dispatch32.sv','tb_dw_column_block.sv')]
 if a.cluster:
  from map_dw_cluster import selected
  paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+selected()+[HERE/'tensor_arena12.sv',HERE/'tb_dw_column_cluster.sv']
 records=[]
 if a.bounded:
  assert a.cluster
  paths=[HERE/'variants/bounded_columns'/f.name if f.name in ('dw_column_reader.sv','tb_dw_column_cluster.sv') else f for f in paths]
 try:
  for f in paths+[Path(__file__),ROOT/'network_ab/contract_reference.py']:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  oracle=vectors(root/'vectors',a.limit,quick=a.quick);manifest=dict(inputs=records,vectors={f.name:sha(f) for f in (root/'vectors').iterdir()},oracle=oracle,striped=a.striped,window=a.window)
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*(['+define+DW_STRIPED'] if a.striped else []),*[root/f.name for f in paths]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in a.seeds:
   body,seconds=run([MS/'vsim.exe','-c','work.tb_dw_block','-gSEED='+str(seed),'+VECTORS='+(root/'vectors').as_posix(),
    '-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log')
   lines=re.findall(r'^# (PASS DW_BLOCK.*)$',body,re.M);rows=re.findall(r'^# BLOCK (.*)$',body,re.M)
   assert len(lines)==1 and len(rows)==oracle['cases']*2 and 'Errors: 0, Warnings: 0' in body
   results.append(dict(seed=seed,line=lines[0],rows=rows,seconds=seconds,log_sha256=sha(root/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_DW_REAL_PORT_BLOCK_SOURCE_PASS',results=results,oracle=oracle,inputs_sha256=sha(root/'inputs.json'),
   actual_primitive_ram=108,unique_shared_pools=1,coefficient_fixture=True,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','oracle')}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
