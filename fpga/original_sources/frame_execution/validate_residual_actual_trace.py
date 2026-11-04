"""Actual frozen residual allocator bases, planes, row pairs and crop offsets."""
import ast,json,sys,shutil
from pathlib import Path
import validate_residual_column as base
from run_units import HERE,ROOT,sha
from residual_q17_align import encode_alignment,fixtures,residual_value
from validate_dw_block import byte
TRACE=ROOT/'tmp/frame-execution/residual-layout-ports1/width80-rows4.json'
def trace_vectors(root,full=False,part='smoke'):
 root.mkdir()
 cases=[]
 for strip in json.loads(TRACE.read_text()):
  for task in strip['residual_tasks']:
   read={ast.literal_eval(k):v for k,v in task['read_descriptors'].items()}
   write={ast.literal_eval(k):v for k,v in task['write_descriptors'].items()}
   parents=list(dict.fromkeys(k[0] for k in read))
   assert len(parents)==2
   lo,hi=strip['ranges'][task['node']];w=hi-lo+1
   for y in range(task['lo'],task['hi']+1,2):
    h=min(2,task['hi']-y+1);desc=[]
    for yy in range(y,y+h):
     src=[[*read[(name,yy)][:2],lo-strip['ranges'][name][0]] for name in parents]
     dest=write[(task['node'],yy)];assert dest[2:]==[w,4]
     desc.append((src,dest[:2]))
    cases.append(dict(index=len(cases),tile=strip['x0'],node=task['node'],y=y,w=w,h=h,desc=desc))
 assert len(cases)==1080
 if part=='smoke':
  unique={}
  for c in cases:unique.setdefault((c['w'],c['desc'][0][1][0]),c)
  cases=list(unique.values())
 else:
  assert part in ('0','1')
  cases=cases[int(part)*540:(int(part)+1)*540]
 records=[encode_alignment(p) for p in fixtures()]
 rows=[];gold=[];detail=[]
 for ci,case in enumerate(cases):
  w,h=case['w'],case['h'];r=records[case['index']%3];g=4
  planes=[];bases=[];deltas=[];ops=[];obs=[];off=len(gold)
  for y,(src,dest) in enumerate(case['desc']):
   for pp,bb,dd in (src if r['main_scaled'] else src[::-1]):
    assert dd>=0 and bb+((dd+w-1)//4)*g+g<=1024
    planes.append(pp);bases.append(bb);deltas.append(dd)
   ops.append(dest[0]);obs.append(dest[1])
   for x in range(w):
    for ch in range(g):
     out=[]
     for lane in range(8):
      vals=[byte(pp*4+(x+dd)%4,bb+(x+dd)//4*g+ch,lane,ci) for pp,bb,dd in src]
      out.append(residual_value(*vals,r['parameters']))
     gold.append(sum((v&255)<<(8*l) for l,v in enumerate(out)))
  pack=lambda arr,b:sum(v<<(i*b) for i,v in enumerate(arr))
  rows.append([ci,off,w,h,g,r['multiplier'],r['post_multiplier'],pack(planes,2),pack(bases,10),pack(deltas,11),pack(ops,2),pack(obs,10)])
  detail.append(case)
 assert len(gold)<=524288
 (root/'golden.hex').write_text(''.join(f'{v:016x}\n' for v in gold))
 (root/'cases.txt').write_text(f'{len(rows)} {len(gold)}\n'+''.join(' '.join(str(v) if i<7 else f'{v:x}' for i,v in enumerate(row))+'\n' for row in rows))
 return dict(cases=len(rows),gold_words=len(gold),part=part,detail=detail,trace_sha256=sha(TRACE),all_actual_allocator_commands=part!='smoke')
if __name__=='__main__':
 i=sys.argv.index('--part');part=sys.argv[i+1];del sys.argv[i:i+2]
 base.vectors=lambda root,full=False:trace_vectors(root,full,part)
 sys.argv+=['--testbench',str(HERE/'variants/residual_actual/tb_residual_column.sv')]
 before={str(p):sha(p) for p in (Path(__file__),TRACE,Path(base.__file__))}
 base.main()
 root=Path(sys.argv[sys.argv.index('--root')+1])
 for p,h in before.items():assert sha(p)==h
 shutil.copy2(Path(__file__),root/Path(__file__).name)
 (root/'actual-trace-runner.json').write_text(json.dumps(before,indent=2))
