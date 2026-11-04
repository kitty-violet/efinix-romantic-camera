"""Measure the exact residual row widths in the frozen80-column fusion trace."""
import json,random,sys,shutil
from pathlib import Path
import validate_residual_row as base
from run_units import HERE,ROOT,sha
from residual_q17_align import encode_alignment,fixtures,residual_value,requantize
TRACE=ROOT/'tmp/frame-execution/residual-layout-ports1/width80-rows4.json'
def profile_vectors(root):
 root.mkdir();record=encode_alignment(fixtures()[0]);rng=random.Random(301927)
 widths=sorted({v['ranges'][t['node']][1]-v['ranges'][t['node']][0]+1 for v in json.loads(TRACE.read_text()) for t in v['residual_tasks']})
 assert widths==[80,82,84,88]
 rows=[];data={k:[] for k in ('main','skip','align','golden')};off=0
 for ci,width in enumerate([w for w in widths for _ in range(2)]):
  rows.append([ci,off,width,4,ci%3,(ci+1)%3,(ci+2)%3,10,210,500,ci%8,(ci*3+1)%8,1,record['multiplier'],record['post_multiplier']])
  for x in range(width):
   for g in range(4):
    a=[rng.randrange(-128,128) for _ in range(8)];b=[rng.randrange(-128,128) for _ in range(8)]
    al=[requantize(v,0,record['multiplier'])[0] for v in a]
    gold=[residual_value(aa,bb,record['parameters']) for aa,bb in zip(a,b)]
    for k,v in zip(data,(a,b,al,gold)):data[k].append(sum((z&255)<<(8*l) for l,z in enumerate(v)))
  off+=width*4
 for k,v in data.items():(root/(k+'.hex')).write_text(''.join(f'{w:016x}\n' for w in v))
 (root/'cases.txt').write_text(f'{len(rows)} {off}\n'+''.join(' '.join(map(str,row))+'\n' for row in rows))
 return dict(cases=len(rows),words=off,scalar_checks=(off+320)*24,parameters=[record],trace_sha256=sha(TRACE))
if __name__=='__main__':
 before={str(p):sha(p) for p in (Path(__file__),TRACE,Path(base.__file__))}
 base.vectors=profile_vectors;base.main()
 root=Path(sys.argv[sys.argv.index('--root')+1])
 for p,h in before.items():assert sha(p)==h
 shutil.copy2(Path(__file__),root/Path(__file__).name)
 (root/'profile-runner.json').write_text(json.dumps(before,indent=2))

