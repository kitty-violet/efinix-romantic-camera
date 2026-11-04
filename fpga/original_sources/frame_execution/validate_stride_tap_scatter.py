"""Real-arena scatter checks with independently enumerated convolution taps."""
import argparse,json,random,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,sha,run
from stride_tap_layout import source_coordinate,location
def vectors(root):
 root.mkdir();src=[];gold=[];rows=[]
 shapes=[(1280,720,0,0,8,1,0),(1280,720,632,357,8,1,360),(1280,720,160,3,160,1,0),
 (640,360,0,0,16,2,0),(640,360,240,177,80,2,360),(640,360,80,6,80,2,0)]
 def word(x,y,g,c):return sum(((x*37+y*71+g*19+l*23+c*11)&255)<<(8*l) for l in range(8))
 for ci,(sw,sh,ox,oy,w,g,base) in enumerate(shapes):
  ioff=len(src);goff=len(gold);span=w//4*9*g
  left,top=source_coordinate(sw,sh,ox,oy,0);right,bottom=source_coordinate(sw,sh,ox+w-1,oy+2,8)
  points=[(x,y,c) for y in range(top,bottom+1) for x in range(left,right+1) for c in range(g)]
  random.Random(71+ci).shuffle(points)
  src += [(x<<75)|(y<<65)|(c<<64)|word(x,y,c,ci) for x,y,c in points]
  expected={}
  for r in range(3):
   for x in range(w):
    for c in range(g):
     for tap in range(9):
      sx,sy=source_coordinate(sw,sh,ox+x,oy+r,tap);expected[location(x,r,c,tap,g,base)]=word(sx,sy,c,ci)
  gold += [expected[b,base+a] for b in range(12) for a in range(span)]
  rows.append([ci,sw,sh,ox,oy,w,g==2,base,ioff,len(points),goff,len(gold)-goff,span])
 (root/'source.hex').write_text(''.join(f'{x:024x}\n' for x in src));(root/'gold.hex').write_text(''.join(f'{x:016x}\n' for x in gold))
 (root/'cases.txt').write_text(str(len(rows))+'\n'+''.join(' '.join(str(int(x)) for x in row)+'\n' for row in rows))
 return dict(cases=len(rows),source_words=len(src),gold_words=len(gold),shapes=shapes)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  oracle=vectors(root/'vectors')
  files=[EF/'sim_models/verilog/efx_dpram10.v',HERE/'tensor_arena12.sv',HERE/'stride_tap_scatter.sv',HERE/'tb_stride_tap_scatter.sv'];records=[]
  for f in files:shutil.copy2(f,root/f.name);records.append(dict(path=str(f),snapshot=str(root/f.name),sha256=sha(f)))
  m=dict(inputs=records,oracle=oracle,dependencies={str(f):sha(f) for f in (Path(__file__),HERE/'stride_tap_layout.py',ROOT/'network_ab/model_b_contract.json')},
   vectors={f.name:sha(f) for f in (root/'vectors').iterdir()})
  (root/'inputs.json').write_text(json.dumps(m,indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_stride_tap_scatter','+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
  lines=re.findall(r'^# (PASS STRIDE_TAP_SCATTER.*)$',body,re.M);rows=re.findall(r'^# SCATTER (.*)$',body,re.M);assert len(lines)==1 and len(rows)==oracle['cases']*2
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_REAL_ARENA_STRIDE_TAP_SCATTER_PASS',line=lines[0],rows=rows,seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),producer_rq_connected=False,real_stride_compute_connected=False,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
