"""Bounded32-front port-layout search; no RTL throughput claim.
Each RQ bank owns a contiguous spatial strip, instead of4 detached pieces
advancing32 pixels each round. Enumerates actual replicated/NN tap addresses.
"""
import argparse,json,math
from pathlib import Path
from collections import Counter
def screen(width,nn,stride,pm,cp,layout,columns):
 strip=(width+3)//4;period=9;counts=Counter();peak=0;requests=0;phases=0
 max_epoch=(strip+7)//8 if layout=='strips' else (width+31)//32
 for t in range(max_epoch*9+17):
  qs=[set() for _ in range(3*columns)];jobs=0
  for c in range(4):
   for p in range(8):
    start=(p*pm+c*cp)%9
    if t<start:continue
    epoch,k=divmod(t-start,period)
    x=c*strip+epoch*8+p if layout=='strips' else epoch*32+c*8+p
    if x>=width or (layout=='strips' and epoch*8+p>=strip):continue
    ky,kx=divmod(k,3);sx=max(0,min(width*stride-1,x*stride+kx-1));sy=6+ky-1
    if nn:sx//=2;sy//=2
    bank=sy%3*columns+sx%columns
    qs[bank].add((sy,sx//columns));jobs+=1
  if not jobs:continue
  n=max(map(len,qs));ticks=(n+1)//2
  counts[ticks]+=1;peak=max(peak,n);requests+=sum(map(len,qs));phases+=1
 return dict(read_cycles=sum(k*v for k,v in counts.items()),phases=phases,peak_bank_requests=peak,unique_reads=requests,
  phase_service_histogram=dict(counts),width=width,nn=nn,stride=stride,phase_multiplier=pm,cluster_phase=cp,layout=layout,columns=columns)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
 cases=[]
 for width,nn,stride in ((80,False,1),(160,False,1),(320,False,1),(320,False,2),(640,True,1),(1280,True,1)):
  trials=[]
  for layout in ('linear','strips'):
   for pm in (1,2,4,5,7,8):
    for cp in range(9):trials.append(screen(width,nn,stride,pm,cp,layout,4))
  trials.sort(key=lambda r:(r['read_cycles'],r['peak_bank_requests'],r['unique_reads']))
  cases.append(dict(shape=[width,nn,stride],baseline=screen(width,nn,stride,1,0,'linear',4),best=trials[:6],all_trials=trials))
  print(json.dumps(dict(shape=cases[-1]['shape'],baseline=cases[-1]['baseline'],best=trials[0])),flush=True)
 (a.root/'result.json').write_text(json.dumps(dict(status='DW32_LAYOUT_SEARCH_NOT_RTL',cases=cases,
  controller=False,result_writes_included=False,full_frame=False,performance_pass=False),indent=2))
if __name__=='__main__':main()
