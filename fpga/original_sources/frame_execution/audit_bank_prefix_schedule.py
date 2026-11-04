"""Exhaustively enumerate legal fixed-dispatch owner orders within one command.
Per-bank there is at most one LAST in a phase. Handshake stalls only insert
gaps; they cannot reorder these LAST phases. Commands reset the history.
"""
import json,hashlib
from pathlib import Path
from run_units import ROOT,HERE,sha
def stream(mode,pixels,gi=1,go=1):
 pair=mode in ('stride','rgb');rgb=mode=='rgb';pw=mode=='pw'
 units=(pixels+1)//2 if rgb else pixels
 term=8*gi if pw else 8 if rgb else 9
 period=term*go if pw else term
 size=9 if mode=='stride' else 8
 batches=(units+size-1)//size
 open_=[False]*size;batch=0;phase=0
 for cycle in range((batches+2)*period+size):
  for owner in range(size):
   age=(phase-owner)%period
   first=age%term==0;last=age%term==term-1
   launch=batch<batches and batch*size+owner<units
   need=launch if first else open_[owner]
   if first:open_[owner]=launch
   if last and need:
    yield owner//8,owner%8
    open_[owner]=False
  phase+=1
  if phase==period:phase=0;batch+=1
  if batch>=batches and not any(open_):return
 raise AssertionError('Non-terminating fixed schedule')
def main():
 configs=returns=reuses=0;max_distance=0;coverage={};trace=hashlib.sha256()
 for mode in ('pw','dw','stem','rgb','stride'):
  count=0
  for pixels in range(1,161 if mode=='stride' else 321):
   for gi in ((1,2,4) if mode=='pw' else (1,)):
    for go in ((1,2,4) if mode=='pw' else (1,)):
     seen=[{},{}];index=[0,0];actual=0
     for bank,owner in stream(mode,pixels,gi,go):
      if owner in seen[bank]:
       distance=index[bank]-seen[bank][owner]
       assert 1<=distance<=8,(mode,pixels,gi,go,bank,owner,distance)
       max_distance=max(max_distance,distance);reuses+=1
      seen[bank][owner]=index[bank];index[bank]+=1;returns+=1;actual+=1
      trace.update(bytes((bank,owner)))
     expected=((pixels+1)//2 if mode=='rgb' else pixels)*(go if mode=='pw' else 1)
     assert actual==expected,(mode,pixels,gi,go,actual,expected)
     configs+=1;count+=1
  coverage[mode]=count
 result=dict(status='BANK_PREFIX_SCHEDULE_DISTANCE_PASS',configurations=configs,mac_returns=returns,reuses=reuses,
  max_return_distance=max_distance,coverage=coverage,order_sha256=trace.hexdigest(),
  scope='All legal widths1..320 (stride1..160), PW groups1/2/4. Rows repeat the same independent bank order; stem holds each phase for three subphases. Gaps and bypass do not age history. Each command clears prefix state.',
  sources=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),ROOT/'tmp/frame-execution/connected-board-shared-taps1/assembly/unified_column_dispatch32.sv',HERE/'variants/bank_prefix_history/compute_column_cluster.sv')])
 root=ROOT/'tmp/frame-execution/bank-prefix-schedule1';root.mkdir(exist_ok=False)
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
