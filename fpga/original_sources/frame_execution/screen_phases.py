"""Bounded phase/job permutation screen; model only, no performance PASS."""
import argparse,json,shutil,time
from pathlib import Path
from phase_search_model import pipeline
from run_units import HERE,ROOT,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 paths=[HERE/'phase_search_model.py',Path(__file__)];inputs=[]
 for src in paths:
  shutil.copy2(src,root/src.name);inputs.append(dict(path=str(src),sha256=sha(src)))
 profiles=dict(dw=(80,2,4,320,10000,1,80,79,12,False,False),
               nn=(320,2,1,1280,10000,1,320,159,12,True,False),
               stride2=(160,2,2,1280,10000,2,160,319,12,False,False),
               stem=(320,2,4,1280,10000,1,320,319,12,False,True))
 rows=[]
 for inter in (False,True):
  for mul in (1,2,4,5,7,8):
   for cp in range(9):
    r=dict(interleave=inter,phase_mul=mul,cluster_phase=cp,profiles={})
    for name,args in profiles.items():
     x=pipeline(*args,phase_mul=mul,cluster_phase=cp,interleave=inter)
     r['profiles'][name]={k:x[k] for k in ('cycles','bank_conflict_cycles','slot_stall_cycles','word_reads')}
    rows.append(r)
 for name in profiles:
  best=sorted(rows,key=lambda r:r['profiles'][name]['cycles'])[:8]
  print(name,json.dumps(best),flush=True)
 for src,r in zip(paths,inputs):assert sha(src)==r['sha256']
 (root/'result.json').write_text(json.dumps(dict(status='DESIGN_SEARCH_NOT_RTL',performance_pass=False,inputs=inputs,cases=rows),indent=2))
if __name__=='__main__':main()
