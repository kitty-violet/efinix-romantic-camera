"""New RGB graph finite allocation and command budgeting, read-only hardware."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from .detail_rgb_net import plan,fused_plan
from .region_detail_target import ROOT,RUN
from .probe_boundary_budget import HW,tile
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args()
    sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    out=ROOT/'runs'/RUN/'schedule';out.mkdir(exist_ok=True)
    path=out/('full.json' if a.full else 'screen.json');assert not path.exists()
    g=fused_plan();byid={n['id']:n for n in g['nodes']};attempts=[];chosen=None;records=[];totals=Counter();macs=0;dma=Counter()
    sources=[Path(__file__),Path(__file__).with_name('detail_rgb_net.py'),HW/'fourway_frame_program.py',HW/'compile_frame.py']
    binding={str(p):file_hash(p) for p in sources}
    if a.full:chosen=json.loads((out/'screen.json').read_text())['width'];assert chosen
    else:
        for width in (128,64,32):
            trial=dict(width=width,positions=[]);start=time.perf_counter()
            for j,x in enumerate((0,512,1280-width)):
                try:
                    v=compile_tile(g,dict(name='detail_rgb'),tile(g,j,x,width),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
                    trial['positions'].append(dict(x=x,high=v['finite_high_water'],commands=dict(Counter(q['kind'] for q in v['commands']))))
                except (MemoryError,AssertionError) as e:trial['failure']=str(e);break
            trial['seconds']=time.perf_counter()-start;attempts.append(trial);print('DETAIL_ALLOC',json.dumps(trial),flush=True)
            if 'failure' not in trial:chosen=width;break
    if a.full:
        for i,x in enumerate(range(0,1280,chosen)):
            v=compile_tile(g,dict(name='detail_rgb'),tile(g,i,x,chosen),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
            counts=Counter(q['kind'] for q in v['commands']);totals.update(counts)
            for q in v['commands']:
                if q['kind']=='compute':
                    n=byid[q['node']];macs+=q['pixels']*q['rows']*n['channels']*n['terms']
                else:dma[q['kind']]+=q['row']['width']*q['channels']
            records.append(dict(x=x,high=v['finite_high_water'],commands=dict(counts)))
            if i==0:write_json(out/'first_stripe.json',dict(plan=g,program=v))
            if (i+1)%5==0:print('DETAIL_FULL_STRIPES',i+1,flush=True)
    assert all(file_hash(p)==h for p,h in binding.items())
    result=dict(status='FINITE_ALLOCATION_ONLY' if chosen else 'REJECTED_CAPACITY',width=chosen,attempts=attempts,records=records,
        commands=dict(totals),scheduled_macs=macs,ideal_macs=plan()['macs'],dma_bytes=dict(dma),binding=binding,
        rgb_output_compatible=True,production_graph_abi_compatible=False,physical_fit=False,rtl_verified=False,timing_verified=False)
    write_json(path,result);print('DETAIL_SCHEDULE_DONE',chosen,len(records),flush=True)

if __name__=='__main__':main()
