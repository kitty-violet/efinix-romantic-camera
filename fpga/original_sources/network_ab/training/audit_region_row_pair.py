"""Read-only compatibility screen against the hardware stream's new SRAM idea.

All output stays in model-training. This checks inclusive row lifetimes and
padding only. It neither modifies their code nor claims a new executable ABI.
"""
import argparse,copy,json,random,sys,time
from pathlib import Path
from .coarse_skip_net import fused_plan
from .probe_boundary_budget import HW,tile
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='coarse64-row-pair-screen-v2');a=p.parse_args()
    sys.path.insert(0,str(HW));from screen_row_pair_layout import problem,signatures,score
    from screen_row_banked_arena import place
    files=[HW/k for k in ['screen_row_pair_layout.py','screen_row_banked_arena.py','fourway_frame_program.py','compile_frame.py']]+[Path(__file__),Path(__file__).with_name('coarse_skip_net.py')]
    hashes={str(p):file_hash(p) for p in files};root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);plan=fused_plan();rows=[];start=time.perf_counter();chosen=None
    for i,x in enumerate(range(0,1280,64)):
        local,tasks,records=problem(plan,dict(name='research_coarse_skip'),tile(plan,i,x,64));best=None
        candidates=[chosen] if chosen else []
        for step in range(6):
            offsets={'source':0}
            for name,n in local.items():
                if name!='source':offsets[name]=(offsets[n['parents'][0]]+(0 if n['kind']=='nearest_x2' else step))%6
            candidates.append(offsets)
        for offsets in candidates:
            for (name,y),r in records.items():r['bank']=(y+offsets[name])%6
            high,passed=place(records,1024)
            if best is None or max(high)<max(best['high']):best=dict(offsets=offsets,high=high,passed=passed)
            if passed:break
        if not best['passed']:
            names=[n for n in local if any(k[0]==n for k in records)];sigs=signatures(records,names,len(tasks));rng=random.Random(2026100445)
            for trial in range(4):
                offset=[best['offsets'][n] for n in names] if trial==0 else [rng.randrange(6) for _ in names]
                for _ in range(2):
                    for k in range(len(names)):
                        options=[]
                        for v in range(6):offset[k]=v;options.append((score(sigs,offset),v))
                        offset[k]=min(options)[1]
                offsets=dict(zip(names,offset))
                for (name,y),r in records.items():r['bank']=(y+offsets[name])%6
                high,passed=place(records,1024)
                if max(high)<max(best['high']):best=dict(offsets=offsets,high=high,passed=passed)
                if passed:break
        if best['passed']:chosen=best['offsets']
        rows.append(dict(x=x,rows=len(records),**best));print('REGION_ROW_PAIR_SCREEN',x,best,flush=True)
        if not best['passed']:break
    assert all(file_hash(k)==v for k,v in hashes.items())
    write_json(root/'result.json',dict(status='PADDING_LIFETIME_CAPACITY_SCREEN_ONLY',all_stripes_fit=len(rows)==20 and all(r['passed'] for r in rows),stripes=rows,seconds=time.perf_counter()-start,source_hashes=hashes,
        executable_descriptors=False,rtl=False,fit=False,timing=False,
        open_integration=['C32 categorical head is a normal PW, not the special C3 RGB pair mode assumed by the B1 migration script',
            'Category DMA decoding still needs correct six-bank row-pair read mapping',
            'Fused C8/C16 skip and halfword phase must be verified under the new memory interface',
            'Final production data path and ABI from hardware workflow remain under development']))


if __name__=='__main__':main()
