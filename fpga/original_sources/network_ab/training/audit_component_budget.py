"""Check the C3 reduction against actual finite allocation and scheduled MACs."""
import json,sys
from collections import Counter
from pathlib import Path
from .component_region_net import plan,fused_plan
from .probe_boundary_budget import HW,tile
from .train_luma_regions import ROOT,RUN
from .runtime import file_hash,write_json


def main():
    out=ROOT/'runs'/RUN/'component-budget-v1';out.mkdir(exist_ok=False);sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    g=fused_plan();nodes={n['id']:n for n in g['nodes']};attempts=[];chosen=None
    def compile_one(i,x,width):return compile_tile(g,dict(name='research_component'),tile(g,i,x,min(width,1280-x)),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
    for width in [128,112,96,80,64]:
        attempt=dict(width=width,positions=[])
        try:
            for i,x in enumerate([0,width,1280-width]):
                r=compile_one(i,x,width);attempt['positions'].append(dict(x=x,high=r['finite_high_water']))
            chosen=width
        except (AssertionError,MemoryError) as e:attempt['error']=str(e)
        attempts.append(attempt);print('COMPONENT_BUDGET_TRIAL',attempt,flush=True)
        if chosen:break
    assert chosen;rows=[];counts=Counter();macs=0
    for i,x in enumerate(range(0,1280,chosen)):
        r=compile_one(i,x,chosen);rows.append(dict(x=x,high=r['finite_high_water']));counts.update(q['kind'] for q in r['commands'])
        for q in r['commands']:
            if q['kind']=='compute':n=nodes[q['node']];macs+=q['pixels']*q['rows']*n['channels']*n['terms']
        print('COMPONENT_BUDGET_STRIPE',i+1,flush=True)
    write_json(out/'result.json',dict(status='COMPONENT_FINITE_ALLOCATION_ONLY',width=chosen,stripes=rows,attempts=attempts,counts=dict(counts),ideal_macs=plan()['macs'],scheduled_macs=macs,source_sha256=file_hash(__file__),graph_source_sha256=file_hash(Path(__file__).with_name('component_region_net.py')),rtl=False,fit=False,timing=False))


if __name__=='__main__':main()
