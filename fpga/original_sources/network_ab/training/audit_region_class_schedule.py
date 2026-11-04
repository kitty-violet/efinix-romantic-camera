"""All-stripe finite allocation accounting. No RTL/control/STA pass implied."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys
from .train_region_classes import ROOT,RUN,class_plan
from .probe_boundary_budget import HW,tile
from .runtime import file_hash,write_json


def graph():
    nodes=[]
    for row in class_plan()['nodes']:
        n=copy.deepcopy(row)
        if n['kind']=='residual_add':
            p=nodes.pop();n.update(kind='pw_fused',parameter_node=p['id'],inputs=[p['inputs'][0],n['inputs'][1]],
                input_channels=p['input_channels'],input_shape_hwc=p['input_shape_hwc'],terms=p['terms'],macs=p['macs'])
        nodes.append(n)
    nodes[-1]['id']='rgb_head';return dict(nodes=nodes)


def main():
    sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    out=ROOT/'runs'/RUN/'classes32';assert not (out/'full_allocation.json').exists()
    hashes={str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('train_region_classes.py'),HW/'fourway_frame_program.py',HW/'compile_frame.py']}
    plan=graph();byid={n['id']:n for n in plan['nodes']};records=[];totals=Counter();macs=0;dma=Counter()
    for i,x in enumerate(range(0,1280,64)):
        p=compile_tile(plan,dict(name='research_class32'),tile(plan,i,x,64),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        c=Counter(q['kind'] for q in p['commands']);totals.update(c)
        for q in p['commands']:
            if q['kind']=='compute':
                n=byid[q['node']];macs+=q['pixels']*q['rows']*n['channels']*n['terms']
            else:dma[q['kind']]+=q['row']['width']*q['channels']
        records.append(dict(x=x,high=p['finite_high_water'],peak=p['peak_live_padded_bytes'],commands=dict(c)))
        if i%5==4:print('CLASS_ALLOC_STRIPES',i+1,20,flush=True)
    assert all(file_hash(p)==h for p,h in hashes.items())
    result=dict(status='FULL_FRAME_FINITE_ALLOCATION_ONLY',source_hashes=hashes,records=records,
        commands=dict(totals),command_bytes=sum(totals.values())*128,scheduled_macs=macs,ideal_macs=class_plan()['macs'],
        naive_dma_bytes=dict(dma),physical_banks=12,depth_words=1024,word_bits=64,
        hypothetical_palette_stream_rgb_bytes=1280*720*3,production_abi_compatible=False,
        limitations=['Commands are for raw32 logits. Existing RGB writer cannot execute this output format.',
            'A streamed argmax32/palette writer could avoid raw logit DDR, but is NOT implemented or measured here.',
            'No coefficient addresses, Q17, cycle-level route simulation, DDR timing, resource fit or STA verification.',
            'More descriptors and halos than B1. Lower ideal MAC does not establish required FPS.'])
    write_json(out/'full_allocation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_hashes')}),flush=True)


if __name__=='__main__':main()
