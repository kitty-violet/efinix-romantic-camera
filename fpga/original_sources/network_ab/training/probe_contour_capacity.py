"""Bounded isolated scheduling diagnostic for the contour graph.

The old THREE-plane scheduler is only a virtual command-order builder here.
Its virtual capacity is enlarged; the final SIX real banks remain 1024 words
with every lifetime/address/read-owner check enabled. This is not RTL proof.
"""
import copy,importlib.util,json,sys,traceback
from pathlib import Path
from . import compile_component_row_pair as original
from . import capacity_fixed_headers as headers
from . import compile_b1_capacity as diagnostic
from .contour_region_net import plan,budget
from .train_luma_regions import ROOT,RUN
from .runtime import write_json,file_hash


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod


def aligned_tile(graph,index,x,width):
    byid={n['id']:n for n in graph['nodes']};ranges={'rgb_head':(x,min(x+width,1280)-1)}
    for n in reversed(graph['nodes']):
        a,b=ranges[n['id']];a=a//2*2;ranges[n['id']]=(a,b)
        if n['kind']=='nearest_x2':a,b=a//2,b//2
        elif n['kind'] in ('conv3x3','dw3x3'):a,b=a*n['stride']-1,b*n['stride']+1
        for parent in n['inputs']:
            w=byid[parent]['output_shape_hwc'][1] if parent in byid else 1280;lo,hi=max(0,a),min(w-1,b)
            if parent in ranges:lo,hi=min(lo,ranges[parent][0]),max(hi,ranges[parent][1])
            ranges[parent]=(lo,hi)
    a,b=ranges['input']
    return dict(tile_index=index,commit_interval=dict(x0=x,x1=min(x+width,1280)-1),source_interval=dict(x0=a//4*4,x1=b),
        steps=[dict(operation=n['id'],output_interval=dict(x0=ranges[n['id']][0],x1=ranges[n['id']][1])) for n in graph['nodes']])


def main():
    out=ROOT/'runs'/RUN/'contour-region-allocation-v2';diagnostic.plan=plan;headers.plan=plan;headers.budget=budget;diagnostic.coefficient_image=headers.coefficient_image
    diagnostic.diagnostic(out);sys.path.insert(0,str(original.HW));sources={}
    virtual_source=original.HW/'fourway_frame_program.py';sources[str(virtual_source)]=file_hash(virtual_source)
    text=virtual_source.read_text();assert text.count('1024')==4
    vpath=out/'frozen_virtual_scheduler.py';vpath.write_text(text.replace('1024','8192'));load('fourway_frame_program',vpath)
    source=Path(original.__file__);sources[str(source)]=file_hash(source);text=source.read_text();needle="    assert chosen,('ROW_PAIR_CAPACITY',ti,trials)";assert text.count(needle)==1
    text=text.replace(needle,"    if chosen is None:\n        from .capacity_bank_search import search\n        chosen,report=search(records,local,placement);trials.append(dict(node_bank_search=report))\n"+needle)
    path=out/'frozen_compiler.py';path.write_text(text);compiler=load('network_ab.training._contour_alloc',path);compiler.tile=aligned_tile
    graph=diagnostic.fused_plan();trials=[]
    for width in (64,48,32):
        row=dict(width=width,positions=[])
        try:
            for i,x in enumerate([0,width,1280-width]):
                _,r=compiler.compile_one(graph,i,x,width);row['positions'].append(r);print('CONTOUR_ALLOCATION',width,x,r['high'],flush=True)
            row['status']='THREE_POSITIONS_PASS_NOT_FULL_FRAME'
        except (AssertionError,MemoryError) as e:
            row.update(status='FAIL',error=str(e),traceback=traceback.format_exc());print('CONTOUR_ALLOCATION_FAIL',width,repr(e),flush=True)
        trials.append(row);write_json(out/'allocation.json',dict(budget=budget(),graph=plan(),trials=trials,source_hashes=sources,
            virtual_scheduler_capacity=8192,physical_bank_count=6,physical_bank_depth=1024,
            graph_source_sha256=file_hash(Path(__file__).with_name('contour_region_net.py')),script_sha256=file_hash(__file__),synthetic_parameters=True,rtl=False,fit=False,timing=False))
        if row['status'].startswith('THREE'):break
    assert all(file_hash(p)==h for p,h in sources.items())


if __name__=='__main__':main()
