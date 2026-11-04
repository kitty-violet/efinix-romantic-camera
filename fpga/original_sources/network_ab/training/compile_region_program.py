"""Bounded 16-bit descriptor ABI for the learned categorical graph.

The old B1 image/descriptor ABI is not silently repurposed. This package has
new explicit schemas and nonoverlapping DDR windows; production remains
unchanged. No host upload, frame-rate, board fit or STA pass is implied.
"""
import argparse,copy,json,random,sys,zlib
from collections import Counter
from functools import lru_cache
from pathlib import Path
from .runtime import file_hash,write_json
from .region_detail_target import ROOT,RUN
from .validate_region_native import coefficient_image,pack
from .audit_region_class_schedule import graph
from .probe_boundary_budget import HW,tile

CWIDTHS={1:5,2:9,3:3,4:3,5:3,6:6,7:11,8:11,9:11,10:12,11:32,12:28,13:8,14:32,15:8,16:15,17:15,18:15,19:15,20:8,21:32,22:8,23:11,24:18,25:18,28:8,29:29,30:10,31:16}
DWIDTHS={0:1,1:5,2:9,3:3,4:2,5:10,6:1,7:28,8:1,9:10,10:11,30:10,31:16}
CSCHEMA=[('peer_completed',16),('program_index',16)]+[(f'f{i}',n) for i,n in CWIDTHS.items()]
DSCHEMA=[('peer_completed',16)]+[(f'f{i}',n) for i,n in DWIDTHS.items()]+[('region',1)]


def offsets(schema):
    offset=0;result={}
    for key,n in schema:result[key]=(offset,n);offset+=n
    return result,offset


def encode(values,schema,size):
    result=0;offset=0
    for key,n in schema:
        value=values[key];assert 0<=value<1<<n,(key,value,n);result|=value<<offset;offset+=n
    assert offset<=size*8
    # Independent extraction of every serialized field.
    pos,_=offsets(schema)
    assert all((result>>o)&((1<<n)-1)==values[k] for k,(o,n) in pos.items())
    return result.to_bytes(size,'little')


def reader_compatible(q):
    if q['kind']!='compute':return True
    ps=[r['plane'] for r in q['inputs']];ordinal=[sum(ps[k]!=ps[k-1] for k in range(1,j+1)) for j in range(len(ps))]
    slots=[2*p+int(o>=3) for p,o in zip(ps,ordinal)]
    return not any(s%2 for s in slots[:3]) and not(len(slots)>5 and slots[5] in (0,4))


def normalize(q,check_reader=True):
    q=copy.deepcopy(q)
    for key in ['inputs','outputs','skip']:
        if key in q:
            for r in q[key]:r['plane']=(r['plane']+2)%3
    if 'row' in q:q['row']['plane']=(q['row']['plane']+2)%3
    if q['kind']=='compute':
        for r in q['skip']:
            d=q['output_x']-r['origin'];assert d>=0 and d%4 in (0,2);shift=d//4*4
            r.update(base=r['base']+shift//4*r['groups'],origin=r['origin']+shift,width=r['width']-shift,delta=d%4)
        if check_reader:assert reader_compatible(q)
    return q


def reader_commands(raw,allow_split=False):
    q=normalize(raw,not allow_split)
    if reader_compatible(q):return [q]
    # Only the top/bottom NN aliases need this. Two smaller actual commands
    # cover the identical output rows, keeping the same physical SRAM bases.
    # Do not add unused row-5 mux inputs to the existing hardware reader.
    assert allow_split and q['nn'] and q['operator']=='dw3x3' and not q['skip'] and q['rows']==4
    result=[]
    for y in (0,2):
        r=copy.deepcopy(q);r.update(rows=2,inputs=q['inputs'][y:y+4],outputs=q['outputs'][y:y+2])
        assert len(r['inputs'])==4 and len(r['outputs'])==2 and reader_compatible(r);result.append(r)
    assert [o for r in result for o in r['outputs']]==q['outputs']
    return result


def prepare(a):
    sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    from prepare_concurrent_frame import footprint
    root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);package=json.loads(a.package.read_text())
    image_writer=coefficient_image
    if a.fixed_headers:
        from .region_fixed_headers import coefficient_image as image_writer
    if a.separated_headers:
        from .region_separated_headers import coefficient_image as image_writer
    ops=image_writer(package,root)
    graph_factory=graph;width=64;index_bits=16;dma_address=0x04300000;coeff_address=0x04400000
    if a.context:
        from .context_region_net import fused_plan as graph_factory
        width=a.width or 32;assert width%4==0 and 4<=width<=(128 if a.components else 64)
        index_bits=17;dma_address=0x04500000;coeff_address=0x04600000
        if a.native:
            from .skip_filter_net import fused_plan as graph_factory
        if a.coarse:
            from .coarse_skip_net import fused_plan as graph_factory
        if a.surface32:
            from .surface32_net import fused_plan as graph_factory
        if a.components:
            from .component_region_net import fused_plan as graph_factory
    cschema=[(k,index_bits if k in ('peer_completed','program_index','f31') else n) for k,n in CSCHEMA]
    dschema=[(k,index_bits if k in ('peer_completed','f31') else n) for k,n in DSCHEMA]
    nodes=package['plan']['nodes'];indices={n['id']:i for i,n in enumerate(nodes)};plan=graph_factory();commands=[];allocation=[]
    fields=[];last={};dma_pixels=0;totals=Counter();macs=0
    for ti,x in enumerate(range(0,1280,width)):
        p=compile_tile(plan,dict(name='research_luma'),tile(plan,ti,x,width),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        allocation.append(dict(x=x,high=p['finite_high_water']))
        normalized=[q for raw in p['commands'] for q in reader_commands(raw,a.context)]
        for q in normalized:
            i=len(commands);f=[0]*32;f[30]=ti;f[31]=i
            if q['kind']!='compute':
                r=q['row'];store=q['kind']=='store';f[:11]=[int(store),indices['rgb'],r['width'],r['groups'],r['plane'],r['base'],int(a.components or not store),
                    (0x800000 if store else 0)+(r['y']*1280+r['origin'])*3,int(store),r['y'],r['origin']]
                assert q['channels']==(32 if store and not a.components else 3);dma_pixels+=r['width'] if store else 0
            else:
                name='rgb' if q['node']=='rgb_head' else q['node'];parameter='rgb' if q['parameter_node']=='rgb_head' else q['parameter_node'];op=ops[parameter]
                res=package['residual'][name] if q['skip'] else dict(main_scaled=False,multiplier=0,post_multiplier=0)
                fl=int(q['operator'] in ('pw1x1','pw_fused'))|int(q['nn'])<<1|int(q['operator']=='conv3x3')<<2|int(q['stride2'])<<3|int(bool(q['skip']))<<4|int(res['main_scaled'])<<5
                ib=pack([r['base'] for r in q['inputs']],10);ob=pack([r['base'] for r in q['outputs']],10);sb=pack([r['base'] for r in q['skip']],10)
                f[:26]=[2,indices[name],q['pixels'],q['rows'],q['input_groups'],q['output_groups'],fl,q['output_x'],q['source_width'],q['inputs'][0]['origin'],
                    pack([r['plane'] for r in q['inputs']],2),ib&0xffffffff,ib>>32,pack([r['plane'] for r in q['outputs']],2),ob&0xffffffff,ob>>32,
                    op['weight_offset'],op['bias_offset'],op['multiplier_offset'],op['relu_offset'],pack([r['plane'] for r in q['skip']],2),sb&0xffffffff,sb>>32,
                    q['skip'][0]['delta'] if q['skip'] else 0,res['multiplier'],res['post_multiplier']]
                f[28]=q['mask'];last[indices[name]]=i
                if q['skip']:last[indices[parameter]]=i
                if q['nn']:last[indices[next(n for n in nodes if n['id']==name)['inputs'][0]]]=i
                pn=next(n for n in nodes if n['id']==parameter);macs+=q['pixels']*q['rows']*pn['channels']*pn['terms']
            totals[q['kind']]+=1;commands.append(q);fields.append(f)
        print('REGION_PROGRAM_STRIPE',ti+1,flush=True)
    assert set(last)==set(range(len(nodes))) and dma_pixels==1280*720
    for node,i in last.items():fields[i][29]|=1<<node
    counts=[0,0];ids=[[],[]];deps=[[],[]];read=[[0]*12288 for _ in range(2)];write=[[0]*12288 for _ in range(2)]
    for i,q in enumerate(commands):
        e=int(q['kind']!='compute');other=1-e;rr,ww=footprint(q)
        dep=max([0]+[write[other][j] for j in rr]+[max(read[other][j],write[other][j]) for j in ww])
        counts[e]+=1;ids[e].append(i);deps[e].append(dep)
        for j in rr:read[e][j]=counts[e]
        for j in ww:write[e][j]=counts[e]
    assert max(counts)<1<<index_bits and len(commands)<1<<index_bits
    @lru_cache(maxsize=64)
    def fp(i):return footprint(commands[i])
    interleavings=[]
    for seed in (7,71,193):
        rng=random.Random(seed);done=[0,0];live=[None,None];both=0;ticks=0
        while done!=counts:
            for e in (0,1):
                if live[e] is None and done[e]<counts[e] and deps[e][done[e]]<=done[1-e]:live[e]=ids[e][done[e]]
            if all(i is not None for i in live):
                ar,aw=fp(live[0]);br,bw=fp(live[1]);assert not (aw&(br|bw) or bw&ar);both+=1
            assert any(i is not None for i in live),done
            e=rng.randrange(2)
            if live[e] is not None:done[e]+=1;live[e]=None
            ticks+=1
        interleavings.append(dict(seed=seed,overlapping_pairs=both,events=ticks))
    streams=[]
    for e,(label,schema,size,address) in enumerate([('compute',cschema,64,0x04000000),('dma',dschema,16,dma_address)]):
        payload=bytearray()
        for index,dep in zip(ids[e],deps[e]):
            f=fields[index];v=dict(peer_completed=dep,program_index=index,region=int(f[0]==1))
            v.update({f'f{k}':value for k,value in enumerate(f)});payload.extend(encode(v,schema,size))
        path=root/(label+'.bin');path.write_bytes(payload)
        (root/(label+'.hex')).write_text(''.join(f'{int.from_bytes(payload[i:i+size],"little"):0{size*2}x}\n' for i in range(0,len(payload),size)))
        streams.append(dict(name=label,count=len(ids[e]),bytes=len(payload),address=address,record_bytes=size,schema=schema,used_bits=offsets(schema)[1],sha256=file_hash(path),crc32=zlib.crc32(payload)))
    windows=[(s['address'],s['address']+s['bytes']) for s in streams]+[(coeff_address,coeff_address+0x5000)]
    assert all(a<b and 0x04000000<=a<b<=0x05000000 for a,b in windows)
    assert all(b<=c or d<=a for i,(a,b) in enumerate(windows) for c,d in windows[i+1:])
    bound=[Path(__file__),HW/'fourway_frame_program.py',HW/'compile_frame.py',HW/'prepare_concurrent_frame.py',Path(__file__).with_name('validate_region_native.py')]
    if a.fixed_headers:bound.append(Path(__file__).with_name('region_fixed_headers.py'))
    if a.context:bound.append(Path(__file__).with_name('context_region_net.py'))
    if a.native:bound.append(Path(__file__).with_name('skip_filter_net.py'))
    if a.coarse:bound.extend(Path(__file__).with_name(k) for k in ('coarse_skip_net.py','wide_context_net.py','skip_filter_net.py'))
    if a.surface32:bound.extend(Path(__file__).with_name(k) for k in ('surface32_net.py','coarse_skip_net.py','wide_context_net.py','skip_filter_net.py'))
    if a.separated_headers:bound.append(Path(__file__).with_name('region_separated_headers.py'))
    if a.components:bound.extend(Path(__file__).with_name(k) for k in ('component_region_net.py','coarse_skip_net.py','wide_context_net.py','skip_filter_net.py'))
    frozen=root/'frozen_sources';frozen.mkdir()
    for f in bound:(frozen/f.name).write_bytes(f.read_bytes())
    result=dict(status='FULL_FRAME_NEW_ABI_OFFLINE_CHECKED_NOT_BOARD_RELEASE',package=str(a.package),package_sha256=file_hash(a.package),source_hashes={str(p):file_hash(p) for p in bound},
        commands=len(commands),counts=dict(totals),nodes=len(nodes),complete_node_mask=(1<<len(nodes))-1,allocation=allocation,scheduled_macs=macs,
        streams=streams,coefficient_address=coeff_address,randomized_interleavings=interleavings,final_rgb_pixels=dma_pixels,production_B1_compatible=False,hardware_issuer=False,fixed_header_windows=a.fixed_headers,
        stripe_width=width,index_bits=index_bits,context=a.context,native=a.native,coarse=a.coarse,surface32=a.surface32,separated_headers=a.separated_headers,components=a.components,region_format='YCoCg3' if a.components else 'argmax32')
    write_json(root/'program.json',result);print('REGION_PROGRAM_PASS',len(commands),counts,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True,type=Path);p.add_argument('--name',required=True);p.add_argument('--fixed-headers',action='store_true');p.add_argument('--context',action='store_true');p.add_argument('--native',action='store_true');p.add_argument('--coarse',action='store_true');p.add_argument('--surface32',action='store_true');p.add_argument('--components',action='store_true');p.add_argument('--separated-headers',action='store_true');p.add_argument('--width',type=int);a=p.parse_args()
    if a.native or a.coarse or a.surface32 or a.components:a.context=True
    if a.separated_headers:a.fixed_headers=True
    assert sum([a.native,a.coarse,a.surface32,a.components])<=1
    prepare(a)


if __name__=='__main__':main()
