"""C3 graph on six real 128-bit banks; explicit independent descriptor ABI.

The production tree is read only. Inclusive actual command lifetimes, padding,
semantic last writers and randomized cross-engine hazards are checked here.
This is a compiler proof, not RTL/fit/STA evidence.
"""
import argparse,copy,gzip,json,random,sys,time,zlib
from collections import Counter
from pathlib import Path
from .component_region_net import fused_plan
from .compile_region_program import CWIDTHS,DWIDTHS,encode,offsets
from .region_fixed_headers import coefficient_image
from .validate_region_native import pack
from .probe_boundary_budget import HW,tile
from .train_luma_regions import ROOT,RUN
from .runtime import file_hash,write_json


def placement(records,depth=1024):
    neighbors={k:set() for k in records};active=[set() for _ in range(6)]
    for k,r in sorted(records.items(),key=lambda kv:kv[1]['born']):
        b=r['bank'];active[b]={q for q in active[b] if records[q]['last']>=r['born']}
        neighbors[k].update(active[b])
        for q in active[b]:neighbors[q].add(k)
        active[b].add(k)
    best=None
    for order in (lambda k:(-records[k]['length'],records[k]['born'],k),lambda k:(-(records[k]['last']-records[k]['born']),-records[k]['length'],k),lambda k:(records[k]['born'],k)):
        positions={};high=[0]*6
        for k in sorted(records,key=order):
            r=records[k];at=0
            for lo,hi in sorted((positions[q],positions[q]+records[q]['length']) for q in neighbors[k] if q in positions):
                if lo-at>=r['length']:break
                at=max(at,hi)
            positions[k]=at;high[r['bank']]=max(high[r['bank']],at+r['length'])
        if best is None or max(high)<max(best[0]):best=(high,positions)
    high,pos=best
    if max(high)>depth:return high,None
    for k,peers in neighbors.items():
        for q in peers:assert pos[k]+records[k]['length']<=pos[q] or pos[q]+records[q]['length']<=pos[k]
    return high,pos


def compile_one(graph,ti,x,width):
    from fourway_frame_program import compile_tile
    from screen_row_pair_layout import problem
    from compile_row_pair_program import packets
    group=dict(name='research_component');spec=tile(graph,ti,x,width)
    raw=compile_tile(graph,group,spec,0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)['commands']
    local,_,records=problem(graph,group,spec);spans={}
    def physical(n):
        while local[n]['kind']=='nearest_x2':n=local[n]['parents'][0]
        return n
    def touch(n,d,i):
        key=(physical(n),d['y']);assert key in records
        if key not in spans:spans[key]=[i,i]
        else:spans[key][1]=i
    for i,c in enumerate(raw):
        if c['kind']!='compute':touch('source' if c['kind']=='load' else c['final'],c['row'],i)
        else:
            parents=local[c['node']]['parents']
            for key,n in [('inputs',parents[0]),('outputs',c['node']),('skip',parents[1] if len(parents)>1 else parents[0])]:
                for d in c[key]:touch(n,d,i)
    assert set(spans)==set(records)
    for k,r in records.items():r['born'],r['last']=spans[k]
    trials=[];chosen=None
    for step in range(6):
        ofs={'source':0}
        for n,v in local.items():
            if n!='source':ofs[n]=(ofs[v['parents'][0]]+(0 if v['kind']=='nearest_x2' else step))%6
        for (n,y),r in records.items():r['bank']=(y+ofs[n])%6
        high,pos=placement(records);trials.append(dict(step=step,high=high))
        if pos is not None:chosen=(ofs,high,pos);break
    assert chosen,('ROW_PAIR_CAPACITY',ti,trials)
    ofs,high,pos=chosen
    for k,r in records.items():r['base']=pos[k]
    result=[];checked=padding=0
    for c in raw:
        out=copy.deepcopy(c)
        def desc(n,d,output=False,skip=False):
            r=records[physical(n),d['y']];v={k:value for k,value in d.items() if k not in ('plane','base')};v.update(bank=r['bank'],base=r['base'],phase=r['phase'])
            if output:
                x0=c['output_x']//(2 if c['stride2'] else 1);delta=x0-d['origin'];assert delta>=0
                v['base']+=((delta+v['phase'])//2)*v['groups'];v['phase']=(delta+v['phase'])%2;v['origin']=x0
            if skip:
                delta=c['output_x']-v['origin'];assert delta>=0 and delta%4 in (0,2) and v['phase']==0
                advance=delta//4*4;v['base']+=advance//2*v['groups'];v['origin']+=advance;v['width']-=advance;v['delta']=delta%4
            return v
        if c['kind']!='compute':
            out['row']=desc('source' if c['kind']=='load' else c['final'],c['row']);assert out['row']['phase']==0
        else:
            parents=local[c['node']]['parents'];out['inputs']=[desc(parents[0],d) for d in c['inputs']];out['outputs']=[desc(c['node'],d,True) for d in c['outputs']]
            out['skip']=[desc(parents[1],d,skip=True) for d in c['skip']]
            assert len({d['phase'] for d in out['inputs']})==1
            bank_rows={}
            for d in out['inputs']:
                row=(d['y'],d['base']);assert d['bank'] not in bank_rows or bank_rows[d['bank']]==row;bank_rows[d['bank']]=row
            assert len({d['bank'] for d in out['outputs']})==len(out['outputs'])
            for a,b,g in packets(c):
                a=max(0,min(c['source_width']-1,a));b=max(0,min(c['source_width']-1,b))
                if c['nn']:a//=2;b//=2
                for d in out['inputs']:
                    la,lb=a-d['origin'],b-d['origin'];assert 0<=la<d['width']
                    if not 0<=lb<d['width']:assert c['stride2'] and lb==d['width'];padding+=1
                    aa=d['base']+((la+d['phase'])//2)*d['groups']+g;bb=d['base']+((lb+d['phase'])//2)*d['groups']+g
                    assert 0<=aa<1024 and aa==bb and aa<d['base']+((d['width']+d['phase']+1)//2)*d['groups']
                    checked+=2
        result.append(out)
    # Independent logical identity audit in actual compiler command order.
    owners=[None]*12288;reads=writes=0
    def access(n,d,x0,count,write):
        nonlocal reads,writes
        n=physical(n)
        for x0 in range(x0,x0+count):
            assert 0<=x0<d['width']
            for g in range(d['groups']):
                u=d['bank']*2048+2*(d['base']+((x0+d['phase'])//2)*d['groups']+g)+(x0+d['phase'])%2;assert 0<=u<12288
                expected=(n,d['y'],d['origin']+x0,g)
                if write:owners[u]=expected;writes+=1
                else:assert owners[u]==expected,(ti,n,d,x0,expected,owners[u]);reads+=1
    for c in result:
        if c['kind']!='compute':access('source' if c['kind']=='load' else c['final'],c['row'],0,c['row']['width'],c['kind']=='load')
        else:
            parents=local[c['node']]['parents'];pw=c['operator'] in ('pw1x1','pw_fused');lo=max(0,c['output_x']-(0 if pw else 1));hi=min(c['source_width']-1,c['output_x']+(c['pixels']-1 if pw else 2*c['pixels']-1 if c['stride2'] else c['pixels']))
            if c['nn']:lo//=2;hi//=2
            for d in c['inputs']:access(parents[0],d,lo-d['origin'],hi-lo+1,False)
            for d in c['skip']:access(parents[1],d,d['delta'],c['pixels'],False)
            for d in c['outputs']:access(c['node'],d,0,c['pixels'],True)
    return result,dict(tile=ti,x=x,width=width,high=high,offsets=ofs,pair_scalar_checks=checked,padding_halves=padding,semantic_reads=reads,semantic_writes=writes,trials=trials)


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=64);a=p.parse_args()
    sys.path.insert(0,str(HW));from compile_row_pair_descriptors import access,replay
    start=time.perf_counter();out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False);package=json.loads(a.package.read_text());ops=coefficient_image(package,out)
    graph=fused_plan();nodes=package['plan']['nodes'];indices={n['id']:i for i,n in enumerate(nodes)};commands=[];fields=[];last={};tiles=[];macs=0;totals=Counter()
    for ti,x in enumerate(range(0,1280,a.width)):
        cc,meta=compile_one(graph,ti,x,min(a.width,1280-x));tiles.append(meta)
        for c in cc:
            i=len(commands);c.update(command=i,tile=ti);f=[0]*32;f[30]=ti;f[31]=i
            if c['kind']!='compute':
                d=c['row'];store=c['kind']=='store';f[:11]=[int(store),indices['rgb'],d['width'],d['groups'],d['bank'],d['base'],1,(0x800000 if store else 0)+(d['y']*1280+d['origin'])*3,int(store),d['y'],d['origin']]
            else:
                name='rgb' if c['node']=='rgb_head' else c['node'];parameter='rgb' if c['parameter_node']=='rgb_head' else c['parameter_node'];op=ops[parameter]
                res=package['residual'][name] if c['skip'] else dict(main_scaled=False,multiplier=0,post_multiplier=0)
                fl=int(c['operator'] in ('pw1x1','pw_fused'))|int(c['nn'])<<1|int(c['operator']=='conv3x3')<<2|int(c['stride2'])<<3|int(bool(c['skip']))<<4|int(res['main_scaled'])<<5
                ib=pack([d['base'] for d in c['inputs']],10);ob=pack([d['base'] for d in c['outputs']],10);sb=pack([d['base'] for d in c['skip']],10)
                f[:26]=[2,indices[name],c['pixels'],c['rows'],c['input_groups'],c['output_groups'],fl,c['output_x'],c['source_width'],c['inputs'][0]['origin'],pack([d['bank'] for d in c['inputs']],3),ib&0xffffffff,ib>>32,pack([d['bank'] for d in c['outputs']],3),ob&0xffffffff,ob>>32,op['weight_offset'],op['bias_offset'],op['multiplier_offset'],op['relu_offset'],pack([d['bank'] for d in c['skip']],3),sb&0xffffffff,sb>>32,c['skip'][0]['delta'] if c['skip'] else 0,res['multiplier'],res['post_multiplier']]
                f[28]=c['mask'];last[indices[name]]=i
                if c['skip']:last[indices[parameter]]=i
                if c['nn']:last[indices[next(n for n in nodes if n['id']==name)['inputs'][0]]]=i
                pn=next(n for n in nodes if n['id']==parameter);macs+=c['pixels']*c['rows']*pn['channels']*pn['terms']
            commands.append(c);fields.append(f);totals[c['kind']]+=1
        print('COMPONENT_PAIR_TILE',ti,meta['high'],flush=True)
    assert set(last)==set(range(len(nodes))) and len(commands)<131072
    for n,i in last.items():fields[i][29]|=1<<n
    cw=dict(CWIDTHS);cw.update({10:18,13:12,20:12,30:5,31:17});dw=dict(DWIDTHS);dw.update({4:3,30:5,31:17})
    schemas=[[('peer_completed',17),('program_index',17)]+[(f'f{i}',n) for i,n in cw.items()]+[('storage_phase',1),('out_phase',4)],[('peer_completed',17)]+[(f'f{i}',n) for i,n in dw.items()]+[('region',1)]]
    streams=[[],[]];regions=[];last_write=[None]*12288;last_read=[[0]*12288 for _ in range(2)];counts=[0,0]
    for i,c in enumerate(commands):
        e=int(c['kind']!='compute');r,w=access(c);regions.append((r,w));dep=0
        for u in r|w:
            prior=last_write[u]
            if prior and prior[0]!=e:dep=max(dep,prior[1])
            if u in w:dep=max(dep,last_read[1-e][u])
        counts[e]+=1
        for u in r:last_read[e][u]=counts[e]
        for u in w:last_write[u]=(e,counts[e])
        vals=dict(peer_completed=dep,program_index=i,region=int(c['kind']=='store'),**{f'f{j}':v for j,v in enumerate(fields[i])})
        if not e:vals.update(storage_phase=c['inputs'][0]['phase'],out_phase=pack([d['phase'] for d in c['outputs']],1))
        size=16 if e else 64;bits=offsets(schemas[e])[1];assert bits<=(124 if e else 504)
        value=int.from_bytes(encode(vals,schemas[e],size),'little')|((3<<124) if e else (0xc3<<504));streams[e].append(dict(index=i,dep=dep,packed=value))
    trials=[replay(streams,regions,seed) for seed in (71,391,1027)];stream_meta=[]
    for e,name in enumerate(('compute','dma')):
        size=64 if e==0 else 16;binary=b''.join(s['packed'].to_bytes(size,'little') for s in streams[e]);path=out/(name+'.bin');path.write_bytes(binary)
        (out/(name+'.hex')).write_text(''.join(f'{s["packed"]:0{size*2}x}\n' for s in streams[e]));stream_meta.append(dict(name=name,schema=schemas[e],count=counts[e],record_bytes=size,used_bits=offsets(schemas[e])[1],version=0xc3 if e==0 else 3,version_offset=504 if e==0 else 124,address=0x04000000 if e==0 else 0x04500000,sha256=file_hash(path),crc32=zlib.crc32(binary)))
    with gzip.open(out/'commands.json.gz','wt',encoding='utf-8') as f:json.dump(commands,f,separators=(',',':'))
    (out/'fields.hex').write_text(''.join(f'{sum(v<<(32*i) for i,v in enumerate(f)):0256x}\n' for f in fields))
    sources=[Path(__file__),HW/'fourway_frame_program.py',HW/'screen_row_pair_layout.py',HW/'compile_row_pair_program.py',HW/'compile_row_pair_descriptors.py'];frozen=out/'frozen';frozen.mkdir()
    for p in sources:(frozen/p.name).write_bytes(p.read_bytes())
    write_json(out/'program.json',dict(status='C3_ROW_PAIR_COMPILER_ONLY_PASS',components=True,region_format='YCoCg3',index_bits=17,stripe_width=a.width,nodes=len(nodes),complete_node_mask=(1<<len(nodes))-1,commands=len(commands),counts=dict(totals),streams=stream_meta,coefficient_address=0x04600000,package=str(a.package),package_sha256=file_hash(a.package),tiles=tiles,scheduled_macs=macs,dependency_replays=trials,source_hashes={str(p):file_hash(p) for p in sources},seconds=time.perf_counter()-start,rtl=False,fit=False,timing=False))
    print('COMPONENT_PAIR_PROGRAM_PASS',len(commands),totals,macs,flush=True)


if __name__=='__main__':main()
