"""Prove and specialize ONLY RAM write-data owner selection, without a cycle.

The full mapped graph is the authority. A changed data value is allowed only
when the corresponding physical RAM port has CLKEB=0. No protocol, address,
write-enable, clock, reset or protected primitive is modified.
"""
import argparse, collections, functools, itertools, json, re
from pathlib import Path
from run_units import ROOT, sha
from native_local_drivers import graph, raw_cells, CELL, lexical_tokens, render_token
from probe_full_native_split_map import params

BASE=ROOT/'tmp/frame-execution/connected-board-camera-checked3'
ZERO=('literal',"1'b0",None)

def inspect_source(text):
    g=graph(text); ps=params(text); loads=collections.defaultdict(list); drivers={}
    for n,(kind,pins) in g.items():
        out={'EFX_LUT4':'O','EFX_FF':'Q'}.get(kind)
        if out: drivers[pins[out][0]]=n
        for p,ts in pins.items():
            if p==out:continue
            for i,t in enumerate(ts):loads[t].append((n,p,i))
    aliases={}
    for a,b in re.findall(r'^\s*assign\s+(.*?)\s*=\s*(.*?)\s*/\*',text,re.M):
        aa,bb=lexical_tokens(a),lexical_tokens(b)
        assert len(aa)==len(bb),(a,b)
        for x,y in zip(aa,bb):aliases[x]=y
    def evaluator(fixed):
        visiting=set()
        @functools.lru_cache(None)
        def value(t):
            if t in fixed:return frozenset([fixed[t]])
            if t[0]=='literal':
                if t[1] in ("1'b0","1'b1"):return frozenset([int(t[1][-1])])
                return frozenset([0,1])
            if t in visiting:return frozenset([0,1])
            visiting.add(t)
            if t in aliases:r=value(aliases[t])
            elif t in drivers and g[drivers[t]][0]=='EFX_LUT4':
                name=drivers[t];pins=g[name][1];mask=ps[name]['LUTMASK'][1]
                ins=[value(pins.get('I'+str(i),[ZERO])[0]) for i in range(4)]
                r=frozenset((mask>>sum(v<<i for i,v in enumerate(vs)))&1 for vs in itertools.product(*ins))
            else:r=frozenset([0,1])
            visiting.remove(t);return r
        return value
    banks=[]; groups=[]
    for b in range(6):
        names=[n for n,(k,p) in g.items() if k=='EFX_FF' and n.endswith(f'/router/held_owner[{b}][1]~FF')]
        assert len(names)==1,(b,names)
        owner1=names[0]; prefix=owner1[:owner1.index('/router/')+8]
        owner0=prefix+f'held_owner[{b}][0]~FF'; held=prefix+f'held[{b}]~FF'
        ts=[g[n][1]['Q'][0] for n in (owner0,held,owner1)]
        rams=[n for n,(k,p) in g.items() if k=='EFX_DPRAM10' and f'/arena/ROW[{b}].' in n]
        assert len(rams)==14,(b,len(rams))
        ram_enables=sorted({g[n][1]['CLKEB'][0] for n in rams},key=str)
        proof=[]
        for v in (0,1,4,5,7): # owner0,held,owner1: inactive or invalid owner 3
            fixed={t:(v>>i)&1 for i,t in enumerate(ts)};ev=evaluator(fixed)
            outputs=[sorted(ev(t)) for t in ram_enables]
            assert all(x==[0] for x in outputs),('RAM_NOT_PROVEN_DISABLED',b,v,outputs)
            proof.append(dict(bits=v,clke_values=outputs))
        decoders=[];candidates=set()
        for ram in rams:
            for t in g[ram][1]['WDATAB']:
                if t[0]=='literal':continue
                mux=drivers[t];k,p=g[mux]
                assert k=='EFX_LUT4' and ps[mux]['LUTMASK'] in ((16,0xacac),(16,0x44f0)),(mux,ps[mux])
                select_port='I2' if ps[mux]['LUTMASK'][1]==0xacac else 'I3'
                candidates.add(drivers[p[select_port][0]])
        for n in sorted(candidates):
            k,p=g[n];assert k=='EFX_LUT4'
            truth=[evaluator({t:(v>>i)&1 for i,t in enumerate(ts)})(p['O'][0]) for v in range(8)]
            assert truth==[frozenset([v]) for v in [0,0,0,0,0,0,1,0]],(n,truth)
            out=p['O'][0];payload=[];other=[]
            for cell,port,index in loads[out]:
                kp,pp=g[cell]
                if kp!='EFX_LUT4':other.append((cell,port,index));continue
                dest=pp['O'][0]; fan=loads[dest]
                if not fan or not all(x[0] in rams and x[1]=='WDATAB' for x in fan):
                    other.append((cell,port,index));continue
                expected_port='I2' if ps[cell]['LUTMASK'][1]==0xacac else 'I3'
                assert port==expected_port,(cell,port,expected_port)
                # Every destination is the intended RAM bank. All payload
                # consumers are listed, so no hidden external output is allowed.
                assert dest not in aliases.values(),('aliased output',dest)
                payload.append(dict(cell=cell,port=port,ram_loads=fan))
            decoders.append(dict(decoder=n,payload_count=len(payload),other_loads=other))
            if payload:groups.append(dict(bank=b,decoder=n,owner_ff=owner1,old_select=list(out),loads=payload))
        bits=sum(len(x['ram_loads']) for row in groups if row['bank']==b for x in row['loads'])
        assert bits==128,(b,bits)
        banks.append(dict(bank=b,owner_ff=owner1,owner_tokens=ts,ram_count=len(rams),ram_enables=ram_enables,
                          disabled_proof=proof,decoders=decoders,payload_bits=bits))
    return g,ps,banks,groups

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    source=BASE/'outflow/ti60f225_oob.map.v'
    try:
        g,ps,banks,groups=inspect_source(source.read_text())
        result=dict(status='PAYLOAD_ONLY_RAM_ENABLE_PROOF_PASS',banks=banks,groups=groups,
            group_count=len(groups),payload_bits=sum(b['payload_bits'] for b in banks),
            inputs={str(source):sha(source),str(Path(__file__)):sha(__file__)},
            scope='Two-state combinational proof for all other inputs, all 84 physical RAM CLKEB pins. Sequential four-state clone test is a separate gate.',
            hardware_access=False,board_release=False)
        (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('banks','groups')},ensure_ascii=False),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e)),indent=2));raise

if __name__=='__main__':main()
