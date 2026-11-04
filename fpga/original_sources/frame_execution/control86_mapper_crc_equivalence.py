"""Bounded Boolean proof of the mapper's five new mux cones at CRC FF inputs.

Intermediate auto-net names are deliberately not compared: the mapper renumbers
them. Compare all32 CRC next-state pins using identical FF output boundaries.
"""
import functools,re


def prove(old,new):
    _,og,op,od,ol=old
    _,ng,np,nd,nl=new
    special={f'CutToMuxOpt_{i}/Lut_{j}' for i in range(5) for j in range(2)}
    assert {n for n in ng if n.startswith('CutToMuxOpt_')}==special
    assert not any(n.startswith('CutToMuxOpt_') for n in og)
    assert all(ng[n][0]=='EFX_LUT4' for n in special)
    reached=set();seen=set()
    def forward(token):
        if token in seen:return
        seen.add(token)
        for name,pin in nl.get(token,[]):
            kind,ports=ng[name]
            if kind!='EFX_LUT4':reached.add((name,pin));continue
            forward(ports['O'][0])
    for name in special:forward(ng[name][1]['O'][0])
    targets={f'cpu/sequencer/crc[{i}]~FF' for i in range(32)}
    assert reached and all(n in targets and p=='D' for n,p in reached),reached
    nodes={};unique={};leaves=set();cone=[set(),set()]
    def node(var,lo,hi):
        if lo==hi:return lo
        key=(var,lo,hi)
        if key not in unique:
            assert len(nodes)<200000,'bounded local CRC proof node limit'
            unique[key]=len(nodes)+2;nodes[unique[key]]=key
        return unique[key]
    @functools.lru_cache(None)
    def ite(c,t,f):
        if c<2:return t if c else f
        if t==f:return t
        if t==1 and f==0:return c
        var=min(nodes[x][0] for x in (c,t,f) if x>=2)
        def part(x,high):return nodes[x][2 if high else 1] if x>=2 and nodes[x][0]==var else x
        return node(var,ite(*(part(x,False) for x in (c,t,f))),ite(*(part(x,True) for x in (c,t,f))))
    @functools.lru_cache(None)
    def value(side,token):
        if token[0]=='literal':
            match=re.fullmatch(r"\d+'([bhd])([0-9a-f]+)",token[1],re.I);assert match,token
            v=int(match[2],{'b':2,'h':16,'d':10}[match[1].lower()]);assert v in (0,1);return v
        graph,params,drivers=(og,op,od) if side==0 else (ng,np,nd)
        name=drivers.get(token);assert name is not None,('unbound',token)
        kind,ports=graph[name]
        if kind=='EFX_FF':
            assert od.get(token)==nd.get(token)==name
            assert og[name]==ng[name] and op.get(name,{})==np.get(name,{})
            assert ports['Q']==[token]
            leaves.add(token)
            # Selects first keeps multiplexed CRC recurrence bounded.
            priority=0 if 'nibble' in name else 1 if 'boot_observer_state' in name else 2
            return node((priority,name),0,1)
        assert kind=='EFX_LUT4',(name,kind)
        assert set(params[name])=={'LUTMASK'},name
        cone[side].add(name)
        mask=params[name]['LUTMASK'][1]
        inputs=[ports.get(f'I{i}',[('literal',"1'b0",None)])[0] for i in range(4)]
        def build(i,offset):
            if i==4:return (mask>>offset)&1
            lo,hi=build(i+1,offset),build(i+1,offset|(1<<i))
            return lo if lo==hi else ite(value(side,inputs[i]),hi,lo)
        return build(0,0)
    for name in sorted(targets):
        assert og[name][0]==ng[name][0]=='EFX_FF'
        a=value(0,og[name][1]['D'][0]);b=value(1,ng[name][1]['D'][0])
        assert a==b,('CRC next-state function differs',name)
    assert special<=cone[1]
    return dict(status='CONTROL86_AUTOMAPPER_CRC32_BINARY_EQUIVALENT',
        old_map_sha256=old[0],new_map_sha256=new[0],proven_new_mux_cells=sorted(special),
        directly_affected_crc_pins=len(reached),crc_next_state_pins_checked=32,
        ff_boundaries=sorted(leaves,key=str),old_cone_cells=len(cone[0]),new_cone_cells=len(cone[1]),
        bdd_nodes=len(nodes),node_limit=200000,
        scope='All binary assignments at identical FF output boundaries;32 CRC D equations. No auto-net identity assumption, no whole-design/4-state/timing claim.')
