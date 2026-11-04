"""Prove the old observation APB cone before using the same-edge boundary."""
import functools,json,pickle
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections,net
from joint_control_observation_binding import bind_apb

def prove_status_apb(old,new):
    board=new['path'].parent.parent
    native=Path(json.loads((board/'inputs.json').read_text())['native'])
    binding=bind_apb(board,native,new['g'],new['p'])
    previous=ROOT/'tmp/frame-execution/video-service-native61-control-priority'
    key,g,p=pickle.loads((previous/'zero-source-parse.pkl').read_bytes())
    assert key==sha(previous/'native_source.v')
    d,_=connections(g)
    leaves=[net('ceg_net4871')]+[net('cpu/state_oh',i) for i in range(10)]
    boundary=set(leaves)
    token=net('apb_enable')
    def evaluate(graph,params,drivers,values):
        @functools.lru_cache(None)
        def ev(t):
            if t in boundary:return values[t]
            if t[0]=='literal':
                assert t[1] in ("1'b0","1'b1");return int(t[1][-1])
            n=drivers[t];kind,pins=graph[n];assert kind=='EFX_LUT4',(n,kind)
            ix=sum(ev(pins.get('I'+str(i),[('literal',"1'b0",None)])[0])<<i for i in range(4))
            return params[n]['LUTMASK'][1]>>ix&1
        return ev(token)
    # Exact correspondence of the state boundary, not arbitrary signal aliases.
    for t in leaves:
        a=old['d'][t];b=d[t]
        assert a==b and old['g'][a]==g[b] and old['p'][a]==p[b],a
    legal=0
    for bits in range(2048):
        values={t:bits>>i&1 for i,t in enumerate(leaves)}
        a=evaluate(old['g'],old['p'],old['d'],values)
        b=evaluate(g,p,d,values)
        assert a==b,('old observation cone differs from proved projection parent',bits,a,b)
        state=bits>>1
        if state and state&(state-1)==0:
            assert a==(values[net('ceg_net4871')] and values[net('cpu/state_oh',4)])
            legal+=1
    assert legal==20
    binding.update(old_observation_cone_boolean_cases=2048,legal_onehot_cases=legal,
                   old_state_boundary_identical=True,projection_parent_source_sha256=key)
    binding['evidence'].update({str(x):sha(x) for x in (old['path'],previous/'native_source.v',Path(__file__))})
    return binding
