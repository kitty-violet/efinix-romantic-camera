"""Reject pair groups creating conservative graph cycles; never waive warnings.

Uses the actual complete mapped public graph, including newly synthesized
control. Every member of a cyclic strongly connected component is split back
into its two original output functions. The next full map must independently
report no loop; this check alone is not a release.
"""
import argparse, collections, json, pickle, re, time
from pathlib import Path
from run_units import ROOT, sha


def cyclic_components(g):
    kinds={'EFX_LUT4':{'O'},'EFX_ADD':{'O','CO'},'EFX_COMB4':{'O','P','CO','FCO'}}
    selected={n:(k,p) for n,(k,p) in g.items() if k in kinds}
    drivers={}
    for n,(k,pins) in selected.items():
        for p,ts in pins.items():
            if p in kinds[k]:
                for t in ts:
                    if t[0]=='net':assert t not in drivers,(t,n);drivers[t]=n
    pred={};succ={n:set() for n in selected}
    for n,(k,pins) in selected.items():
        pred[n]={drivers[t] for p,ts in pins.items() if p not in kinds[k] for t in ts if t in drivers}
        for parent in pred[n]:succ[parent].add(n)
    degrees={n:len(v) for n,v in pred.items()};todo=collections.deque(n for n,d in degrees.items() if d==0)
    while todo:
        n=todo.popleft()
        for child in succ[n]:
            degrees[child]-=1
            if degrees[child]==0:todo.append(child)
    remaining={n for n,d in degrees.items() if d};seen=set();order=[]
    for n in sorted(remaining):
        if n in seen:continue
        stack=[(n,False)]
        while stack:
            x,finished=stack.pop()
            if finished:order.append(x);continue
            if x in seen:continue
            seen.add(x);stack.append((x,True));stack.extend((y,False) for y in succ[x]&remaining if y not in seen)
    seen=set();components=[]
    for n in reversed(order):
        if n in seen:continue
        component=set();stack=[n];seen.add(n)
        while stack:
            x=stack.pop();component.add(x)
            for y in pred[x]&remaining:
                if y not in seen:seen.add(y);stack.append(y)
        if len(component)>1 or n in succ[n]:components.append(sorted(component))
    return components,len(selected)


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();start=time.monotonic()
    proof_path=a.proof.resolve();proof=json.loads(proof_path.read_text());assert proof['status']=='OFFICIAL_LOGIC_PAIR_FOUR_STATE_FILTER_COMPLETED'
    for f,h in proof['inputs'].items():assert sha(f)==h
    cache=board/'retained-audit-parse.pkl';key,original,g,pa,ps=pickle.loads(cache.read_bytes())
    assert key[:2]==[sha(board/'native_source.v'),sha(board/'outflow/ti60f225_oob.map.v')]
    pairs={n for n,(kind,_) in g.items() if kind=='EFX_COMB4'}
    assert pairs=={f'SERVICE_LOGIC_PAIR_{i:04d}' for i in proof['accepted']}
    components,total=cyclic_components(g);bad=set();rounds=[]
    while components:
        rejected={n for comp in components for n in comp if n in pairs}
        assert rejected and not bad&rejected,components
        rounds.append(dict(components=components,rejected=sorted(rejected)));bad|=rejected
        for n in rejected:
            kind,pins=g.pop(n);assert ps[n]['MODE']=='"LOGIC"'
            assert not pins.get('CO') and not pins.get('FCO')
            # Primary preserves all connected inputs. P has only I0/I1 in
            # the official model, so its split LUT omits I2/I3 altogether.
            for port in ('O','P'):
                inputs=range(4) if port=='O' else range(2)
                g[n+'_analysis_split'+port]=('EFX_LUT4',{**{'I'+str(i):pins['I'+str(i)] for i in inputs if 'I'+str(i) in pins},'O':pins[port]})
        components,total=cyclic_components(g)
    warned=set(re.findall(r'driver: (SERVICE_LOGIC_PAIR_\d+)',(board/'outflow/ti60f225_oob.warn.log').read_text()))
    assert warned and warned<=bad,(warned-bad)
    excluded=sorted(int(n.rsplit('_',1)[1]) for n in bad)
    proof.update(accepted=sorted(set(proof['accepted'])-set(excluded)),rejected=sorted(set(proof['rejected'])|set(excluded)),
        parent_four_state_proof=str(proof_path),parent_four_state_proof_sha256=sha(proof_path),
        conservative_cycles_excluded=excluded,conservative_graph_vertices=total,cycle_rounds=rounds,
        filter_seconds=time.monotonic()-start,new_full_synthesis_required=True)
    for f in [Path(__file__),cache,board/'outflow/ti60f225_oob.map.v',board/'outflow/ti60f225_oob.warn.log',proof_path]:proof['inputs'][str(f)]=sha(f)
    out.mkdir();(out/'result.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(dict(status='CONSERVATIVE_COMPLETE_PUBLIC_COMBINATIONAL_GRAPH_ACYCLIC',accepted=len(proof['accepted']),excluded=excluded,rounds=len(rounds),seconds=proof['filter_seconds'],physical_release=False)))


if __name__=='__main__':main()
