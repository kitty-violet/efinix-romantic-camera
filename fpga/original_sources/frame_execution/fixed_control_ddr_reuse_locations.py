"""Reuse function-matched complete tiles/chains from the passing full board.

This changes placement only. Fingerprints aid matching, not a claim of formal
equivalence. The frozen new netlist and original constraints remain authoritative.
"""
import argparse, collections, hashlib, itertools, json, re
from pathlib import Path
from run_units import ROOT, EF, sha
from audit_snapshot_cells import cells
from audit_control_state_implementation import norm
from audit_canonical_multiseed import audit

REFERENCE=ROOT/'tmp/frame-execution/command-progress-seed3-1/canonical-seed1'
PREFIX='video_processing_pipeline/pipeline/cnn/'

def locations(path):
    return {w[0]:tuple(map(int,w[1:])) for line in path.read_text().splitlines()
            if len(w:=line.split())==4 and not line.startswith('#')}

def read(path):
    text=path.read_text(); cc=cells(text); params={}
    for n,k,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text,re.M):
        params.setdefault(norm(n),{})[k]=norm(v)
    return {n:(k,p,params.get(n,{})) for n,(k,p,a) in cc.items()}

def digest(x):
    return hashlib.sha256(repr(x).encode()).hexdigest()

def fingerprints(cc):
    drivers={}; memo={}; busy=set()
    for n,(k,p,par) in cc.items():
        outs=('O','CO') if k=='EFX_ADD' else ('O',) if k=='EFX_LUT4' else ('Q',) if k in ('EFX_FF','EFX_SRL8') else ()
        for pin in outs:
            if pin in p: drivers[p[pin]]=(n,pin)
    def net(v):
        if v in memo:return memo[v]
        if v in busy: return digest(('cycle-boundary',v))
        if v not in drivers:return digest(('boundary',v))
        n,pin=drivers[v];k,p,params=cc[n]
        if k not in ('EFX_LUT4','EFX_ADD'):
            return digest(('register-boundary',n,k,pin,sorted(params.items())))
        busy.add(v)
        key=logic(n); busy.remove(v)
        memo[v]=digest((key,pin));return memo[v]
    def logic(n):
        k,p,params=cc[n]
        if k=='EFX_LUT4':
            assert set(params)=={'LUTMASK'},(n,params)
            mask=int(params['LUTMASK'].split("'h")[1],16)
            pins=[p.get('I'+str(i),"1'b0") for i in range(4)]
            keys=[None if s in ("1'b0","1'b1") else net(s) for s in pins]
            variables=sorted(set(keys)-{None}); truth=[]
            for bits in itertools.product((0,1),repeat=len(variables)):
                vals=dict(zip(variables,bits));idx=sum((int(s=="1'b1") if key is None else vals[key])<<i for i,(s,key) in enumerate(zip(pins,keys)))
                truth.append((mask>>idx)&1)
            # Remove physically connected but logically unused inputs.
            for i in reversed(range(len(variables))):
                stride=1<<(len(variables)-1-i)
                if all(truth[j]==truth[j+stride] for j in range(len(truth)) if not j&stride):
                    truth=[t for j,t in enumerate(truth) if not j&stride];variables.pop(i)
            return k,tuple(variables),tuple(truth)
        return k,tuple((q,net(v)) for q,v in sorted(p.items()) if q not in ('O','CO','Q')),tuple(sorted(params.items()))
    return {n:logic(n) for n in cc}

def plan(current, reference=REFERENCE):
    previous=audit(reference);assert previous['current_timing_pass']
    old=Path(previous['physical']);oldnet=old/'outflow/ti60f225_oob.map.v'
    newnet=Path(current)/'outflow/ti60f225_oob.map.v'
    a,b=read(oldnet),read(newnet);oldpos=locations(old/'outflow/ti60f225_oob.out.placeloc')
    af,bf=fingerprints(a),fingerprints(b);bykey=collections.defaultdict(list)
    for n,key in af.items():bykey[key].append(n)
    # Exact named instances have precedence; otherwise duplicate one-bit
    # RAM/DSP or identical FF inputs could consume a real instance twice.
    mapping={n:n for n in a.keys()&b.keys() if a[n]==b[n] and n in oldpos}
    used=set(mapping.values())
    exact=collections.defaultdict(list)
    for n,c in a.items():exact[repr(c)].append(n)
    for n,c in b.items():
        if n in mapping:continue
        options=[o for o in exact.get(repr(c),[]) if o not in used]
        if len(options)==1:mapping[n]=options[0];used.add(options[0])
    for n,key in bf.items():
        if n in mapping:continue
        options=[o for o in bykey.get(key,[]) if o not in used]
        if n in options:mapping[n]=n;used.add(n)
        elif len(options)==1:mapping[n]=options[0];used.add(options[0])
    counts=collections.Counter(mapping.values())
    mapping={n:o for n,o in mapping.items() if counts[o]==1 and o in oldpos}
    # Select CNN logic plus mandatory DSP/RAM. Do not pin generated PHY cells.
    eligible={n for n in mapping if n.startswith(PREFIX) or
              any(PREFIX in v for v in b[n][1].values()) or b[n][0] in ('EFX_DSP24','EFX_DSP48','EFX_RAM10','EFX_DPRAM10')}
    inverse={o:n for n,o in mapping.items()};tiles=collections.defaultdict(set)
    for n,pos in oldpos.items():tiles[pos[:2]].add(n)
    # Keep all old occupants of a tile, or leave it entirely automatic; avoids
    # the documented loss of LUT/FF packing for partly fixed tiles.
    chosen=set()
    for n in eligible:
        occupants=tiles[oldpos[mapping[n]][:2]]
        if occupants <= inverse.keys():chosen.update(inverse[o] for o in occupants)
    # Every ADD carry chain and SRL shift chain must be wholly assigned.
    neighbors=collections.defaultdict(set);drivers={}
    for n,(k,p,par) in b.items():
        if k in ('EFX_ADD','EFX_SRL8'):
            for q in ('CO',) if k=='EFX_ADD' else ('Q',):
                if q in p:drivers[p[q]]=n
    for n,(k,p,par) in b.items():
        key='CI' if k=='EFX_ADD' else 'D' if k=='EFX_SRL8' else None
        other=drivers.get(p.get(key)) if key else None
        if other and b[other][0]==k:neighbors[n].add(other);neighbors[other].add(n)
    while True:
        remove={n for n in chosen if neighbors[n]-chosen}
        expanded=set(remove)
        for n in remove:expanded.update(inverse[o] for o in tiles[oldpos[mapping[n]][:2]])
        if not expanded:break
        chosen-=expanded
    assignments={n:oldpos[mapping[n]] for n in sorted(chosen)}
    assert len(set(assignments.values()))==len(assignments)
    assert all(not (neighbors[n]-chosen) for n in chosen)
    # Check paired FF D is still the co-located logic output (same tile rule).
    for n,pos in assignments.items():
        k,p,_=b[n]
        if k!='EFX_FF':continue
        others=[inverse[o] for o in tiles[pos[:2]] if inverse[o]!=n]
        for other in others:
            kind,ports,_=b[other]
            assert kind in ('EFX_LUT4','EFX_ADD','EFX_SRL8'),(n,other,kind)
            assert p['D'] in [ports.get('O'),ports.get('Q')],(n,other,'packing connection differs')
    result=dict(status='MATCHED_COMPLETE_TILE_PLACEMENT_PREPARED_NOT_TIMING_PASS',
        reference=str(reference),reference_result_sha256=sha(reference/'result.json'),
        old_map_sha256=sha(oldnet),new_map_sha256=sha(newnet),
        old_placement_sha256=sha(old/'outflow/ti60f225_oob.out.placeloc'),
        matched_before_tile_filter=len(mapping),assignment_count=len(assignments),
        types=dict(collections.Counter(b[n][0] for n in chosen)),
        mapping={n:mapping[n] for n in sorted(chosen)},assignments=assignments,
        generator_sha256=sha(__file__),logic_or_constraints_changed=False,hardware_access=False)
    assert len(assignments)>15000,result['types']
    return result

def generate(root):
    root=Path(root);proof=plan(root);target=root/'fixed_dsp.placeloc'
    prior=locations(target)
    # All DSP/RAM anchors are explicitly matched to the passing reference.
    assert set(prior)<=proof['assignments'].keys(),set(prior)-proof['assignments'].keys()
    target.write_text('# Matched complete tiles and chains; original full-board passing placement.\n'+
        ''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in proof['assignments'].items()))
    proof['assignments_sha256']=sha(target)
    (root/'reuse_placement.json').write_text(json.dumps(proof,indent=2))
    (root/'reuse_placement_generator.py').write_bytes(Path(__file__).read_bytes())
    return proof

def verify_placed(root):
    root=Path(root);r=json.loads((root/'reuse_placement.json').read_text())
    assert sha(root/'outflow/ti60f225_oob.map.v')==r['new_map_sha256']
    assert sha(root/'fixed_dsp.placeloc')==r['assignments_sha256']
    assert sha(root/'reuse_placement_generator.py')==r['generator_sha256']
    actual=locations(root/'outflow/ti60f225_oob.out.placeloc')
    for n,p in r['assignments'].items():assert actual[n]==tuple(p),(n,p,actual.get(n))
    return dict(status='MATCHED_COMPLETE_TILE_LOCATIONS_VERIFIED',count=len(r['assignments']),
        placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();r=plan(a.board);a.output.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('assignments','mapping')}))
