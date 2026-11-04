"""Deterministic node-to-bank search; never shortens a live interval."""
import numpy as np


def search(records,local,placement):
    names=sorted({n for n,y in records});nt=max(r['last'] for r in records.values())+2
    curves={n:np.zeros((6,nt),np.int32) for n in names}
    for (n,y),r in records.items():curves[n][y%6,r['born']]+=r['length'];curves[n][y%6,r['last']+1]-=r['length']
    for n in names:curves[n]=np.cumsum(curves[n],axis=1,dtype=np.int32)
    rotations={n:[np.roll(curves[n],i,axis=0) for i in range(6)] for n in names}
    score=lambda usage:(int(usage.max()),int(np.maximum(usage-940,0).sum()),int(np.max(usage,axis=1).sum()))
    rng=np.random.default_rng(2026100457);candidates=[];seen=set();attempts=[]
    for trial in range(12):
        offsets={'source':0}
        for n in local:
            if n=='source':continue
            offsets[n]=offsets[local[n]['parents'][0]] if local[n]['kind']=='nearest_x2' else int(rng.integers(6))
        use=sum(rotations[n][offsets[n]] for n in names)
        for sweep in range(5):
            changed=False
            for n in rng.permutation([n for n in names if n!='source']):
                stripped=use-rotations[n][offsets[n]];best=offsets[n];bestscore=score(use)
                for off in range(6):
                    candidate=score(stripped+rotations[n][off])
                    if candidate<bestscore:best=off;bestscore=candidate
                if best!=offsets[n]:offsets[n]=best;changed=True
                use=stripped+rotations[n][offsets[n]]
            if not changed:break
        key=tuple(offsets[n] for n in names)
        if key not in seen:seen.add(key);candidates.append((score(use),offsets.copy()))
    for lower,offsets in sorted(candidates,key=lambda x:x[0]):
        if lower[0]>1024:continue
        for (n,y),r in records.items():r['bank']=(y+offsets[n])%6
        high,pos=placement(records);attempts.append(dict(lower_bound=lower[0],high=high))
        if pos is not None:
            for n in local:
                if n not in offsets or local[n]['kind']=='nearest_x2':offsets[n]=offsets[local[n]['parents'][0]]
            print('CAPACITY_BANK_SEARCH_PASS',lower[0],high,flush=True)
            return (offsets,high,pos),dict(candidates=len(candidates),attempts=attempts,lifetimes_unchanged=True)
    print('CAPACITY_BANK_SEARCH_FAIL',attempts,flush=True)
    return None,dict(candidates=len(candidates),attempts=attempts,lower_bounds=[x[0][0] for x in candidates],lifetimes_unchanged=True)
