"""Finite-slot DW/stem phase schedule with actual one-read/one-write banks.
The schedule is a design model, not RTL equivalence. No second write port,
unbounded collector, free read or extra arithmetic pool is introduced.
"""
from collections import deque
from functools import lru_cache
@lru_cache(maxsize=None)
def pipeline(width,height,groups,source_width,source_height,stride,origin,source_origin,out_y,nn,stem,columns=5,phase_mul=1,cluster_phase=0,interleave=False):
    jobs=width*height*groups;count=27;terms=27 if stem else 9;period=terms
    def decode(index):
        g,pixel=divmod(index,width*height);y,x=divmod(pixel,width)
        return g,y,x
    def phase_of(owner):return ((owner%9*phase_mul+owner//9*cluster_phase)%9)*(3 if stem else 1)
    def job_index(batch,owner):return batch*27+((owner%9)*3+owner//9 if interleave else owner)
    last=max((i//27)*period+phase_of(i%27)+terms for i in range(jobs))
    slot=[0]*27;returns=[];collector=[deque() for _ in range(3)]
    wall=0;writes=0;reads=0;conflict=0;slot_stalls=0;max_slots=0;delivered=set();rr=0
    peak_returns=0
    def tick():
        nonlocal wall,writes,rr,peak_returns
        wall+=1
        due=[x for x in returns if x[0]<=wall]
        returns[:]=[x for x in returns if x[0]>wall]
        for _,owner,index,bank in sorted(due):collector[owner//9].append((owner,index,bank))
        used=set()
        for offset in range(3):
            c=(rr+offset)%3
            if collector[c] and collector[c][0][2] not in used:
                owner,index,bank=collector[c].popleft();used.add(bank)
                assert index not in delivered;delivered.add(index);slot[owner]-=1
                assert slot[owner]>=0
                writes+=1
        rr=(rr+1)%3
        peak_returns=max(peak_returns,len(returns)+sum(map(len,collector)))
    tick() # one real synchronous memory pipeline fill
    for t in range(last):
        request=[set() for _ in range(3*columns)];work=[]
        for owner in range(27):
            phase=phase_of(owner)
            if t<phase:continue
            batch,k=divmod(t-phase,period);index=job_index(batch,owner)
            if k>=terms or index>=jobs:continue
            g,y,x=decode(index);tap=k//3 if stem else k;ky,kx=divmod(tap,3)
            sx=min(source_width-1,max(0,(origin+x)*stride+kx-1))
            sy=min(source_height-1,max(0,(out_y+y)*stride+ky-1))
            if nn:sx//=2;sy//=2
            lx=sx-source_origin;assert lx>=0
            bank=(sy%3)*columns+lx%columns
            # RGB word packs two pixels in the SAME column bank; preserves
            #5-way port distribution while halving padded RGB live capacity.
            word=lx//(2*columns) if stem else lx//columns
            request[bank].add((sy,word,g))
            work.append((owner,index,k,(out_y+y)%3*columns+x%columns))
        if not work:tick();continue
        while any(k==0 and slot[owner]>=2 for owner,_,k,_ in work):
            tick();slot_stalls+=1
            if slot_stalls>10*jobs:raise AssertionError('slot/collector deadlock')
        service=max(map(len,request));assert service>=1
        conflict+=service-1;reads+=sum(map(len,request))
        for _ in range(service):tick()
        for owner,index,k,bank in work:
            if k==0:slot[owner]+=1;max_slots=max(max_slots,max(slot));assert slot[owner]<=2
            if k==terms-1:returns.append((wall+8,owner,index,bank))
    while returns or any(collector):tick()
    assert writes==jobs and len(delivered)==jobs and not any(slot)
    return dict(cycles=wall,jobs=jobs,word_reads=reads,result_writes=writes,
        bank_conflict_cycles=conflict,slot_stall_cycles=slot_stalls,
        max_slots_per_front=max_slots,peak_inflight_returns=peak_returns,
        banks=3*columns,read_ports_per_bank=1,write_ports_per_bank=1,results_per_cycle=3,
        implementation='FINITE_PIPELINED_PORT_MODEL_NOT_RTL')
