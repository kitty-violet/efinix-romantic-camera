"""Two ordered engines with conservative, actual arena-word dependencies.

Preserve order inside each engine. For every cross-engine RAW/WAR/WAW edge,
the later command waits for the earlier command's complete ACK/drain barrier.
DDR transfers keep their original order. No input/feature values are changed.
"""
import argparse,hashlib,json,random,shutil
from functools import lru_cache
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def row_words(row):
    return {((row['plane']*4+x%4)*1024+row['base']+(x//4)*row['groups']+g)
            for x in range(row['width']) for g in range(row['groups'])}


def footprint(command):
    if command['kind']=='compute':
        reads=set().union(*(row_words(r) for r in command['inputs']+command['skip']))
        writes=set().union(*(row_words(r) for r in command['outputs']))
    elif command['kind']=='load':reads,writes=set(),row_words(command['row'])
    else:reads,writes=row_words(command['row']),set()
    assert not reads&writes and all(0<=a<12288 for a in reads|writes)
    return reads,writes


def reorder_transfers(program):
    """Advance transfers across independent work, preserving per-address order."""
    compute_read=[0]*12288;compute_write=[0]*12288;dma_last=[-1]*12288
    external_last={};transfers=[];computes=[];required=[]
    windows={'encoder':(0,33554432),'residual':(33554432,50331648),'decoder':(50331648,8388608)}
    for i,q in enumerate(program):
        reads,writes=footprint(q)
        if q['kind']=='compute':
            computes.append(i);count=len(computes)
            for address in reads:compute_read[address]=count
            for address in writes:compute_write[address]=count
            continue
        dep=max([0]+[compute_write[a] for a in reads]+[max(compute_read[a],compute_write[a]) for a in writes])
        previous={dma_last[a] for a in reads|writes};row=q['row'];channels=q['channels']
        start=windows[q['group']][int(q['kind']=='store')]+(row['y']*q['global_width']+row['origin'])*channels
        beats=range(start//16,(start+row['width']*channels+15)//16)
        previous.update(external_last.get(a,-1) for a in beats)
        dep=max([dep]+[required[j] for j in previous if j>=0]);slot=len(transfers)
        transfers.append(i);required.append(dep)
        for address in reads|writes:dma_last[address]=slot
        for address in beats:external_last[address]=slot
    order=sorted(range(len(transfers)),key=lambda d:(required[d],d));merged=[];d=0
    for c in range(len(computes)+1):
        while d<len(order) and required[order[d]]<=c:
            merged.append(transfers[order[d]]);d+=1
        if c<len(computes):merged.append(computes[c])
    assert d==len(order) and sorted(merged)==list(range(len(program)))
    # Each read must see the same prior write and successive writes stay ordered.
    # This independently detects changed RAW/WAR/WAW ordering after permutation.
    def versions(sequence):
        version=[-1]*12288;signatures={}
        for i in sequence:
            reads,writes=footprint(program[i]);signatures[i]=tuple((a,version[a]) for a in sorted(reads|writes))
            for address in writes:version[address]=i
        return signatures
    assert versions(range(len(program)))==versions(merged),'RAM read/write version changed by reorder'
    return merged,dict(status='PER_ADDRESS_DEPENDENCY_PRESERVING_DMA_REORDER',moved_commands=sum(i!=j for i,j in enumerate(merged)),
                       original_release_inversions=sum(a>b for a,b in zip(required,required[1:])),extra_ram_words=0)


def prepare(source,target,reorder_dma=False):
    assert not target.exists();target.mkdir(parents=True)
    manifest=json.loads((source/'frame_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha(source/name)==digest;(target/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target/name)
    program=json.loads((target/'program.json').read_text())
    if reorder_dma:
        order,reorder_audit=reorder_transfers(program)
        words=(target/'program.hex').read_text().splitlines();program=[dict(program[i],command=n) for n,i in enumerate(order)]
        (target/'program.json').write_text(json.dumps(program,separators=(',',':')))
        (target/'program.hex').write_text(''.join(words[i]+'\n' for i in order))
        manifest['reordering']=reorder_audit;manifest['reorder_source_manifest_sha256']=sha(source/'frame_manifest.json')
    read_history=[[0]*12288 for _ in range(2)];write_history=[[0]*12288 for _ in range(2)]
    commands=[[],[]];dependencies=[[],[]];counts=[0,0]
    for i,command in enumerate(program):
        engine=0 if command['kind']=='compute' else 1;other=1-engine
        reads,writes=footprint(command)
        dependency=max([0]+[write_history[other][a] for a in reads]+[
            max(write_history[other][a],read_history[other][a]) for a in writes])
        counts[engine]+=1;commands[engine].append(i);dependencies[engine].append(dependency)
        for a in reads:read_history[engine][a]=counts[engine]
        for a in writes:write_history[engine][a]=counts[engine]
    # Independent interleaving exercise: stall either engine, and ensure that
    # every pair of simultaneously active commands has disjoint write hazards.
    @lru_cache(maxsize=32)
    def fp(i):return footprint(program[i])
    concurrency=[]
    for seed in (7,71,193):
        rng=random.Random(seed);finished=[0,0];live=[None,None];both=0;ticks=0
        while finished!=counts:
            for e in (0,1):
                if live[e] is None and finished[e]<counts[e] and dependencies[e][finished[e]]<=finished[1-e]:
                    live[e]=commands[e][finished[e]]
            if all(i is not None for i in live):
                ar,aw=fp(live[0]);br,bw=fp(live[1]);assert not (aw&(br|bw) or bw&ar),(seed,live)
                both+=1
            assert any(i is not None for i in live),('dependency deadlock',finished)
            e=rng.randrange(2)
            if live[e] is not None:finished[e]+=1;live[e]=None
            ticks+=1
        concurrency.append(dict(seed=seed,overlapping_command_pairs=both,events=ticks))
    for e,label in enumerate(('compute','dma')):
        (target/f'{label}_commands.hex').write_text(''.join(f'{v:08x}\n' for v in commands[e]))
        (target/f'{label}_dependencies.hex').write_text(''.join(f'{v:08x}\n' for v in dependencies[e]))
    plan=dict(status='CROSS_ENGINE_PHYSICAL_DEPENDENCIES_PASS_NOT_RTL',compute_commands=counts[0],dma_commands=counts[1],
              scope='full conservative arena rows; completion includes responses; DMA preserves per-address external-memory dependencies',
              randomized_interleavings=concurrency,program_sha256=sha(target/'program.json'),generator_sha256=sha(Path(__file__)))
    (target/'concurrency_audit.json').write_text(json.dumps(plan,indent=2))
    manifest['concurrency']=plan
    manifest['evidence'][str(Path(__file__).resolve())]=sha(Path(__file__))
    for name in ('program.json','program.hex','compute_commands.hex','compute_dependencies.hex','dma_commands.hex','dma_dependencies.hex','concurrency_audit.json'):
        manifest['files'][name]=sha(target/name)
    (target/'frame_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(plan),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--reorder-dma',action='store_true');a=p.parse_args()
    prepare(a.source.resolve(),a.target.resolve(),a.reorder_dma)
