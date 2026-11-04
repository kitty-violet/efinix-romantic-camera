"""Execute every Model-B row task with finite bank allocation and explicit costs.
This is an architecture scheduling simulation, NOT complete RTL/board timing.
Source-calibrated PW service, generated DW port schedules and unimplemented
engine contracts retain separate provenance. It cannot grant performance PASS.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, re, sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'network_ab'))
sys.path.insert(0,str(HERE.parent/'full_frame_closure'))
from model_contract import compile_contract,load_contract
from wavefront_lifetimes import make_tasks
import hardware_schedule_model as hw
from pipeline_cost import pipeline
CLOCK=150_000_000
SHAPES=((32,32),(8,16),(16,32),(32,16),(16,8),(8,3))
def ceildiv(a,b):return (a+b-1)//b
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def calibration():
    path=ROOT/'output/toolchain/pw_continuous_feed_audit_2026-09-30.json'
    audit=json.loads(path.read_text(encoding='utf-8'))
    assert audit['status']=='PW_CONTINUOUS_FEED_SOURCE_AND_RESOURCE_CANDIDATE_PASS'
    chosen=next(r for r in audit['source_runs'] if r['width']==256 and r['seed']==71)
    result=Path(chosen['source_result']['path'])
    assert sha(result)==chosen['source_result']['sha256']
    r=json.loads(result.read_text())
    lines=r['results'][0]['lines'];answer={}
    for i,shape in enumerate(SHAPES):
        rows=[tuple(map(int,m.groups())) for line in lines if (m:=re.search(
            rf'MEASURE ABC width=256 profile={i} batch=(\d+) accepted=(\d+) started=(\d+) completed=(\d+)',line))]
        ii=chosen['steady'][i]['steady_interval']
        # Whole-batch startup and drain actually present in the source fixture.
        overhead=max(0,rows[-1][3]-rows[0][1]-(len(rows)-1)*ii)
        load=next(int(m[1]) for line in lines if (m:=re.search(rf'MEASURE RESIDENT load profile={i} cycles=(\d+)',line)))
        answer[shape]=dict(ii=ii,first_and_drain=overhead,load=load)
    return answer,[dict(path=str(path),sha256=sha(path)),dict(path=str(result),sha256=sha(result))]

class Arena:
    """12 real banks: y mod3, x mod4, 64 bits, common row extent per4 banks.
    Fixed-depth, first-fit allocation; no ideal byte-capacity-only assertion.
    Each row stores ceil(C/8) words per pixel, including padded RGB.
    """
    def __init__(self,depth,columns=4,rgb_pair=False):
        self.columns=columns;self.rgb_pair=rgb_pair
        self.depth=depth;self.used=[[] for _ in range(3)];self.rows={};self.high=[0]*3
        self.peak_words=0
    def allocate(self,key,width,channels,plane_offset=0,fixed_base=None):
        plane=(key[1]+plane_offset)%3;length=ceildiv(width,self.columns*(2 if self.rgb_pair and channels==3 else 1))*ceildiv(channels,8)
        at=0
        for lo,hi,_ in sorted(self.used[plane]):
            if lo-at>=length:break
            at=max(at,hi)
        if fixed_base is not None:
            at=fixed_base
            assert at>=0 and all(at+length<=lo or at>=hi for lo,hi,_ in self.used[plane]),('live address overlap',key,at,length)
        if at+length>self.depth:raise MemoryError((key,plane,length,at,self.depth))
        self.used[plane].append((at,at+length,key));self.rows[key]=(plane,at,width,ceildiv(channels,8))
        self.high[plane]=max(self.high[plane],at+length)
        self.peak_words=max(self.peak_words,self.columns*sum(hi-lo for rows in self.used for lo,hi,_ in rows))
    def release(self,key):
        plane,*_=self.rows.pop(key)
        self.used[plane]=[r for r in self.used[plane] if r[2]!=key]
    def address(self,key,x,g):
        plane,at,width,groups=self.rows[key]
        assert 0<=x<width and 0<=g<groups
        return plane*self.columns+x%self.columns,at+(x//self.columns)*groups+g

@lru_cache(maxsize=None)
def dw_ports(width,height,groups,source_width,source_height,stride,origin,source_origin,out_y,
             nn,read_ports,stem,group_major):
    """Issue real global tap coordinates through a 12-bank/two-port contract.
    27 active DW fronts (9 perRQ bank), phase0..8, batch every9 phases.
    Stem uses28 fronts, phases3*j, launch every30 phases. Neither assumes a
    window supplied by the TB. Each requested word is counted and deduplicated.
    A phase freezes until every RAM bank finishes its reads. Output writes
    have separate explicit accounting below; no free third memory port.
    """
    jobs=width*height*groups;count=28 if stem else 27;period=30 if stem else 9
    phases=27 if stem else 9
    batches=ceildiv(jobs,count);last=(batches-1)*period+(27 if stem else 8)+phases
    ticks=reads=conflict=0;max_reads=0
    for t in range(last):
        request=[set() for _ in range(12)];active=0
        for c in range(3):
            owners=10 if stem and c==0 else 9
            start=0 if c==0 else (10 if stem else 9) if c==1 else (19 if stem else 18)
            for j in range(owners):
                phase=3*j if stem else j
                if t<phase:continue
                b,k=divmod(t-phase,period)
                index=b*count+start+j
                if b>=batches or k>=phases or index>=jobs:continue
                g,pixel=divmod(index,width*height) if group_major else (index%groups,index//groups)
                y,x=divmod(pixel,width)
                # RGB stem input has one padded OC8 source word; channel=k%3.
                tap=k//3 if stem else k
                ky,kx=divmod(tap,3)
                sx=min(source_width-1,max(0,(origin+x)*stride+kx-1))
                sy=min(source_height-1,max(0,(out_y+y)*stride+ky-1))
                if nn:sx//=2;sy//=2
                local_x=sx-source_origin
                if local_x<0:raise AssertionError(('halo_read',sx,source_origin))
                bank=sy%3*4+local_x%4
                request[bank].add((sy,local_x//4,g))
                active+=1
        if not active:continue
        service=max(ceildiv(len(q),read_ports) for q in request)
        ticks+=service;reads+=sum(map(len,request));max_reads=max(max_reads,max(map(len,request)))
        conflict+=service-1
    return dict(phases=last,cycles=ticks,bank_conflict_cycles=conflict,word_reads=reads,max_bank_reads=max_reads,
                jobs=jobs,implementation='GENERATED_PORT_SCHEDULE_NOT_RTL')

def run(block,depth,ddr_gbps,read_ports,mode):
    plan=compile_contract(load_contract());manifest=hw.build_tile_manifest(320);cal,_=calibration()
    events=[];total=Counter();peak_words=0;peaks=[0]*3;node_cycles=Counter()
    frame_cycle=0;violations=[]
    # Conservative executable schedule: source/commit DMA is serialized at
    # actual row dependencies. This is deliberately NOT an overlap proof.
    for group in manifest['groups']:
        repeats=range(1,4) if group['name']=='residual' else (0,)
        for repeat in repeats:
            for tile in group['tiles']:
                local,row_bytes,tasks=make_tasks(plan,group,tile,block)
                columns={name:row_bytes[name]//n['output_shape_hwc'][2] for name,n in local.items()}
                last_use={}
                for t in tasks:
                    for k in t['dependencies']:last_use[tuple(k)]=t['id']
                arena=Arena(depth,5 if mode=='bank5_pipeline_candidate' else 4,mode=='bank5_pipeline_candidate')
                step_by_id={s['operation'].replace('resX','res1'):s for s in tile['steps']}
                for t in tasks:
                    node=t['node'];kind=t['kind'];rows=t['hi']-t['lo']+1
                    for k in t['dependencies']:
                        assert tuple(k) in arena.rows,('use_before_ready',t,k)
                    try:
                        for k in t['outputs']:
                            n=local[k[0]];arena.allocate(tuple(k),columns[k[0]],n['output_shape_hwc'][2])
                    except MemoryError as e:
                        violations.append(dict(group=group['name'],tile=tile['tile_index'],task=t['id'],detail=str(e)))
                        return dict(block_rows=block,depth=depth,mode=mode,status='REJECTED_FINITE_BANK_CAPACITY',
                                    violations=violations,performance_pass=False,measured_frame_ms=None)
                    cost=Counter(); detail={}
                    if kind in ('source','commit'):
                        keys=t['outputs'] if kind=='source' else t['dependencies']
                        byte_count=sum(ceildiv(row_bytes[k[0]],16)*16 for k in keys)
                        #100MHz128-bit DDR, bandwidth converts explicitly to150MHz cycles.
                        cost['DDR_data']=math.ceil(byte_count*150_000_000/(ddr_gbps*1e9))
                        cost['DDR_row_control']=len(keys)*12 # explicit unvalidated service contract
                        detail['bytes']=byte_count
                    else:
                        n=local[node];w=columns[node];c=n['output_shape_hwc'][2];g=ceildiv(c,8)
                        words=w*rows*g;detail.update(width=w,rows=rows,oc8_jobs=words)
                        if kind=='pw1x1':
                            shape=(n['input_channels'],c);m=cal[shape]
                            batches=rows*ceildiv(w,28)
                            cost['PW_service']=(batches-1)*m['ii']+m['first_and_drain']
                            cost['PW_header']=m['load']
                            detail['batches']=batches
                            # More aggressive policies are explicitly candidates:
                            # no measured claim for combining rows or RGB pairs.
                            if mode in ('joint_candidate','bank5_pipeline_candidate') and c==3:
                                cost['PW_service']=ceildiv(w*rows,4)+m['first_and_drain']
                                detail['candidate']='RGB pair + continuous 256-bit physical read port, RTL pending'
                        elif kind in ('dw3x3','conv3x3'):
                            parent=n['local_inputs'][0];pn=local[parent]
                            nearest=pn['kind']=='nearest_x2'
                            # Source intervals retain true global coordinates.
                            step=step_by_id[node];src_origin=step['input_interval']['x0']
                            if nearest:src_origin//=2
                            # Interior rows with the same parity/bank phase have
                            # identical request conflicts. Preserve NN floor/2
                            # with modulus6; keep top/bottom clamp cases exact.
                            interior=t['lo']*n['stride']>0 and t['hi']*n['stride']+1<n['input_shape_hwc'][0]
                            ykey=t['lo']%6+6 if interior else t['lo']
                            hkey=10000 if interior else n['input_shape_hwc'][0]
                            p=dw_ports(w,rows,g,n['input_shape_hwc'][1],hkey,
                                n['stride'],step['output_interval']['x0'],src_origin,ykey,nearest,read_ports,kind=='conv3x3',mode=='joint_candidate')
                            if mode=='bank5_pipeline_candidate':
                                p=pipeline(w,rows,g,n['input_shape_hwc'][1],hkey,n['stride'],step['output_interval']['x0'],src_origin,ykey,nearest,kind=='conv3x3')
                            cost['DW_stem_issue']=p['cycles']
                            cost['DW_stem_drain']=12
                            # Read either full DW weights or RGB terms into an
                            # actual per-layer header/weight staging contract.
                            cost['DW_stem_load']=ceildiv(n['raw_weights'],40)+c*2+4
                            cost['DW_stem_result_write']=0 if mode=='bank5_pipeline_candidate' else ceildiv(words,3)
                            if mode=='bank5_pipeline_candidate':cost['DW_stem_drain']=0
                            detail['port_schedule']=p
                        elif kind=='residual_add':
                            # Existing convolution RQ cannot output the two
                            # required INT32 branches. Count the proposed work,
                            # never silently replace it with an int8 operation.
                            cost['residual_alignment']=2*ceildiv(words,3)
                            cost['residual_post_RQ']=ceildiv(words,3)+12
                            cost['residual_config']=12
                            if mode in ('joint_candidate','bank5_pipeline_candidate'):
                                # Candidate logic alignment may stream into the
                                # same24 RQ lanes, but must pass numerical/map.
                                cost['residual_alignment']=4
                                detail['candidate']='parallel INT32 alignment into existing RQ, RTL pending'
                        else:raise AssertionError(kind)
                        cost['task_control']=2
                    cycles=sum(cost.values());assert cycles>0
                    if node not in ('source','commit'):
                        name=node.replace('res1',f'res{repeat}') if repeat else node
                        node_cycles[name]+=cycles
                    events.append(dict(sequence=len(events),group=group['name'],repeat=repeat,tile=tile['tile_index'],
                        task=t['id'],node=node,kind=kind,lo=t['lo'],hi=t['hi'],start=frame_cycle,end=frame_cycle+cycles,
                        cost=dict(cost),detail=detail))
                    frame_cycle+=cycles;total.update(cost)
                    for k in t['dependencies']:
                        k=tuple(k)
                        if last_use[k]==t['id']:arena.release(k)
                    peak_words=max(peak_words,arena.peak_words);peaks=[max(a,b) for a,b in zip(peaks,arena.high)]
                assert not arena.rows
    expected={n['id'] for n in plan['nodes'] if n['kind']!='nearest_x2'}
    assert set(node_cycles)==expected,expected-set(node_cycles)
    # NN is consumed by real tap-coordinate division above; materialized
    # writes are not invented, but controller/FIFO implementation remains open.
    overhead=dict(gesture_conv=6490,gesture_GAP_FC_contract=4096,frame_commit_contract=128)
    total.update(overhead);frame_cycle+=sum(overhead.values())
    assert sum(total.values())==frame_cycle
    return dict(block_rows=block,depth=depth,read_ports=read_ports,mode=mode,ddr_gbps=ddr_gbps,
        status='COMPLETE_MAIN_GRAPH_SCHEDULING_SIMULATION_NOT_RTL',
        simulated_main_graph_plus_declared_overheads_cycles=frame_cycle,
        simulated_ms=frame_cycle/150000,simulated_fps=150000000/frame_cycle,
        components=dict(total),per_node=dict(node_cycles),
        finite_bank_high_water=peaks,peak_live_padded_bytes=peak_words*8,
        arena_ram10_contract=(15*4*ceildiv(depth,512) if mode=='bank5_pipeline_candidate' else 12*7*ceildiv(depth,1024)),
        performance_pass=False,measured_frame_ms=None,production_rtl_complete=False,
        other_video_DDR_included=False,
        assumptions=['Single shared compute pool, layer/task serialization; no compute double booking.',
            'DW/stem phase and bank conflicts enumerated, but not yet RTL-calibrated.',
            'DW result writes conservatively serialized; overlap requires implementation proof.',
            'DDR source/commit transfers serialized; measured bandwidth still unavailable.',
            'EFX_DPRAM10 width10 uses7 primitives per64bit bank;12 banks=84 blocks atdepth1024.',
            'Current_services DW is pixel/group-interleaved; joint_candidate uses group-major spatial jobs.',
            'DDR row control12/task2/gesture heads4096/frame commit128 are explicit design contracts, not measurements.',
            'Camera/display/CPU competing DDR and full baseline pipelines require system co-simulation.',
            'NN memory view accounts for clamped coordinates; its production control remains unimplemented.',
            'Residual alignment semantics cannot be implemented by existing int8 convolution RQ.'],
        trace=events)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bank5-only',action='store_true');a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=False)
    cal,evidence=calibration();results=[]
    for mode in (('bank5_pipeline_candidate',) if a.bank5_only else ('serial_reference','joint_candidate')):
        for block in (1,2,4,8):
            r=run(block,512 if mode=='bank5_pipeline_candidate' else 1024,1.28,1 if mode=='bank5_pipeline_candidate' else 2,mode)
            trace=r.pop('trace',[])
            f=a.root/f'{mode}-rows{block}.json'
            f.write_text(json.dumps(dict(result=r,trace=trace),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            r['trace_file']=str(f.resolve());r['trace_sha256']=sha(f);results.append(r)
            print(json.dumps(r,ensure_ascii=False),flush=True)
    evidence += [dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),
        ROOT/'network_ab/model_b_contract.json',HERE/'pipeline_cost.py',HERE.parent/'full_frame_closure/wavefront_lifetimes.py')]
    result=dict(status='FULL_GRAPH_DESIGN_SCREEN_NOT_PERFORMANCE_PASS',cases=results,evidence=evidence,
                local_PW_calibration={str(k):v for k,v in cal.items()})
    (a.root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()

