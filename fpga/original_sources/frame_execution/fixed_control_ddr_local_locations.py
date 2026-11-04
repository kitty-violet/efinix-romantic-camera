"""Move complete tiles on a legal same-map route, driven by expanded STA.

Coordinates come only from previously exported legal same-device placements.
The cost function is a geometric heuristic, never a timing or release claim.
"""
import argparse,collections,hashlib,json,math,re
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import read,locations,REFERENCE
from audit_canonical_multiseed import audit

BASE=ROOT/'tmp/frame-execution/control-ddr-route100-c2/canonical-seed1'
SURVEY=ROOT/'tmp/frame-execution/control-ddr-negative-paths2'

def plan(root):
    prior=audit(BASE);assert prior['wns_ns']==-3.927 and prior['whs_ns']>=0
    physical=Path(prior['physical']);net=physical/'outflow/ti60f225_oob.map.v'
    assert sha(net)==sha(Path(root)/'outflow/ti60f225_oob.map.v')
    report=SURVEY/'outflow/canonical_setup.rpt';result=json.loads((SURVEY/'result.json').read_text())
    assert result['reports']['canonical_setup.rpt']==sha(report) and result['original_and_route_unchanged']
    cc=read(net);allpos=locations(physical/'outflow/ti60f225_oob.out.placeloc');assert cc.keys()<=allpos.keys()
    tiles=collections.defaultdict(list)
    for n,p in allpos.items():tiles[p[:2]].append(n)
    assigned={n:allpos[n] for n in cc};initial=dict(assigned)
    neg=[];critical=collections.defaultdict(float);sources=collections.Counter()
    for block in report.read_text().split('++++ Path ')[1:]:
        slack=float(re.search(r'Slack\s*:\s*([-+.\d]+)',block)[1])
        if slack>=0:continue
        start=re.search(r'Path Start\s*:\s*(\S+)',block)[1]
        end=re.search(r'Path End\s*:\s*(\S+)',block)[1];sources[start]+=1
        data=block.split('\nData Path\n')[1].split('\nCapture Clock Path')[0]
        for fanout,delay,signal in re.findall(r'net \(fo=(\d+)\)\s+([\d.]+)\s+[\d.]+\s+(\S+)',data):
            critical[signal]=max(critical[signal],float(delay))
        neg.append(dict(slack_ns=slack,start=start,end=end))
    assert len(neg)==508
    outputs={'EFX_FF':('Q',),'EFX_LUT4':('O',),'EFX_ADD':('O','CO'),'EFX_SRL8':('Q',)}
    driver={};loads=collections.defaultdict(list)
    for n,(k,p,params) in cc.items():
        for pin,v in p.items():
            if pin in outputs.get(k,()):driver[v]=n
            elif pin not in ('CLK','WCLK','RCLK','CLKA','CLKB','SR','RST','RSTA','RSTB'):
                if v in ("1'b0","1'b1") or '{' in v:continue
                loads[v].append(n)
    # Previously used legal LUT/FF sites determine compatible vacant choices.
    passing=Path(audit(REFERENCE)['physical']);oldc=read(passing/'outflow/ti60f225_oob.map.v')
    oldp=locations(passing/'outflow/ti60f225_oob.out.placeloc');valid=collections.defaultdict(set)
    for n,(k,p,par) in oldc.items():
        if n in oldp and k in ('EFX_LUT4','EFX_FF'):valid[k].add(oldp[n][:2])
    occupied=set(tiles);moves=[]
    movable={xy:ns for xy,ns in tiles.items() if all(n in cc and cc[n][0] in ('EFX_LUT4','EFX_FF') for n in ns)
        and all(n.startswith('video_processing_pipeline/') or cc[n][0]=='EFX_LUT4' for n in ns)}
    edges=collections.defaultdict(list)
    repair_drivers={driver[s] for s,d in critical.items() if d>=1.0 and s in driver}
    for signal,src in driver.items():
        sinks=loads.get(signal,[])
        if not sinks or len(sinks)>128:continue
        weight=1.0+min(critical.get(signal,0),4)
        for dst in sinks:
            if dst!=src:
                edges[src].append((dst,weight));edges[dst].append((src,weight))
    def cost(ns,xy):
        return sum(w*(abs(xy[0]-assigned[o][0])+abs(xy[1]-assigned[o][1]))
                   for n in ns for o,w in edges[n] if o not in ns and o in assigned)
    ranked=[]
    for xy,ns in movable.items():
        if not set(ns)&repair_drivers:continue
        kinds={cc[n][0] for n in ns}
        possible=set.intersection(*(valid[k] for k in kinds))-occupied
        possible={p for p in possible if abs(p[0]-xy[0])+abs(p[1]-xy[1])<=32}
        if not possible:continue
        before=cost(ns,xy);target=min(possible,key=lambda p:(cost(ns,p),p))
        gain=before-cost(ns,target)
        if gain>5 and gain>before*0.02:ranked.append((gain,xy,target,ns,before))
    # A bounded local change instead of another global floorplan.
    primary={'LUT__171161','video_processing_pipeline/pipeline/cnn/core/dut/row_banks[10]~FF',
        'video_processing_pipeline/pipeline/cnn/core/dut/row_bases[10]~FF'}
    for gain,xy,target,ns,before in sorted(ranked,key=lambda r:(bool(set(r[3])&primary),r[0]),reverse=True):
        if len(moves)>=16:break
        if target in occupied:continue
        after=cost(ns,target)
        if after>=cost(ns,xy)*0.98:continue
        for n in ns:assigned[n]=(*target,assigned[n][2])
        occupied.add(target)
        moves.append(dict(cells=ns,before=list(xy),after=list(target),weighted_distance_before=cost(ns,xy),weighted_distance_after=after))
    assert 3<=len(moves)<=16 and len(set(assigned.values()))==len(assigned)
    assert any('LUT__171161' in m['cells'] for m in moves),'The worst actual RQ net must be addressed'
    assert all(cc[n][0] in ('EFX_LUT4','EFX_FF') for n in assigned if initial[n]!=assigned[n])
    return dict(status='SAME_MAP_LOCAL_REPAIR_NOT_TIMING_PASS',reference=str(BASE),
        reference_result_sha256=sha(BASE/'result.json'),reference_timing_pass_required=False,
        old_map_sha256=sha(net),new_map_sha256=sha(net),old_placement_sha256=sha(physical/'outflow/ti60f225_oob.out.placeloc'),
        survey_result_sha256=sha(SURVEY/'result.json'),survey_report_sha256=sha(report),survey=str(SURVEY),
        destination_reference=str(REFERENCE),destination_reference_placement_sha256=sha(passing/'outflow/ti60f225_oob.out.placeloc'),
        destination_reference_map_sha256=sha(passing/'outflow/ti60f225_oob.map.v'),
        assignment_count=len(assigned),assignments=assigned,moves=moves,negative_endpoint_paths=neg,
        source_groups=dict(sources),generator_sha256=sha(__file__),logic_or_constraints_changed=False,hardware_access=False)

def generate(root):
    root=Path(root);proof=plan(root);target=root/'fixed_dsp.placeloc'
    assert all(tuple(proof['assignments'][n])==v for n,v in locations(target).items())
    target.write_text('# Same legal mapped layout; bounded complete-tile moves from measured STA.\n'+''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in sorted(proof['assignments'].items())))
    proof['assignments_sha256']=sha(target)
    (root/'reuse_placement.json').write_text(json.dumps(proof,indent=2))
    (root/'reuse_placement_generator.py').write_bytes(Path(__file__).read_bytes());return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();r=plan(a.board);a.output.write_text(json.dumps(r,indent=2))
    print(json.dumps(dict(assignments=r['assignment_count'],moved_tiles=len(r['moves']),moved_cells=sum(len(m['cells']) for m in r['moves']),moves=r['moves']),indent=2))
