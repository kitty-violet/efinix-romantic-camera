"""Include alternate combinational inputs and explicitly reviewed CE loads.

Sequential elements terminate the backward walk. No clock/reset/carry/cascade
route is locked, and every retained trace comes from one legal implementation.
"""
import collections,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import graph,lexical_tokens
from probe_full_native_split_map import params
from survey_camera_near_critical import parse
from camera_route_cone_plan import OUTPUTS,BASE,SOURCE

SPECIAL=re.compile(r'CLK|RESET|RST|^SR$|^WE|^RE$|ADDREN|CASC|^CI$|^CO$')
def label(t):return t[1]+('['+str(t[2])+']' if t[2] is not None else '')

def build():
    reports=[BASE/'camera-near-critical1/outflow/near_setup.rpt',
        *[BASE/f'camera-checked-route{i}/canonical-seed3/outflow/canonical_setup.rpt' for i in (3,5,7,8)],
        BASE/'camera-checked-route6/canonical-rcf-complete1/outflow/canonical_setup.rpt']
    paths=[p for f in reports for p in parse(f.read_text())]
    seeds={n['net'] for p in paths for n in p['nets'] if n['delay_ns']>0}
    reported=collections.defaultdict(set)
    for p in paths:
        for n in p['nets']:reported[n['net']].add(n['fanout'])
    bad={n['net'] for p in parse(reports[0].read_text()) if p['slack_ns']<0 for n in p['nets']}
    assert len(bad)==3
    source=SOURCE/'outflow/ti60f225_oob.map.v';text=source.read_text();g=graph(text);attributes=params(text)
    positions={p[0]:list(map(int,p[1:])) for line in (SOURCE/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines()
        if not line.startswith('#') and len(p:=line.split())==4}
    index=collections.defaultdict(list);drivers=collections.defaultdict(list);aliases=set()
    for lhs,rhs in re.findall(r'^\s*assign\s+(.+?)\s*=\s*(.+?);',text,re.M):
        for expr in (lhs,rhs):aliases.update(t for t in lexical_tokens(re.sub(r'/\*.*?\*/','',expr)) if t[0]=='net')
    for name,(kind,pins) in g.items():
        for port,ts in pins.items():
            for t in set(ts):
                if t[0]!='net':continue
                index[label(t)].append((name,kind,port,t))
                if port in OUTPUTS.get(kind,set()):drivers[t].append((name,kind,port,t))
    queue=[t for n in seeds for *_,t in index[n]];seen=set();boundaries=collections.Counter()
    while queue:
        t=queue.pop()
        if t in seen:continue
        seen.add(t)
        ds=drivers[t]
        if len(ds)!=1 or t in aliases:boundaries['unresolved_or_alias']+=1;continue
        name,kind,_,_=ds[0]
        if kind in ('EFX_LUT4','EFX_ADD'):
            queue.extend(q for port,ts in g[name][1].items() if port not in OUTPUTS[kind] for q in ts if q[0]=='net')
        else:boundaries[kind]+=1
    available_file=BASE/'camera-checked-bound8/outflow/ti60f225_oob.rcf.template'
    assert sha(source)==sha(BASE/'camera-checked-bound8/outflow/ti60f225_oob.map.v')
    available=set(re.findall(r'^restoreNetFromTraceFile (.+)$',available_file.read_text(),re.M))
    kept=[];excluded=[];ce_rows=[]
    for net in sorted({label(t) for t in seen}):
        terms=index[net];ds=[x for x in terms if x[2] in OUTPUTS.get(x[1],set())]
        why=[]
        if len(ds)!=1 or len({x[3] for x in terms})!=1:why.append('ambiguous_native_label')
        if any(x[3] in aliases for x in terms):why.append('assign_alias_not_locked')
        if not why:
            driver=ds[0];loads=[x for x in terms if x!=driver]
            if not loads:why.append('no_direct_loads')
            if net in reported and reported[net]!={len(loads)}:why.append('sta_native_fanout_mismatch')
        if net in bad:why.append('original_failed_path_released')
        if net not in available:why.append('not_in_same_map_router_template')
        if any(n not in positions or n.startswith('protected_inst/') for n,_,_,_ in terms):why.append('non_public_or_no_position')
        if any(SPECIAL.search(p) for _,_,p,_ in terms):why.append('clock_reset_memory_control_or_dedicated_port')
        enabled=[]
        if not why:
            for name,kind,port,_ in loads:
                if port!='CE':continue
                a=attributes[name];pins=g[name][1]
                if kind not in ('EFX_FF','EFX_SRL8','EFX_DSP24') or pins['CLK']!=[('net','process_clk_150',None)]:
                    why.append('unreviewed_enable_clock_or_type');break
                assert a['CLK_POLARITY']==(1,1) and a['CE_POLARITY'] in ((1,0),(1,1))
                if kind=='EFX_FF':assert a['SR_SYNC']==(1,1)
                if kind=='EFX_DSP24':assert a['RST_SYNC']==(1,1)
                enabled.append(dict(cell=name,kind=kind,clock=pins['CLK'],reset=pins.get('SR',pins.get('RST',[])),attributes=a))
        if why:
            excluded.append(dict(net=net,reasons=why));continue
        kept.append(dict(net=net,driver=driver[:3],loads=[x[:3] for x in loads],locations={x[0]:positions[x[0]] for x in terms}))
        if enabled:ce_rows.append(dict(net=net,loads=enabled))
    assert 2000<len(kept)<5000 and not bad.intersection(r['net'] for r in kept)
    files=[*reports,source,SOURCE/'outflow/ti60f225_oob.place',SOURCE/'outflow/ti60f225_oob.out.placeloc',
        SOURCE/'outflow/ti60f225_oob.troutingtraces',SOURCE/'ti60f225_oob.sdc',available_file,Path(__file__),
        HERE/'camera_route_cone_plan.py',HERE/'native_local_drivers.py',HERE/'probe_full_native_split_map.py',HERE/'survey_camera_near_critical.py']
    return dict(status='COMPLETE_COMBINATIONAL_FANIN_AND_CE_REVIEW_NOT_PASS',rows=kept,excluded=excluded,
        enable_review=ce_rows,boundaries=dict(boundaries),reported_paths=len(paths),seed_nets=len(seeds),
        expanded_tokens=len(seen),preserved_nets=len(kept),total_router_nets=79318,preserved_percent=100*len(kept)/79318,
        excluded_failed_path_nets=sorted(bad),source_files={str(f):sha(f) for f in files},rationale=[
        'Route8 failed at an alternate address-check input to the writer CE cone. One worst path per endpoint does not enumerate other combinational inputs.',
        'Walk every LUT/ADD input backward from all measured path nets, terminating at sequential/storage primitives or unresolved aliases. Review complete native loads and source locations for the resulting cone.',
        'Review each retained FF/SRL/DSP CE load at process150, synchronous FF/DSP reset, actual polarity and immutable attributes. Different existing reset groups are recorded, not relocated or merged.',
        'Retain only unambiguous public nets in the same-map routing template, all from one mutually legal source4 trace. Release original3 failed nets, clocks, resets, memory control and dedicated carry/cascade.',
        'This is a larger targeted constraint set and can increase congestion; size is no guarantee. No placement/map/clock/PHY changes; one1350s route attempt followed by complete original STA and all release gates.'],hardware_access=False)

if __name__=='__main__':
    p=build();print(json.dumps({k:v for k,v in p.items() if k not in ('rows','excluded','enable_review','source_files')},indent=2))
