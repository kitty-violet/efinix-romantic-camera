"""Review complete fanouts for the measured timing cones; no hardware changes."""
import collections,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import graph
from survey_camera_near_critical import parse
from probe_full_native_split_map import params

BASE=ROOT/'tmp/frame-execution'
SOURCE=BASE/'camera-checked-bound4'
OUTPUTS={'EFX_FF':{'Q'},'EFX_LUT4':{'O'},'EFX_SRL8':{'Q'},
         'EFX_ADD':{'O','CO'},'EFX_RAM10':{'RDATA'},
         'EFX_DPRAM10':{'RDATAA','RDATAB'},'EFX_DSP24':{'O','CASCOUT'}}
SPECIAL=re.compile(r'CLK|RESET|RST|^SR$|^CE$|^WE|^RE$|ADDREN|CASC|^CI$|^CO$')

def build():
    reports=[BASE/'camera-near-critical1/outflow/near_setup.rpt',
             *[BASE/f'camera-checked-route{i}/canonical-seed3/outflow/canonical_setup.rpt' for i in (3,5,7)],
             BASE/'camera-checked-route6/canonical-rcf-complete1/outflow/canonical_setup.rpt']
    rows=[];reasons=collections.defaultdict(list);reported_fanouts=collections.defaultdict(set)
    for report in reports:
        paths=parse(report.read_text())
        for p in paths:
            for n in p['nets']:
                if n['delay_ns']>0:
                    reasons[n['net']].append(dict(report=str(report),end=p['end'],slack_ns=p['slack_ns']))
                    reported_fanouts[n['net']].add(n['fanout'])
        rows+=paths
    bad={n['net'] for p in parse(reports[0].read_text()) if p['slack_ns']<0 for n in p['nets']}
    assert len(bad)==3
    text=(SOURCE/'outflow/ti60f225_oob.map.v').read_text();g=graph(text);attributes=params(text)
    positions={}
    for line in (SOURCE/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines():
        p=line.split()
        if len(p)==4 and not line.startswith('#'):positions[p[0]]=list(map(int,p[1:]))
    index=collections.defaultdict(list)
    for name,(kind,pins) in g.items():
        for port,ts in pins.items():
            for token in set(ts):
                if token[0]!='net':continue
                label=token[1]+('['+str(token[2])+']' if token[2] is not None else '')
                if label in reasons:index[label].append((name,kind,port,token))
    kept=[];excluded=[]
    for net in sorted(reasons):
        terms=index[net];assert terms,net
        drivers=[x for x in terms if x[2] in OUTPUTS.get(x[1],set())]
        # STA prints escaped scalar names and vector bit selects identically.
        # Do not constrain an ambiguous label or invent an alias resolution.
        if len(drivers)!=1 or len({x[3] for x in terms})!=1:
            excluded.append(dict(net=net,reasons=['ambiguous_native_net_label_not_locked'],terminals=len(terms)))
            continue
        driver=drivers[0];loads=[x for x in terms if x!=driver]
        if not loads:
            excluded.append(dict(net=net,reasons=['alias_only_no_direct_loads_not_locked'],terminals=len(terms)))
            continue
        assert all(x[3]==driver[3] for x in terms),net
        if reported_fanouts[net]!={len(loads)}:
            excluded.append(dict(net=net,reasons=['report_native_fanout_mismatch_not_locked'],
                native_loads=len(loads),reported_fanouts=sorted(reported_fanouts[net])))
            continue
        why=[]
        if net in bad:why.append('original_failed_path_released')
        if any(n not in positions or n.startswith('protected_inst/') for n,_,_,_ in terms):why.append('non_public_or_no_location')
        if net=='local_driver_net_261':
            assert driver[1:3]==('EFX_LUT4','O') and len(loads)==18
            for name,kind,port,_ in loads:
                assert kind=='EFX_FF' and port=='CE'
                assert g[name][1]['CLK']==[('net','process_clk_150',None)]
                assert g[name][1]['SR']==[('net','video_processing_pipeline/pipeline/cnn/core/dut/engine/wsr',5)]
                assert attributes[name]=={'CLK_POLARITY':(1,1),'CE_POLARITY':(1,1),'SR_POLARITY':(1,0),'D_POLARITY':(1,1),'SR_SYNC':(1,1),'SR_VALUE':(1,0),'SR_SYNC_PRIORITY':(1,1)}
        elif any(SPECIAL.search(port) for _,_,port,_ in terms):why.append('clock_reset_enable_or_dedicated_port_not_locked')
        if why:
            excluded.append(dict(net=net,reasons=why,fanout=len(loads)))
            continue
        kept.append(dict(net=net,driver=driver[:3],loads=[x[:3] for x in loads],
                         locations={x[0]:positions[x[0]] for x in terms}))
    # This bound only prevents accidental whole-design locking. It is not a
    # physical feasibility threshold and does not substitute for routing/STA.
    assert 200<len(kept)<400 and len(kept)/79318<0.005
    assert not bad.intersection(r['net'] for r in kept)
    files=[*reports,SOURCE/'outflow/ti60f225_oob.map.v',SOURCE/'outflow/ti60f225_oob.place',
           SOURCE/'outflow/ti60f225_oob.out.placeloc',SOURCE/'outflow/ti60f225_oob.troutingtraces',
           SOURCE/'ti60f225_oob.sdc',Path(__file__),HERE/'survey_camera_near_critical.py',HERE/'native_local_drivers.py']
    return dict(status='MEASURED_CONES_REVIEWED_NOT_PHYSICAL_PASS',rows=kept,excluded=excluded,
        report_count=len(reports),reported_paths=len(rows),candidate_nets=len(reasons),
        preserved_nets=len(kept),total_router_nets=79318,
        preserved_percent=100*len(kept)/79318,excluded_failed_path_nets=sorted(bad),
        source_files={str(f):sha(f) for f in files},
        rationale=[
          'Protect ordinary data nets of200 distinct near-critical endpoints and all reported regressed paths inroutes3/5/6/7 together, instead of chasing one worst path per run.',
          'All restored traces come from one mutually legal route4. Complete driver/load terminals, escaped identifiers and fixed coordinates are checked against the same immutable map/place.',
          'Release all3 nets ofroute4 failed RAM path. Clocks, resets, dedicated carry/cascade and control-enable nets remain free, except the separately reviewed18-load synchronous CE tree.',
          'Less than0.5percent of79318 routed nets are retained. This is a scope limit, not proof of closure; unprotected paths can still regress and excessive locking can hurt routing.',
          'No placement or synthesis rerun, no latency/frequency/PHY/constraint changes. Legal routing and strict whole-design original STA are required before fresh CDC/Gray and release checks.'],hardware_access=False)

if __name__=='__main__':
    p=build();print(json.dumps({k:v for k,v in p.items() if k not in ('rows','source_files','excluded')},indent=2))
