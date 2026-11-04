"""Read-only review of existing placement evidence; never runs implementation.

Metrics are tool estimates, not physical track-capacity measurements. Net
statistics below cover known FF/LUT/ADD/SRL scalar drivers only; do not label
them the complete routed design or the exact unresolved routing conflicts.
"""
import argparse,collections,json,re
from pathlib import Path
from run_units import ROOT,sha
from fixed_control_ddr_reuse_locations import read,locations

CASES={
 'passing_control':'command-progress-seed3-1/pnr-seed1',
 'same_map_start':'control-ddr-combined-physical4/pnr-seed1',
 'large_reuse':'control-ddr-reuse-physical1/pnr-seed1',
 'control_reuse':'control-ddr-control-reuse-physical1/pnr-seed1',
 'lut_only':'control-ddr-lut-physical1/pnr-seed1',
}
OUTPUTS={'EFX_FF':('Q',),'EFX_LUT4':('O',),'EFX_ADD':('O','CO'),'EFX_SRL8':('Q',)}
CLOCKS={'CLK','WCLK','RCLK','CLKA','CLKB'}
RESETS={'SR','RST','RSTA','RSTB'}

def metrics(folder):
    report=folder/'outflow/ti60f225_oob.place.out';text=report.read_text()
    peaks={}
    for size,value,x,y in re.findall(r'Peak congestion smeared over (.*?) of the chip is ([\d.]+) at (\d+),(\d+)',text):
        peaks[size]=dict(value=float(value),xy=[int(x),int(y)])
    assert set(peaks)=={'4x4','1/16','1/4'}
    hpwl=float(re.search(r'Congestion-weighted HPWL per net: ([\d.]+)',text)[1])
    fixed=folder/'fixed_dsp.placeloc'
    resource=folder/'outflow/ti60f225_oob.place.rpt'
    xlr=int(re.search(r'XLRs: (\d+)',resource.read_text())[1])
    result=dict(folder=str(folder),peaks=peaks,weighted_hpwl=hpwl,xlr=xlr,
        fixed_primitive_count=len(locations(fixed)) if fixed.exists() else None,
        evidence={str(p):sha(p) for p in (report,resource)})
    if fixed.exists():result['evidence'][str(fixed)]=sha(fixed)
    return result

def graph_review(folder):
    net=folder/'outflow/ti60f225_oob.map.v';pos=folder/'outflow/ti60f225_oob.out.placeloc'
    cc=read(net);xy=locations(pos);drivers={};old_loads=collections.defaultdict(list)
    all_loads=collections.defaultdict(set)
    for n,(k,ports,params) in cc.items():
        for pin,v in ports.items():
            if pin in OUTPUTS.get(k,()):drivers[v]=n
            elif pin not in CLOCKS and '{' not in v and v not in ("1'b0","1'b1"):
                all_loads[v].add(n)
                if pin not in RESETS:old_loads[v].append(n)
    skipped=[];all_high=[]
    for signal,src in drivers.items():
        if len(old_loads[signal])>128:
            sinks=[n for n in set(old_loads[signal]) if n in xy and n!=src]
            pts=[xy[n] for n in sinks]+[xy[src]]
            skipped.append(dict(signal=signal,driver=src,old_cost_load_entries=len(old_loads[signal]),
                distinct_sink_cells=len(sinks),bbox=[min(p[0] for p in pts),min(p[1] for p in pts),max(p[0] for p in pts),max(p[1] for p in pts)]))
        if len(all_loads[signal]-{src})>128:
            all_high.append(dict(signal=signal,distinct_sink_cells=len(all_loads[signal]-{src})))
    return dict(mapped_primitive_count=len(cc),
        omitted_by_old_cost_count=len(skipped),omitted_by_old_cost=sorted(skipped,key=lambda r:r['old_cost_load_entries'],reverse=True),
        high_fanout_including_reset_count=len(all_high),high_fanout_including_reset=sorted(all_high,key=lambda r:r['distinct_sink_cells'],reverse=True),
        scope='Known FF/LUT/ADD/SRL scalar drivers; clocks excluded; no expansion of vector expressions; not exact route-conflict attribution.',
        evidence={str(net):sha(net),str(pos):sha(pos)})

def review():
    rows={name:metrics(ROOT/'tmp/frame-execution'/p) for name,p in CASES.items()}
    old,new=rows['same_map_start'],rows['lut_only']
    graph=graph_review(Path(new['folder']))
    changes=dict(weighted_hpwl_pct=100*(new['weighted_hpwl']/old['weighted_hpwl']-1),
        **{size:100*(new['peaks'][size]['value']/old['peaks'][size]['value']-1) for size in old['peaks']})
    reasons=[]
    if new['weighted_hpwl']>old['weighted_hpwl'] and all(new['peaks'][s]['value']>old['peaks'][s]['value'] for s in ('1/16','1/4')):
        reasons.append('Local peak improved but weighted wirelength and both broader congestion estimates worsened.')
    if new['fixed_primitive_count']==graph['mapped_primitive_count']:
        reasons.append('All mapped primitives fixed; placer cannot redistribute mapped logic to relieve congestion.')
    if graph['omitted_by_old_cost_count']:
        reasons.append('Distance-only cost excluded high-fanout nets and did not model control-network/site restrictions.')
    assert len(reasons)==3
    # A passing design has a larger 4x4 estimate: no invented absolute >1 gate.
    assert rows['passing_control']['peaks']['4x4']['value']>new['peaks']['4x4']['value']
    return dict(status='REJECT_OLD_GEOMETRIC_REPAIR_AS_ROUTABILITY_JUSTIFICATION',cases=rows,
        same_map_changes_percent=changes,graph=graph,reasons=reasons,
        exact_unresolved_nets_available=False,can_predict_guaranteed_route_success=False,
        hardware_access=False,new_implementation_started=False,generator_sha256=sha(__file__))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert out.is_relative_to(ROOT/'output/toolchain') and not out.exists()
    result=review();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','same_map_changes_percent','reasons')}))
    print('OLD_COST_OMITTED',result['graph']['omitted_by_old_cost_count'])
