"""Whole-map input/RAM/edge geometry comparison, not a routability verdict."""
import argparse, collections, json, pickle, statistics
from pathlib import Path
from run_units import ROOT, sha
from continuous_control_native import connections


def locations(path):
    return {a[0]:tuple(map(int,a[1:])) for s in path.read_text().splitlines()
            if len(a:=s.split())==4 and not s.startswith('#')}


def group(n):
    if n.startswith('raw_frontend/consumer/serializer/'):return 'input_serializer'
    if n.startswith('raw_frontend/consumer/debayer/'):return 'debayer'
    if n.startswith('raw_frontend/'):return 'raw_ingress_feeder'
    if n.startswith('video_processing_pipeline/downscaler/'):return 'downscaler'
    if n.startswith(('service_isp/','service_isp_')):return 'isp'
    if n.startswith('video_processing_pipeline/pipeline/cnn/'):return 'cnn'
    if n.startswith(('cpu/','service_')):return 'service_control'
    return 'other'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--physical',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    board=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
    cache=board/'service-validation-graph.pkl'
    key,g=pickle.load(open(cache,'rb'))
    assert key[0]==sha(board/'outflow/ti60f225_oob.map.v')
    driver,loads=connections(g)
    paths=[f.resolve() for f in [a.reference,a.physical]]
    data=[]
    for root in paths:
        assert sha(root/'outflow/ti60f225_oob.map.v')==key[0]
        loc=locations(root/'outflow/ti60f225_oob.out.placeloc')
        groups=collections.defaultdict(list)
        for n,xy in loc.items():groups[group(n)].append(xy)
        stats={name:dict(named_cells=len(v),median_xy=[statistics.median(q[i] for q in v) for i in range(2)],
                        bbox=[min(q[0] for q in v),min(q[1] for q in v),max(q[0] for q in v),max(q[1] for q in v)])
               for name,v in groups.items()}
        lengths=collections.defaultdict(lambda:dict(pin_connections=0,sum_manhattan=0,max_manhattan=0))
        missing=collections.Counter()
        for token,sinks in loads.items():
            origin=driver.get(token)
            if origin not in loc:continue
            for target,pin in sinks:
                # Dedicated clock/reset/enable and data paths reported apart.
                family='clock_reset_enable' if pin in ('CLK','CLKA','CLKB','RCLK','WCLK','SR','CE','RCE','WCE','CEA','CEB','CLKE','RST') else 'data'
                ga,gb=group(origin),group(target)
                if not ({ga,gb}&{'isp','downscaler','input_serializer','debayer','raw_ingress_feeder'}):continue
                if target not in loc:missing[target]+=1;continue
                distance=sum(abs(loc[origin][i]-loc[target][i]) for i in range(2))
                row=lengths[family+':'+ga+'->'+gb]
                row['pin_connections']+=1;row['sum_manhattan']+=distance
                row['max_manhattan']=max(row['max_manhattan'],distance)
        ram={n:list(loc[n]) for n,(kind,_) in g.items() if kind in ('EFX_RAM10','EFX_DPRAM10') and n in loc}
        assert len(ram)==245
        data.append(dict(physical=str(root),groups=stats,connections=dict(lengths),ram_locations=ram,
                         missing_targets=dict(missing),placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc')))
    left,right=data
    changes=[]
    for n,old in left['ram_locations'].items():
        new=right['ram_locations'][n]
        if old!=new:changes.append(dict(cell=n,group=group(n),before=old,after=new,
                                      manhattan=sum(abs(new[i]-old[i]) for i in range(2))))
    result=dict(status='INPUT_REGION_GEOMETRY_COMPARISON_NOT_ROUTE_PASS',comparison=data,
                moved_ram=changes,moved_ram_count=len(changes),
                map_sha256=key[0],script_sha256=sha(__file__),
                limitations=['Named groups omit anonymous logic membership.',
                             'Pin-weighted Manhattan sums are not actual wirelength, congestion or timing.',
                             'Clock/reset/enable geometry does not replace shared-network or routed checks.'],
                hardware_access=False,board_release=False)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(status=result['status'],moved_ram_count=len(changes))))


if __name__=='__main__':main()
