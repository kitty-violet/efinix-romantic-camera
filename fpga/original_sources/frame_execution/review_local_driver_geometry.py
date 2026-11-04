"""Compare actual local-driver placement; geometry is not a timing proof."""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph,OUTPUT

def positions(path):
    out={}
    for line in path.read_text().splitlines():
        parts=line.split()
        if len(parts)==4 and not line.startswith('#'):out[parts[0]]=tuple(map(int,parts[1:3]))
    return out

def inspect(base,groups):
    net=base/'outflow/ti60f225_oob.map.v';place=base/'outflow/ti60f225_oob.out.placeloc'
    g=graph(net.read_text());xy=positions(place)
    drivers={n:g[n][1][OUTPUT[g[n][0]]][0] for n in groups}
    by_net={t:n for n,t in drivers.items()};assert len(by_net)==len(drivers)
    loads=collections.defaultdict(list)
    for n,(kind,pins) in g.items():
        for pin,bits in pins.items():
            for i,t in enumerate(bits):
                if t not in by_net:continue
                src=by_net[t]
                if n==src and pin==OUTPUT[g[src][0]]:continue
                assert n in xy and src in xy,(n,src)
                category='clock' if pin in ('CLK','WCLK','RCLK','CLKA','CLKB') else ('reset' if pin in ('SR','RST','RSTA','RSTB') else ('control' if pin in ('CE','WE','CLKEA','CLKEB','WEA','WEB') else 'data'))
                distance=sum(abs(xy[n][d]-xy[src][d]) for d in (0,1))
                loads[src].append(dict(cell=n,pin=pin,bit_index=i,category=category,distance=distance))
    result={}
    for n in groups:
        ll=loads[n];assert ll,n
        pts=[xy[n]]+[xy[x['cell']] for x in ll]
        result[n]=dict(pin_bits=len(ll),sink_cells=len({x['cell'] for x in ll}),
            max_manhattan=max(x['distance'] for x in ll),sum_manhattan=sum(x['distance'] for x in ll),
            bbox=[min(p[d] for p in pts) for d in (0,1)]+[max(p[d] for p in pts) for d in (0,1)],
            categories=dict(collections.Counter(x['category'] for x in ll)))
    return result,{str(p):sha(p) for p in (net,place)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base.resolve()
    target=base/'local_driver_geometry.json';assert not target.exists()
    old=ROOT/'tmp/frame-execution/native-split-timing-place1'
    rows_path=ROOT/'tmp/frame-execution/native-local-drivers-map2/replications.json'
    rows=json.loads(rows_path.read_text());originals=[r['driver'] for r in rows]
    names=originals+[x['name'] for r in rows for x in r['clones']]
    before,oldhash=inspect(old,originals);after,newhash=inspect(base,names)
    groups=[]
    for r in rows:
        names=[r['driver']]+[x['name'] for x in r['clones']];values=[after[n] for n in names]
        assert not any(x['categories'].get('clock',0) or x['categories'].get('reset',0) for x in values)
        groups.append(dict(net=r['net'],driver=r['driver'],before=before[r['driver']],
            after=values,max_group_pin_bits=max(x['pin_bits'] for x in values),
            new_total_pin_bits=sum(x['pin_bits'] for x in values),
            sum_manhattan_ratio=sum(x['sum_manhattan'] for x in values)/before[r['driver']]['sum_manhattan']))
    result=dict(status='ACTUAL_LOCAL_DRIVER_GEOMETRY_REVIEW_NOT_ROUTE_PASS',groups=groups,
        median_distance_ratio=sorted(x['sum_manhattan_ratio'] for x in groups)[len(groups)//2],
        improved_groups=sum(x['sum_manhattan_ratio']<1 for x in groups),groups_count=len(groups),
        evidence={**oldhash,**newhash,str(rows_path):sha(rows_path),str(Path(__file__)):sha(__file__)},
        scope='Public load pin bits expanded, including repeated vector bits. Not physical routing-track usage or timing. Old partition limit24 counted sink cell/port groups, not expanded pin bits.',
        hardware_access=False,board_release=False)
    target.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='groups'}))

if __name__=='__main__':main()
