"""Compare complete fanouts of measured slow nets with the passing parent."""
import json
from run_units import ROOT,sha
from native_local_drivers import graph

def view(root,drivers):
    net=root/'outflow/ti60f225_oob.map.v';loc=root/'outflow/ti60f225_oob.out.placeloc'
    g=graph(net.read_text());pos={}
    for line in loc.read_text().splitlines():
        words=line.split()
        if len(words)==4 and not line.startswith('#'):pos[words[0]]=tuple(map(int,words[1:]))
    result={}
    for driver in drivers:
        k,p=g[driver];assert k=='EFX_FF';q=p['Q'][0]
        loads=[dict(cell=n,pin=port,position=pos.get(n)) for n,(kind,pins) in g.items()
               for port,values in pins.items() if q in values and not (n==driver and port=='Q')]
        assert loads and all(x['position'] for x in loads)
        points=[pos[driver]]+[x['position'] for x in loads]
        bbox=[[min(p[i] for p in points),max(p[i] for p in points)] for i in (0,1)]
        result[driver]=dict(position=pos[driver],loads=loads,fanout=len(loads),bbox=bbox,
            hpwl=sum(hi-lo for lo,hi in bbox),max_manhattan=max(sum(abs(p[i]-pos[driver][i]) for i in (0,1)) for p in points),
            signal=q,clock=p['CLK'],reset=p['SR'],enable=p['CE'])
    return dict(nets=result,inputs={str(net):sha(net),str(loc):sha(loc)})

if __name__=='__main__':
    base=ROOT/'tmp/frame-execution'
    out=base/'camera-checked-route2/hotnet_geometry_comparison.json';assert not out.exists()
    drivers=['LOCAL_DRIVER_094','video_processing_pipeline/pipeline/cnn/core/dut/engine/router/held[5]~FF']
    current=view(base/'camera-checked-bound2',drivers);parent=view(base/'native-local-driver-bound1',drivers)
    for n in drivers:
        assert {(r['cell'],r['pin']) for r in current['nets'][n]['loads']}=={(r['cell'],r['pin']) for r in parent['nets'][n]['loads']}
    result=dict(status='SAME_FANOUT_PLACEMENT_COMPARISON_NOT_TIMING_PROOF',current=current,parent=parent,
        script_sha256=sha(__file__),hardware_access=False)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({name:{n:{k:v for k,v in row.items() if k in ('position','fanout','bbox','hpwl','max_manhattan')} for n,row in data['nets'].items()} for name,data in (('current',current),('parent',parent))}))
