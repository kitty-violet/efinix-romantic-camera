"""Read-only placement census for the current snapshot candidate and parent.

This does not identify the router's unresolved nets. It distinguishes named
cell placement, pin fanout and bounding boxes from actual routing evidence.
"""
import collections,json,re
from run_units import ROOT,sha
from audit_snapshot_cells import cells
from fixed_control_ddr_reuse_locations import locations
from review_existing_routability import metrics,CLOCKS,OUTPUTS
from analyze_registered_pressure import group


def analyze(tag):
    folder=ROOT/'tmp/frame-execution'/tag
    net=folder/'outflow/ti60f225_oob.map.v'
    placement=folder/'outflow/ti60f225_oob.out.placeloc'
    cc=cells(net.read_text());xy=locations(placement)
    assert len(cc)>50000 and len(xy)>40000
    counts=collections.defaultdict(collections.Counter)
    points=collections.defaultdict(list)
    loads=collections.defaultdict(set);drivers={}
    for name,(kind,ports,attrs) in cc.items():
        category=group(name);counts[category][kind]+=1
        if name in xy:points[category].append(xy[name])
        for pin,value in ports.items():
            if pin in OUTPUTS.get(kind,()):drivers[value]=name
            elif pin not in CLOCKS and re.fullmatch(r"[A-Za-z_][^{}',\s]*",value):
                loads[value].add(name)
    def bbox(names):
        pts=[xy[n] for n in names if n in xy]
        if not pts:return None
        return [min(p[0] for p in pts),min(p[1] for p in pts),
                max(p[0] for p in pts),max(p[1] for p in pts)]
    m=metrics(folder);hotspots={}
    for scale,peak in m['peaks'].items():
        x,y=peak['xy'];near=collections.defaultdict(collections.Counter)
        for n,(kind,ports,attrs) in cc.items():
            if n in xy and abs(xy[n][0]-x)<=8 and abs(xy[n][1]-y)<=8:
                near[group(n)][kind]+=1
        hotspots[scale]=dict(peak=peak,within8_sites_each_axis=dict(near))
    high=[]
    for signal,sinks in loads.items():
        source=drivers.get(signal);sinks=sinks-{source}
        if len(sinks)<128:continue
        high.append(dict(signal=signal,driver=source,sinks=len(sinks),
                         bbox=bbox(sinks|({source} if source else set())),
                         groups=dict(collections.Counter(group(n) for n in sinks))))
    anchors=json.loads((folder/'cnn_ram_placement.json').read_text())
    assert len(anchors['assignments'])==113
    ram_groups=collections.defaultdict(list)
    for n,p in anchors['assignments'].items():
        assert list(xy[n])==p
        ram_groups[n.rsplit('/',1)[0]].append(n)
    return dict(metrics=m,named_primitive_groups=dict(counts),hotspots=hotspots,
                high_fanout=sorted(high,key=lambda r:r['sinks'],reverse=True),
                named_group_boxes={g:[min(p[0] for p in ps),min(p[1] for p in ps),
                                     max(p[0] for p in ps),max(p[1] for p in ps)]
                                   for g,ps in points.items()},
                anchored_ram_groups={g:dict(count=len(ns),bbox=bbox(ns)) for g,ns in ram_groups.items()},
                evidence={str(p):sha(p) for p in (net,placement,folder/'cnn_ram_placement.json')})


def main():
    root=ROOT/'tmp/board-test/snapshot-physical-pressure1'
    assert not root.exists();root.mkdir()
    cases={}
    for tag in ('split-registered-place1','snapshot-ram-place1'):
        cases[tag]=analyze(tag)
        (root/(tag+'.json')).write_text(json.dumps(cases[tag],indent=2))
        print(tag,json.dumps(cases[tag]['metrics']),flush=True)
    result=dict(status='PLACED_NAMED_CELLS_AND_SCALAR_NET_PRESSURE_CENSUS',cases=cases,
                limitations=['Nearby cells are not proven unresolved routing nets.',
                             'Flattened names cannot be assigned reliably to RTL modules.',
                             'Scalar net census excludes clock pins and vector expressions.',
                             'Primitive counts are not additive physical XLR area.'],
                script_sha256=sha(__file__),hardware_access=False,design_changed=False)
    (root/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
