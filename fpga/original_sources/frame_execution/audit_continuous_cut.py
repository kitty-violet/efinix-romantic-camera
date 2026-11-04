"""Expand vector connections and prove every changed cone output has a source."""
import collections,json,re
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import PARENT,DEST
from native_local_drivers import graph
from camera_joint_native import OUT

def expand_graph(text):
    ranges={n.lstrip('\\'):(int(a),int(b)) for a,b,n in re.findall(r'^\s*wire\s+\[(\d+):(\d+)\]\s+(\S+)\s*;',text,re.M)}
    g=graph(text)
    for kind,pins in g.values():
        for p,ts in pins.items():
            values=[]
            for t in ts:
                if t[0]=='net' and t[2] is None and t[1] in ranges:
                    a,b=ranges[t[1]];step=-1 if a>b else 1
                    values.extend(('net',t[1],i) for i in range(a,b+step,step))
                else:values.append(t)
            pins[p]=values
    return g

def main():
    original=PARENT/'outflow/ti60f225_oob.map.v';g=expand_graph(original.read_text())
    b=json.loads((DEST/'boundary.json').read_text());removed=set(b['removed'])
    exports={tuple(x['token']) for x in b['rows']}|{('net','cpu/boot_valid',None),('net','video_processing_pipeline/compute_start',None)}|{('net','cnn_admitted',i) for i in range(5)}
    consumers=collections.defaultdict(set)
    for n,(k,pins) in g.items():
        if n not in removed:
            for p,ts in pins.items():
                if p not in OUT[k]:
                    for t in ts:consumers[t].add((n,p))
    missing=[];crossings=0
    for n in removed:
        k,pins=g[n]
        for p,ts in pins.items():
            if p in OUT[k]:
                for t in ts:
                    if consumers[t]:
                        crossings+=len(consumers[t])
                        if t not in exports:missing.append(dict(cell=n,token=t,consumers=sorted(consumers[t])))
    r=dict(status='EXPANDED_NATIVE_CUT_PASS' if not missing else 'MISSING_EXPORTED_DRIVER',missing=missing,
        expanded_bus_connections=True,crossing_loads=crossings,removed_cells=len(removed),
        inputs={str(p):sha(p) for p in [original,DEST/'boundary.json',Path(__file__)]},hardware_access=False)
    out=DEST/'expanded_cut_audit.json';assert not out.exists();out.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
    assert not missing
if __name__=='__main__':main()
