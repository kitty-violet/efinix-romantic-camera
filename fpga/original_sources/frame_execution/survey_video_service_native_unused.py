"""Read-only observability audit of existing public native primitives.

All behavioral RTL, protected blocks, top ports, assignments and constraints
are conservative roots. Unknown/side-effecting primitives are always live.
This reports removal candidates only, not area savings or a release.
"""
import argparse,collections,json,re,time
from pathlib import Path
from run_units import ROOT,sha
from audit_continuous_vectors import expand_graph
from native_local_drivers import CELL,REF
from camera_joint_native import OUT
from fixed_control_ddr_reuse_locations import locations

def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--physical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    src=a.native.resolve()/'native_source.v';physical=a.physical.resolve();s=src.read_text();start=time.monotonic()
    g=expand_graph(s);outer=CELL.sub('',s)
    outer=re.sub(r'^\s*defparam\s+[^;]*;','',outer,flags=re.M)
    outer=re.sub(r'^\s*wire\s+[^;]*;','',outer,flags=re.M)
    roots={m['id'].lstrip('\\') for m in REF.finditer(outer) if m['id']}
    sdc=(physical/'ti60f225_oob.sdc').read_text();anchors=locations(physical/'fixed_dsp.placeloc')
    pure={'EFX_LUT4','EFX_FF','EFX_ADD','EFX_SRL8'}
    outputs={n:{t for pin,ts in pins.items() if pin in OUT[k] for t in ts if t[0]=='net'} for n,(k,pins) in g.items()}
    drivers=collections.defaultdict(set)
    for n,ts in outputs.items():
        for t in ts:drivers[t].add(n)
    live={n for n,(k,pins) in g.items() if k not in pure or n.startswith('protected_inst/') or n in anchors or
          n in sdc or any(t[1] in roots or t[1] in sdc for t in outputs[n])}
    todo=list(live)
    while todo:
        n=todo.pop();k,pins=g[n]
        for pin,ts in pins.items():
            if pin in OUT[k]:continue
            for t in ts:
                for d in drivers[t]-live:live.add(d);todo.append(d)
    dead=set(g)-live
    assert not any(n in anchors or n.startswith('protected_inst/') or g[n][0] not in pure for n in dead)
    r=dict(status='NATIVE_PUBLIC_UNUSED_CONES_CANDIDATES_ONLY',source=str(src),source_sha256=sha(src),
           checker_sha256=sha(__file__),constraints_sha256=sha(physical/'ti60f225_oob.sdc'),
           anchors_sha256=sha(physical/'fixed_dsp.placeloc'),seconds=time.monotonic()-start,
           native_primitives=len(g),candidate_count=len(dead),kinds=dict(collections.Counter(g[n][0] for n in dead)),
           cells=sorted(dead),roots='All exterior symbols, top/behavioral/protected/assignments, constraints, hard anchors; all input pins traced including CLK/CE/SR',
           hardware_access=False,source_edited=False,board_release=False)
    out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='cells'},indent=2))
if __name__=='__main__':main()
