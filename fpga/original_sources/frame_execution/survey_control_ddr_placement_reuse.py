"""Read-only inventory for a future local placement repair, not a release gate."""
import json,re,collections
from pathlib import Path
from run_units import ROOT,sha
from audit_snapshot_cells import cells
from audit_control_state_implementation import norm

old=ROOT/'tmp/frame-execution/command-progress-seed3-1/pnr-seed1'
new=ROOT/'tmp/frame-execution/connected-board-control-ddr-combined1'
out=ROOT/'tmp/frame-execution/control-ddr-combined-placement-reuse.json'
assert not out.exists()
def read(root):
    text=(root/'outflow/ti60f225_oob.map.v').read_text()
    cc=cells(text);params={}
    for n,p,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text,re.M):
        params.setdefault(norm(n),{})[p]=norm(v)
    return {n:(kind,ports,params.get(n,{})) for n,(kind,ports,attrs) in cc.items()}
a=read(old);b=read(new)
placed={line.split()[0] for line in (old/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines() if line and not line.startswith('#')}
same={n for n in a.keys()&b.keys() if a[n]==b[n] and n in placed}
typed=collections.Counter(b[n][0] for n in same)
groups=collections.Counter(n.split('/')[0] for n in same if '/' in n)
data=dict(status='PLACEMENT_REUSE_INVENTORY_ONLY_NOT_IMPLEMENTED',old=str(old),new=str(new),
    old_cells=len(a),new_cells=len(b),same_named_type_ports_parameters_with_old_location=len(same),
    cell_types=dict(typed),largest_hierarchies=groups.most_common(15),
    evidence={str(p):sha(p) for p in (old/'outflow/ti60f225_oob.map.v',new/'outflow/ti60f225_oob.map.v',old/'outflow/ti60f225_oob.out.placeloc',Path(__file__))},
    limitation='Matching names, primitive types, ports and parameters are candidates only; upstream aliases and new net connectivity require further checks. No placement constraints were changed and no routing or timing success is inferred.',hardware_access=False)
out.write_text(json.dumps(data,indent=2));print(json.dumps({k:v for k,v in data.items() if k!='evidence'},indent=2))
