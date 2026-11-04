"""Read-only primitive/topology comparison; never equate primitive sum with XLR."""
import collections,json
from pathlib import Path
from run_units import ROOT,sha
from audit_snapshot_cells import cells
from fixed_control_ddr_reuse_locations import locations
from review_existing_routability import metrics

def group(name):
    if name.startswith('cpu/system_hardJtag_debug_logic_jtagBridge/observation/'):return 'independent_snapshot'
    if name.startswith(('cell_probe/','cell_diagnostic[')):return 'finite_primitive_probe'
    if name.startswith(('clock_probe/','ddr_clock_diagnostic[')):return 'ddr_clock_probe'
    if name.startswith('cpu/first_fault/'):return 'controller_first_fault'
    if 'startup_cause' in name:return 'startup_cause'
    if '/engine/rq/' in name or '/engine/od[' in name or '/engine/ox[' in name:return 'rq'
    if '/engine/mac/source/' in name or '/engine/mac/column' in name:return 'column_source'
    if name.startswith('cpu/'):return 'other_cpu'
    return 'other_or_flattened_names'

def main():
    root=ROOT/'tmp/board-test/registered-pressure1';assert not root.exists();root.mkdir()
    result={}
    for tag in ('split-outputs-place3','split-registered-place1'):
        folder=ROOT/'tmp/frame-execution'/tag;net=folder/'outflow/ti60f225_oob.map.v'
        cc=cells(net.read_text());xy=locations(folder/'outflow/ti60f225_oob.out.placeloc')
        count=collections.defaultdict(collections.Counter)
        for n,(kind,p,a) in cc.items():count[group(n)][kind]+=1
        m=metrics(folder);hotspots={}
        for scale,peak in m['peaks'].items():
            x,y=peak['xy'];near=collections.defaultdict(collections.Counter)
            # This is a fixed-radius local census around the reported peak,
            # not the tool's undisclosed congestion smoothing kernel.
            for n,(kind,p,a) in cc.items():
                if n in xy and abs(xy[n][0]-x)<=8 and abs(xy[n][1]-y)<=8:near[group(n)][kind]+=1
            hotspots[scale]=dict(peak=peak,within8_sites_each_axis=dict(near))
        result[tag]=dict(named_primitive_groups=dict(count),metrics=m,hotspots=hotspots,
            map_sha256=sha(net),placement_sha256=sha(folder/'outflow/ti60f225_oob.out.placeloc'))
    out=dict(status='READ_ONLY_NAMED_PRIMITIVE_PRESSURE_CENSUS',cases=result,
        limits=['Flattened anonymous names cannot be reliably assigned to RTL functions.',
                'LUT/FF sharing means counts cannot be summed into exact physical XLR savings.',
                'Congestion peak and nearby cells do not identify all unresolved routing nets.'],
        hardware_access=False,design_changed=False,script_sha256=sha(__file__))
    (root/'result.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    for tag,v in result.items():
        print(tag,json.dumps({k:c for k,c in v['named_primitive_groups'].items() if k!='other_or_flattened_names'}))

if __name__=='__main__':main()
