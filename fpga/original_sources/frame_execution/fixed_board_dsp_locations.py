"""Place exact fixed video DSP chains at verified Ti60 sites; no clock exceptions."""
import collections,re
from run_units import ROOT,sha

def generate(root, rq_only=False):
    reference=ROOT/'tmp/efinity-shared-rq-dsp-input-absorb-full-seed1/outflow'
    net=reference/'pw_shared_rq_physical.map.v'
    loc=reference/'pw_shared_rq_physical.out.placeloc'
    names=set(re.findall(r'EFX_DSP(?:24|48)\s+\\([^\s]+)',net.read_text()))
    sites=collections.defaultdict(set)
    for line in loc.read_text().splitlines():
        z=line.split()
        if len(z)==4 and z[0] in names: sites[int(z[1])].add(int(z[2]))
    assert len(sites)==10 and all(len(y)==16 for y in sites.values())
    halves=[]
    for x,ys in sorted(sites.items()):
        ys=sorted(ys);assert all(b-a==20 for a,b in zip(ys,ys[1:]))
        halves.extend([[(x,y) for y in ys[:8]],[(x,y) for y in ys[8:]]])
    text=(root/'outflow/ti60f225_oob.map.v').read_text()
    actual={name:kind for kind,name in re.findall(r'(EFX_DSP(?:24|48))\s+\\([^\s]+)',text)}
    prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
    locations={};occupancy=collections.defaultdict(list)
    def put(name,kind,xy,sub):
        assert actual.get(name)==kind,(name,actual.get(name))
        assert name not in locations
        locations[name]=(*xy,sub);occupancy[xy].append((kind,sub))
    # Each four-row compute lane has 32 physical cascade DSPs and its eight
    # RQ DSPs in five adjacent half-columns. Avoid putting every RQ at one edge.
    for row in range(4):
        for pair in range(0 if rq_only else 4):
            for tap,xy in enumerate(halves[row*5+pair]):
                for sub in (0,1):
                    put(prefix+f'mac/arithmetic/ROW[{row}].CHANNEL[{pair*2+sub}].TAP[{tap}].mac','EFX_DSP24',xy,sub)
        for channel,xy in enumerate(halves[row*5+4]):
            put(prefix+f'rq/LANE[{row*8+channel}].rq/multiply_add','EFX_DSP48',xy,0)
    expected={n for n,k in actual.items() if not rq_only or k=='EFX_DSP48'}
    assert set(locations)==expected and len(occupancy)==(32 if rq_only else 160)
    if rq_only:
        # Four complete half-columns are occupied by RQ. The remaining
        # sixteen half-columns each fit one uninterrupted eight-DSP chain.
        free_halves=[h for h in halves if not any(xy in occupancy for xy in h)]
        assert len(free_halves)==16 and sum(map(len,free_halves))==128
    assert all(sorted(v)==[('EFX_DSP24',0),('EFX_DSP24',1)] or v==[('EFX_DSP48',0)] for v in occupancy.values())
    output=root/'fixed_dsp.placeloc';assert not output.exists()
    output.write_text(''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in locations.items()))
    settings=root/'efx_pnr_settings.ini';assert not settings.exists()
    settings.write_text('loc_assignment = fixed_dsp.placeloc\ndump_placeloc = on\n')
    return dict(layout='rq_reserved_mac_automatic' if rq_only else 'four_local_rows',
        dsp24=0 if rq_only else 256,dsp48=32,physical_sites=32 if rq_only else 160,
        whole_netlist_dsp24=256,whole_netlist_dsp48=32,
        reference_netlist=str(net),reference_netlist_sha256=sha(net),
        reference_placement=str(loc),reference_placement_sha256=sha(loc),
        assignments_sha256=sha(output),settings_sha256=sha(settings),generator_sha256=sha(__file__))
