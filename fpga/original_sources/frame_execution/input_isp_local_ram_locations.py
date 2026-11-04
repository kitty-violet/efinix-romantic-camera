"""Anchor only the three new ISP RAMs beside the existing video-input area.

The chosen sites are real RAM sites occupied by unanchored input FIFOs in the
previous legal placement. Those FIFOs and all ordinary logic remain movable.
This is a placement experiment, not a promise of routing or timing closure.
"""
import json, pickle, shutil
from pathlib import Path
from run_units import ROOT, HERE, EF, sha
import fixed_cnn_ram_locations as original

REFERENCE = ROOT/'tmp/frame-execution/video-service-place49-input-c1'
BOARD = ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'
SITES = {'service_isp/red': (132,242,0),
         'service_isp/green': (147,242,0),
         'service_isp/blue': (156,242,0)}


def locations(path):
    return {a[0]: tuple(map(int,a[1:])) for s in path.read_text().splitlines()
            if len(a:=s.split())==4 and not s.startswith('#')}


def generate(root, reference):
    proof = original.generate(root, reference)
    root = Path(root)
    mapped = root/'outflow/ti60f225_oob.map.v'
    assert sha(mapped)==sha(BOARD/'outflow/ti60f225_oob.map.v')
    key,g=pickle.load(open(BOARD/'service-validation-graph.pkl','rb'))
    assert key[0]==sha(mapped)
    old=locations(REFERENCE/'outflow/ti60f225_oob.out.placeloc')
    assignment=root/'fixed_dsp.placeloc'
    fixed=locations(assignment)
    assert len(fixed)==401 and not set(SITES)&set(fixed)
    assert not set(SITES.values())&set(fixed.values())
    owners={p:n for n,p in old.items() if p in set(SITES.values())}
    assert len(owners)==3
    for n,p in SITES.items():
        assert g[n][0]=='EFX_RAM10'
        occupant=owners[p]
        assert occupant not in fixed and g[occupant][0] in ('EFX_RAM10','EFX_DPRAM10')
        assert occupant.startswith('raw_frontend/consumer/serializer/pair_fifo/')
    base_proof=root/'cnn_ram_placement.base.json'
    shutil.copy2(root/'cnn_ram_placement.json',base_proof)
    assignment.write_text(assignment.read_text()+
        '# Three input ISP RAMs; displaced unanchored FIFOs remain free.\n'+
        ''.join(f'{n} {x} {y} {z}\n' for n,(x,y,z) in SITES.items()))
    review=dict(status='THREE_INPUT_RAM_PLACEMENT_EXPERIMENT_NOT_ROUTE_PASS',
        candidate_map_sha256=sha(mapped),assignments={n:list(p) for n,p in SITES.items()},
        former_automatic_occupants={str(p):n for p,n in owners.items()},
        original_hard_anchors=401,total_hard_anchors=404,ordinary_anchors=0,
        risks=['Former input FIFO RAMs must relocate; compare all RAM movement and input/control nets.',
               'Upload/readback controls may become longer; they remain normally timed.',
               'No claim that local geometric improvement ensures full routing closure.'],
        evidence={str(p):sha(p) for p in [Path(__file__),HERE/'fixed_cnn_ram_locations.py',
            base_proof,REFERENCE/'outflow/ti60f225_oob.out.placeloc',
            REFERENCE/'fixed_dsp.placeloc',BOARD/'service-validation-graph.pkl',
            EF/'doc/topics/loc-primitives.html',EF/'doc/topics/loc-rules.html']},
        hardware_access=False,board_release=False)
    path=root/'input_isp_ram_placement.json'
    path.write_text(json.dumps(review,indent=2),encoding='utf8')
    proof['assignments_sha256']=sha(assignment)
    proof['additional_ram_assignment_proof']=str(path)
    proof['additional_ram_assignment_proof_sha256']=sha(path)
    proof['original_assignment_proof_sha256']=sha(base_proof)
    (root/'cnn_ram_placement.json').write_text(json.dumps(proof,indent=2),encoding='utf8')
    return proof


def verify_placed(root):
    result=original.verify_placed(root)
    root=Path(root)
    proof=json.loads((root/'input_isp_ram_placement.json').read_text())
    for f,h in proof['evidence'].items():assert sha(f)==h,f
    loc=locations(root/'outflow/ti60f225_oob.out.placeloc')
    for n,p in SITES.items():assert loc[n]==p,(n,loc[n],p)
    (root/'input_isp_ram_actual_locations.json').write_text(json.dumps(dict(
        status='THREE_INPUT_RAM_SITES_MATCH_NOT_TIMING_PASS',
        assignments=proof['assignments'],proof_sha256=sha(root/'input_isp_ram_placement.json'),
        placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),
        hardware_access=False,board_release=False),indent=2),encoding='utf8')
    return result
