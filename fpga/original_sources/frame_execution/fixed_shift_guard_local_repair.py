"""Retain same-netlist primitive positions and shorten one measured RAM path.

All coordinates come from legal tool exports. The vacant destination was used
by a LUT4 in a previously passing implementation, not inferred from a grid.
"""
import json,re
from pathlib import Path
from run_units import ROOT,EF,sha
from audit_canonical_multiseed import audit

BEST=ROOT/'tmp/frame-execution/shift-guard-feedback-physical1'
OLD=ROOT/'tmp/frame-execution/first-fault-reference-physical1/canonical-seed1'
TARGET='LUT__133530'
DEST=(56,167,0)

def locations(path):
 return {w[0]:tuple(map(int,w[1:])) for line in Path(path).read_text().splitlines()
         if len(w:=line.split())==4 and not line.startswith('#')}

def plan(root):
 root=Path(root);previous=audit(BEST/'canonical-seed1');assert previous['wns_ns']==-0.043 and previous['whs_ns']>=0
 base=Path(previous['physical']);net=base/'outflow/ti60f225_oob.map.v';current=root/'outflow/ti60f225_oob.map.v'
 assert sha(net)==sha(current)==previous['identity']['map']
 text=net.read_text();cells={n.lstrip('\\'):(k,body) for k,n,body in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\) /\*',text,re.M|re.S)}
 placed=locations(base/'outflow/ti60f225_oob.out.placeloc')
 assert len(cells)==56835 and set(cells)<=set(placed)
 assert cells[TARGET][0]=='EFX_LUT4' and placed[TARGET]==(61,171,0)
 for signal in ('core/sd [730]','core/sd [666]','mac/source/half_q','n63664'):assert signal in cells[TARGET][1]
 assert not any(p[:2]==DEST[:2] for p in placed.values()),'Destination occupied'
 older=audit(OLD);assert older['current_timing_pass'];op=Path(older['physical'])
 olderloc=locations(op/'outflow/ti60f225_oob.out.placeloc')
 assert olderloc['LUT__135001']==DEST
 assert re.search(r'^\s*EFX_LUT4 LUT__135001\s*\(', (op/'outflow/ti60f225_oob.map.v').read_text(),re.M)
 assigned={n:placed[n] for n in cells};assert len(set(assigned.values()))==len(assigned)
 assigned[TARGET]=DEST
 return assigned,dict(reference=str(BEST/'canonical-seed1'),reference_result_sha256=sha(BEST/'canonical-seed1/result.json'),
  reference_map_sha256=sha(net),reference_placement_sha256=sha(base/'outflow/ti60f225_oob.out.placeloc'),
  destination_evidence=str(op/'outflow/ti60f225_oob.out.placeloc'),destination_evidence_sha256=sha(op/'outflow/ti60f225_oob.out.placeloc'),
  moved_cell=TARGET,before=[61,171,0],after=list(DEST),primitive_count=len(assigned),rtl_or_map_changed=False)

def generate(root):
 root=Path(root);assigned,proof=plan(root);path=root/'fixed_dsp.placeloc'
 existing=locations(path)
 assert all(assigned.get(n)==v for n,v in existing.items()),'Would alter prior DSP/RAM/feedback constraints'
 before=sha(path)
 path.write_text('# Same mapped cells, preserve legal locations; single measured CNN LUT move.\n'+''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in sorted(assigned.items())))
 for name in ('cnn_ram_placement.json','cnn_feedback_placement.json'):
  f=root/name;m=json.loads(f.read_text());assert m['assignments_sha256']==before
  m['before_local_repair_assignments_sha256']=before;m['assignments_sha256']=sha(path);f.write_text(json.dumps(m,indent=2))
 proof.update(status='SAME_MAP_LOCAL_PLACEMENT_CANDIDATE_NOT_PASS',generator_sha256=sha(__file__),base_assignments_sha256=before,assignments_sha256=sha(path),assignments=assigned,
  docs={str(EF/'doc/topics'/n):sha(EF/'doc/topics'/n) for n in ('loc-primitives.html','loc-rules.html')})
 (root/'local_repair_placement.json').write_text(json.dumps(proof,indent=2));(root/'local_repair_generator.py').write_bytes(Path(__file__).read_bytes())
 return proof

def verify_placed(root):
 root=Path(root);r=json.loads((root/'local_repair_placement.json').read_text());expected,evidence=plan(root)
 for k,v in evidence.items():assert r[k]==v,k
 assert r['generator_sha256']==sha(__file__)==sha(root/'local_repair_generator.py')
 assert r['assignments_sha256']==sha(root/'fixed_dsp.placeloc')
 assert {n:tuple(v) for n,v in r['assignments'].items()}==expected==locations(root/'fixed_dsp.placeloc')
 actual=locations(root/'outflow/ti60f225_oob.out.placeloc')
 for n,v in expected.items():assert actual[n]==v,(n,actual.get(n),v)
 return dict(status='SAME_MAP_ALL_PRIMITIVE_POSITIONS_VERIFIED',primitive_count=len(expected),moved_cells=1,
  proof_sha256=sha(root/'local_repair_placement.json'),actual_placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)
