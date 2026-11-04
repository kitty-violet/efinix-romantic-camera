"""Keep the successful CNN placement; move a public Gray source and its LUT.
Protected synchronizer heads remain under the tool's control.
"""
import json,re
from pathlib import Path
from run_units import ROOT,EF,sha
from audit_canonical_multiseed import audit
from fixed_shift_guard_local_repair import locations,plan as cnn_plan
PREFIX='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/controller_main/bram_u2/u_efx_fifo_top/xefx_fifo_ctl/async_clk.'
SOURCE=PREFIX+'waddr_cntr_gry_r[1]~FF'
HEAD=PREFIX+'wr2rd_addr_sync/async_reg[0][1]~FF'
DRIVER='LUT__157777'
REF=ROOT/'tmp/frame-execution/shift-guard-local-physical1/canonical-seed1'

def plan(root):
 root=Path(root);assigned,proof=cnn_plan(root);prior=audit(REF)
 assert prior['wns_ns']==-0.032 and prior['whs_ns']>=0 and set(prior['checks'].values())=={0}
 physical=Path(prior['physical']);placed=locations(physical/'outflow/ti60f225_oob.out.placeloc')
 assert all(placed[n]==p for n,p in assigned.items())
 assert placed[HEAD]==(32,2,2) and HEAD not in assigned
 assert assigned[SOURCE]==(64,2,2) and assigned[DRIVER]==(64,2,0)
 assert not any(p[:2]==(31,2) for p in placed.values())
 net=(root/'outflow/ti60f225_oob.map.v').read_text()
 cells={n.lstrip('\\'):(k,body) for k,n,body in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\) /\*',net,re.M|re.S)}
 assert cells[SOURCE][0]=='EFX_FF' and cells[DRIVER][0]=='EFX_LUT4'
 def port(name,key):
  match=re.search(r'\.'+key+r'\(([^()]*)\)',cells[name][1]);assert match
  return re.sub(r'\s+','',match[1]).replace('\\','')
 assert port(DRIVER,'O')==port(SOURCE,'D') and port(SOURCE,'CLK')=='ddr_core_clk200'
 # Destination legality is established by actual prior LUT/FF packing there.
 oldphys=Path(audit(Path(proof['reference']))['physical'])
 oldloc=locations(oldphys/'outflow/ti60f225_oob.out.placeloc')
 assert oldloc['LUT__157464']==(31,2,0)
 assert any(n.endswith('~FF') and p==(31,2,2) for n,p in oldloc.items())
 assigned[DRIVER]=(31,2,0);assigned[SOURCE]=(31,2,2)
 assert len(set(assigned.values()))==len(assigned)==56835
 proof.update(ddr_source_move=dict(reference=str(REF),reference_result_sha256=sha(REF/'result.json'),placement=str(physical/'outflow/ti60f225_oob.out.placeloc'),placement_sha256=sha(physical/'outflow/ti60f225_oob.out.placeloc'),source=SOURCE,driver=DRIVER,head=HEAD,head_before=[32,2,2],source_before=[64,2,2],source_after=[31,2,2],driver_before=[64,2,0],driver_after=[31,2,0],protected_head_assigned=False))
 return assigned,proof

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
 return dict(status='SAME_MAP_ALL_PRIMITIVE_POSITIONS_VERIFIED',primitive_count=len(expected),moved_cells=3,
  proof_sha256=sha(root/'local_repair_placement.json'),actual_placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),board_release=False)


