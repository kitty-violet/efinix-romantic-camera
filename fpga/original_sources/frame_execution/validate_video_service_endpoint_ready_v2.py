"""Expanded ready cut with exact retained-driver binding after LUT repacking."""
from pathlib import Path
from bind_video_service_mapped_cut import bind_cut
source=Path(__file__).with_name('validate_video_service_fault_ready.py');code=source.read_text()
for a,b in [('fault-ready-independent-audit.json','endpoint-ready-independent-audit.json'),('FAULT_READY_SOURCE_INDEPENDENT_PASS','ENDPOINT_EXPANDED_READY_BDD_PASS'),('video-service-native50-weight-zero-factored','video-service-native56-global-targeted'),('len(boundary)==20','len(boundary)<=32'),('reg[19:0] stim=0;wire before_q,after_q;integer i;','reg[31:0] stim=0;reg[31:0] rng=1094032841;wire before_q,after_q;integer i;'),('ACTUAL_FAULT_READY_EXHAUSTIVE_PASS','ACTUAL_ENDPOINT_EXPANDED_READY_PASS'),('1048576','393282'),('known_controls=20','known_controls=expanded'),('known_control_inputs=20','known_control_inputs=len(boundary)'),('key,_,new,_,np=','key,source_graph,new,_,np=')]:
 assert a in code,a;code=code.replace(a,b)
code=code.replace('stim=i;#2;', 'rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};stim=(i<262144)?i:rng;#2;')
code=code.replace("scope='Actual old versus new EFX primitive handshake cut; all Boolean control combinations; no whole-board or arbitrary-X claim'", "scope='Expanded ready cut: complete independent BDD; actual vendor262144low-bit states+131138LFSR vectors; no arbitrary-X claim'")
needle="  assert external<=set(boundary),(side,external-set(boundary))\n  for t in external:lines.append(f'assign {value(t)}=stim[{boundary.index(t)}];')"
assert code.count(needle)==1
code=code.replace(needle,"""  bindings={t:i for i,t in enumerate(boundary)}
  if side and board:bindings,binding_rows=bind_cut(source_graph,new,bindings,external)
  assert external<=set(bindings),(side,external-set(bindings))
  for t in external:lines.append(f'assign {value(t)}=stim[{bindings[t]}];')""")
code=code.replace('paths=[Path(__file__).resolve(),',"paths=[Path(__file__).resolve(),source,Path(__file__).with_name('bind_video_service_mapped_cut.py'),")
exec(compile(code,str(source)+':endpoint-driver-bound','exec'),globals())
