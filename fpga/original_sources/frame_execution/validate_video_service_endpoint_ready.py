"""Actual32-input expanded ready cut; full Boolean BDD plus bounded vendor test."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_fault_ready_v2.py');code=source.read_text()
needle="exec(compile(code,str(source)+':mapped-boundary-binding','exec'),globals())"
assert code.count(needle)==1
adjust="""
for a,b in [('fault-ready-independent-audit.json','endpoint-ready-independent-audit.json'),('FAULT_READY_SOURCE_INDEPENDENT_PASS','ENDPOINT_EXPANDED_READY_BDD_PASS'),('video-service-native50-weight-zero-factored','video-service-native56-global-targeted'),('len(boundary)==20','len(boundary)<=32'),('reg[19:0] stim=0;wire before_q,after_q;integer i;','reg[31:0] stim=0;reg[31:0] rng=1094032841;wire before_q,after_q;integer i;'),('ACTUAL_FAULT_READY_EXHAUSTIVE_PASS','ACTUAL_ENDPOINT_EXPANDED_READY_PASS'),('1048576','393282'),('known_controls=20','known_controls=expanded'),('known_control_inputs=20','known_control_inputs=len(boundary)')]:
 assert a in code,a;code=code.replace(a,b)
code=code.replace('stim=i;#2;', 'rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};stim=(i<262144)?i:rng;#2;')
code=code.replace('scope=\\'Actual old versus new EFX primitive handshake cut; all Boolean control combinations; no whole-board or arbitrary-X claim\\'', 'scope=\\'Expanded ready cut: complete canonical BDD in independent audit; this actual vendor test uses262144 directed low-bit states plus131138LFSR vectors; no arbitrary-X claim\\'')
"""
code=code.replace(needle,adjust+'\n'+needle)
exec(compile(code,str(source)+':expanded-ready','exec'),globals())
