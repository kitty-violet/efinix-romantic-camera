"""Freeze trial verdicts without promoting either rejected source to production."""
import json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_control_payload_refactor import gate
from audit_unused_reader import verify_unused_reader_binding
from prepare_held_reader import transform

out=ROOT/'tmp/frame-execution/held-reader-trial-audit1'
assert not out.exists();out.mkdir()
board=ROOT/'tmp/frame-execution/connected-board-unused-peripherals1'
before,selected=verify_unused_reader_binding(json.loads((board/'gate_binding.json').read_text()))
profile=HERE/'board_resource_candidate.json';profile_data=json.loads(profile.read_text())
assert profile_data['selected_board']=='tmp/frame-execution/connected-board-unused-peripherals1'
assert len(profile_data['overrides'])==13 and len(profile_data['gates'])==17
assert not any('held_reader' in x['path'] or 'held_even_reader' in x['path'] for x in profile_data['overrides'])
ref=board/'assembly/unified_column_reader.sv'
for variant,even in [('held_reader',False),('held_even_reader',True)]:
 src=HERE/'variants'/variant/ref.name
 assert src.read_text().rstrip()==transform(ref.read_text(),even).rstrip()

roots={
 'all_packet':'held-reader-gate1',
 'even_packet':'held-even-reader-gate1',
 'even_cluster':'held-even-reader-cluster1',
 'even_guard':'held-even-reader-guard1',
 'even_fourstate_guard':'held-even-reader-guard-fourstate1',
 'even_arena':'held-even-reader-arena2',
}
records=[]
for role,name in roots.items():
 root=ROOT/'tmp/frame-execution'/name
 names,record=gate(root,role=='even_cluster')
 variant='held_reader' if role=='all_packet' else 'held_even_reader'
 candidate=HERE/'variants'/variant/ref.name
 assert names[ref.name]==sha(candidate)
 common=(names.keys() & selected['names'].keys())-{ref.name}
 assert all(names[n]==selected['names'][n] for n in common),(role,common)
 records.append(dict(role=role,**record))
blocks=lambda name:re.findall(r'^BLOCK .*$',(ROOT/'tmp/frame-execution'/name/'simulation.log').read_text(),re.M)
old,new=blocks('shared-word-cluster1'),blocks('held-even-reader-cluster1')
assert len(old)==len(new)==24 and old==new
maps={}
for role,name in [('reference','reader-options-map1/reference'),('all','held-reader-map1'),('even','held-even-reader-map1')]:
 root=ROOT/'tmp/frame-execution'/name;result=json.loads((root/'result.json').read_text())
 assert sha(root/'outflow/unified_column_reader.map.rpt')==result['report_sha256']
 assert sha(root/'unified_column_reader.sv')==result['source_sha256']
 assert result['source_sha256']==(sha(ref) if role=='reference' else sha(HERE/'variants'/('held_reader' if role=='all' else 'held_even_reader')/ref.name))
 maps[role]=result
failed=ROOT/'tmp/frame-execution/held-reader-cluster1/simulation.log'
failed_text=failed.read_text()
assert 'read_idx=25/167' in failed_text and 'pending=55ff55' in failed_text
assert 'ov=4 ready=0' in failed_text and 'command watchdog' in failed_text
arena=json.loads((ROOT/'tmp/frame-execution/held-even-reader-arena2/result.json').read_text())
w0,w1,r0,r1,c0,c1=map(int,re.search(r'writes=(\d+)/(\d+) reads=(\d+)/(\d+) contending=(\d+)/(\d+)',arena['background'][0]).groups())
old_transactions=w0+r0;new_transactions=w1+r1
result=dict(status='HELD_READER_TRIAL_AUDITED_BOTH_REJECTED',selected_board=str(board),
 unchanged_profile_sha256=sha(profile),selected_resources=selected['result']['resources'],
 reference_source_sha256=sha(ref),gates=records,isolated_maps=maps,
 all_ports_rejected_reason='writer uses odd ports; retained read locks block result drain and dispatcher consumption, forming a wait cycle',
 all_ports_failure_log_sha256=sha(failed),equal_local_block_cycle_rows=24,
 even_ports_rejected_reason='isolated LUT +9 despite FF -768; contending second client loses accepted transactions; primary capacity bottleneck is LUT',
 same_interval_background=dict(reference=old_transactions,candidate=new_transactions,
  decrease_percent=100*(old_transactions-new_transactions)/old_transactions,
  meaning='Accepted background transactions in this saturated arena fixture, not measured DDR bandwidth or frame FPS'),
 new_whole_board_map=False,full_frame=False,physical_timing_pass=False,board_download_allowed=False)
(out/'frozen_audit.py').write_bytes(Path(__file__).read_bytes())
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('gates','isolated_maps')}))
