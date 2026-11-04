"""Recover a completed MAP after only the old DSP-count postcondition rejected it.

Preserve original logs and manifest. Recheck the exact frozen inputs/project,
official success marker, warnings and explicit architecture primitive counts.
Does not rerun or qualify board capacity/timing.
"""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp')
assert not (r/'result.json').exists()
m=json.loads((r/'inputs.json').read_text())
for row in m['inputs']:
 assert sha(row['snapshot'])==row['sha256']
 assert sha(row['source'])==row['source_sha256']
top='fixed_production_cluster_map'
assert sha(r/(top+'.xml'))==m['project_sha256']
log=r/'map.log';assert re.search(r'map\s*:\s*PASS',log.read_text())
rpt=r/'outflow'/(top+'.map.rpt');warn=r/'outflow'/(top+'.warn.log')
assert 'EFX-0654' not in warn.read_text(errors='replace')
counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
assert (counts.get('DSP24'),counts.get('DSP48'))==(288,16),counts
result=dict(status='COMBINED_FEED_MATRIX_RQ_MAP_PASS_NOT_BOARD',resources=counts,
 seconds=None,inputs_sha256=sha(r/'inputs.json'),report_sha256=sha(rpt),warning_sha256=sha(warn),
 recovery_auditor_sha256=sha(__file__),map_log_sha256=sha(log),
 recovered_postcondition='Official MAP passed; original runner asserted obsolete 256 DSP24 / 32 DSP48 instead of 288 / 16. Both total 160 physical DSP.',
 full_cnn=False,external_ram=True,production_coordinates=True,residual_and_writeback=True,production_adapter=True,production_placement=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
