"""Require nonempty, numeric Gray skew evidence from the routed circuit."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
b=a.board.resolve();assert b.is_relative_to(ROOT/'tmp')
proof=b/'clock_crossing_inventory.json';d=json.loads(proof.read_text())
r=b/'outflow/ti60f225_oob.gray_skew_uncut.rpt';body=r.read_text()
counts=[int(x) for x in re.findall(r'^Endpoints:\s*(\d+)',body,re.M)]
slacks=[float(x) for x in re.findall(r'^Slack:\s*([\-\d.]+) ns',body,re.M)]
assert counts==[7]*10 and len(slacks)==70 and min(slacks)>=0
assert sha(b/'ti60f225_oob.sdc')==d['sdc_sha256']
assert sha(b/'outflow/ti60f225_oob.timing.rpt')==d['timing_sha256']
for bus in d['gray_buses']:
 for start,end in bus['pins']:assert start in body and end in body
out=b/'gray_skew_audit.json';assert not out.exists()
result=dict(status='ROUTED_GRAY_SKEW_PASS_NOT_RESET_OR_CDC_RELEASE',buses=10,endpoints=70,
 max_skew_ns=round(2.5-min(slacks),3),min_margin_ns=min(slacks),limit_ns=2.5,
 report_sha256=sha(r),inventory_sha256=sha(proof),sdc_sha256=sha(b/'ti60f225_oob.sdc'),
 scope='Same routed circuit, original clock relationships retained to prevent this Efinity version filtering asynchronous paths from report_bus_skew.',
 reset_safety_proven=False,cdc_release=False,board_release=False)
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
