"""Audit completed physical reports without rerunning or editing evidence."""
import argparse,json,re
from pathlib import Path
from run_units import sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
out=r/'strict_audit.json';assert not out.exists()
result=json.loads((r/'result.json').read_text());inputs=json.loads((r/'inputs.json').read_text())
assert sha(r/'inputs.json')==result['inputs_sha256']
for x in inputs['inputs']:assert sha(x['snapshot'])==x['sha256']
assert [x['status'] for x in result['stages'][:3]]==['PASS']*3
log=r/'sta_tclsh.log';body=log.read_text();assert 'The entire flow of EFX_STA took' in body and not re.search(r'(?mi)^(ERROR|FATAL)',body)
ct=r/'outflow/dw8_stream_physical.check_timing.rpt';body=ct.read_text();checks={}
for key in ('no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','multiple_clock'):
 m=re.search(r'checking '+key+r' \((\d+)\)',body);assert m;checks[key]=int(m[1]);assert checks[key]==0
timing={}
for key in ('setup','hold','process150','process150_hold'):
 f=r/f'outflow/dw8_stream_physical.strict_{key}.rpt';m=re.search(r'Slack\s*:\s*([-+0-9.]+)',f.read_text());assert m
 timing[key]=dict(slack_ns=float(m[1]),sha256=sha(f))
sites={};loc=r/'outflow/dw8_stream_physical.out.placeloc'
for line in loc.read_text().splitlines():
 z=line.split()
 if len(z)==4 and re.fullmatch(r'arithmetic/ROW\[\d\].CHANNEL\[\d\].TAP\[\d\].mac',z[0]):sites.setdefault(','.join(z[1:3]),[]).append(z[3])
assert len(sites)==128 and all(sorted(v)==['0','1'] for v in sites.values())
ok=all(v['slack_ns']>=0 for v in timing.values())
evidence=dict(status='FIXED_WINDOW_SINGLE_SEED_STRICT_STA_PASS' if ok else 'STRICT_STA_FAIL',physical_dsp=128,checks=checks,timing=timing,
 inputs_sha256=sha(r/'inputs.json'),map_sha256=sha(r/'outflow/dw8_stream_physical.map.v'),placement_sha256=sha(loc),sta_log_sha256=sha(log),check_timing_sha256=sha(ct),
 runner_summary_correction='Original wrapper expected a sta_tclsh:PASS footer that EFX_STA does not print; reports are audited independently here. Original result preserved.',whole_board=False,real_supply=False)
out.write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence))
