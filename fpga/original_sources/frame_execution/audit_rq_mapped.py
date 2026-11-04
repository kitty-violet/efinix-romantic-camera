"""Independently audit the completed four-state mapped test and every omitted pin."""
import collections,json,re
from pathlib import Path
from run_units import HERE,ROOT,sha
def main():
 src=ROOT/'tmp/frame-execution/rq-source2';gate=ROOT/'tmp/frame-execution/rq-mapped1'
 mp=ROOT/'tmp/frame-execution/rq-map2';old=ROOT/'tmp/frame-execution/rq-original-map2'
 body=(gate/'seed71.log').read_text();net=(gate/'rq_single_dsp.map.v').read_text()
 sr=json.loads((src/'result.json').read_text());mr=json.loads((mp/'result.json').read_text());orr=json.loads((old/'result.json').read_text())
 gm=json.loads((gate/'inputs.json').read_text())
 for r in gm['inputs']:assert sha(r['path'])==sha(r['snapshot'])==r['sha256'],r['path']
 assert sha(gate/'rq_single_dsp.map.v')==mr['map_sha256']
 assert 'Errors: 0, Warnings: 287' in body
 assert not re.search(r'\*\* (Error|Fatal)|\bFAIL\b',body)
 line=re.findall(r'^# (PASS RQ_SINGLE_DSP.*)$',body,re.M);assert line==[sr['results'][0]['line']]
 assert 'Errors: 0, Warnings: 0' in (gate/'compile.log').read_text()
 masks={n:int(v,16) for n,v in re.findall(r"defparam (\S+)\.LUTMASK = 16'h([0-9a-fA-F]+);",net)}
 netlines=net.splitlines();pins=collections.Counter();warns=collections.Counter();sites=collections.defaultdict(list)
 for l in body.splitlines():
  if not l.startswith('# ** Warning:'):continue
  m=re.fullmatch(r"# \*\* Warning: \(vsim-(3017|3722)\) .*rq_single_dsp\.map\.v\((\d+)\): \[TFMPC\] - (.*)",l);assert m,l
  code,ln,detail=m[1],int(m[2]),m[3];warns[code]+=1;sites[ln].append((code,detail))
  if code=='3017':continue
  pin=re.fullmatch(r"Missing connection for port '(\w+)'\.",detail)[1]
  kind,name=re.search(r'(EFX_\w+)\s+(\S+)\s*\(',netlines[ln-1]).groups()
  if kind=='EFX_LUT4':
   assert re.fullmatch(r'I[0-3]',pin)
   bit=int(pin[1]);mask=masks[name]
   assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16)),(name,pin)
  elif kind=='EFX_ADD':assert pin in ('O','CO')
  else:assert kind=='EFX_DSP48' and pin in ('CASCOUT','OVFL')
  instance='\n'.join(netlines[ln-1:]).split(';',1)[0]
  assert not re.search(r'\.'+pin+r'\s*\(',instance)
  pins[kind+'.'+pin]+=1
 for ln,rows in sites.items():
  count=[re.fullmatch(r'Too few port connections\. Expected (\d+), found (\d+)\.',d) for c,d in rows if c=='3017']
  assert len(count)==1 and count[0]
  assert int(count[0][1])-int(count[0][2])==sum(c=='3722' for c,d in rows)
 assert sum(warns.values())==287
 result=dict(status='EXACT_SINGLE_DSP_LOCAL_SOURCE_MAP_AND_FOUR_STATE_NUMERIC_PASS',
   source_checks=sum(int(re.search(r'checked=(\d+)',r['line'])[1]) for r in sr['results']),
   mapped_checks=557984,source_map_seed71_trace_equal=True,
   source_seeds=[71,391,1027],mapped_seeds=[71],
   old_resources=orr['resources'],new_resources=mr['resources'],
   warnings=dict(warns),proved_unused_pins=dict(pins),unknown_warnings=0,
   raw_runner_rejection='Mapped simulation completed numerically; original runner required zero warnings. All287 omitted-pin warnings independently proved here; historical failure retained.',
   scope='Exact original32-bit value, signed18-bit Q17, INT8 saturation/ReLU, invalid-field flag and four-edge interface. Not full pool P&R or frame performance.',
   full_frame=False,pnr=False,performance_pass=False,
   evidence=[dict(path=str(p),sha256=sha(p)) for p in (src/'result.json',mp/'result.json',old/'result.json',gate/'inputs.json',gate/'seed71.log',gate/'rq_single_dsp.map.v',Path(__file__))])
 out=ROOT/'output/toolchain/rq_single_dsp_audit_2026-09-30.json';assert not out.exists()
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
