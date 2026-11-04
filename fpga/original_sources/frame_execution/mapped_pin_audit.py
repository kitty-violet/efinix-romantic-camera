"""Prove every generated omitted input irrelevant; never silence unknown warnings."""
import collections,re
def audit(body,net):
 masks={n:int(v,16) for n,v in re.findall(r"defparam (\S+)\.LUTMASK = 16'h([0-9a-fA-F]+);",net)}
 lines=net.splitlines();pins=collections.Counter();codes=collections.Counter();sites=collections.defaultdict(list)
 assert not re.search(r'\*\* (Error|Fatal)|\bFAIL\b',body)
 for line in body.splitlines():
  if not line.startswith('# ** Warning:'):continue
  m=re.fullmatch(r"# \*\* Warning: \(vsim-(3017|3722)\) .*\.map\.v\((\d+)\): \[TFMPC\] - (.*)",line);assert m,line
  code,ln,detail=m[1],int(m[2]),m[3];codes[code]+=1;sites[ln].append((code,detail))
  if code=='3017':continue
  pin=re.fullmatch(r"Missing connection for port '(\w+)'\.",detail)[1]
  kind,name=re.search(r'(EFX_\w+)\s+(\S+)\s*\(',lines[ln-1]).groups()
  if kind=='EFX_LUT4':
   assert re.fullmatch(r'I[0-3]',pin);bit=int(pin[1]);mask=masks[name]
   assert all((mask>>i&1)==(mask>>(i^(1<<bit))&1) for i in range(16)),(name,pin)
  elif kind=='EFX_ADD':assert pin in ('O','CO')
  else:assert kind=='EFX_DSP48' and pin in ('CASCOUT','OVFL')
  assert not re.search(r'\.'+pin+r'\s*\(','\n'.join(lines[ln-1:]).split(';',1)[0]);pins[kind+'.'+pin]+=1
 for ln,rows in sites.items():
  counts=[re.fullmatch(r'Too few port connections\. Expected (\d+), found (\d+)\.',d) for c,d in rows if c=='3017']
  assert len(counts)==1 and counts[0]
  assert int(counts[0][1])-int(counts[0][2])==sum(c=='3722' for c,d in rows)
 total=sum(codes.values());assert re.search(r'Errors: 0, Warnings: '+str(total)+r'\b',body)
 return dict(total=total,codes=dict(codes),proved_unused_pins=dict(pins),unknown=0)
