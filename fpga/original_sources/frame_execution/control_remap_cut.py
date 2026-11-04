"""Bit-exact cut boundaries including whole-vector and nonprimitive loads."""
import copy,re
from native_local_drivers import CELL,REF,lexical_tokens
from continuous_control_native import connections

def boundaries(g,text,owned):
 ranges={}
 for a,b,names in re.findall(r'^\s*wire\s+\[(\d+):(\d+)\]\s*([^;]+);',text,re.M):
  for n in re.sub(r'/\*.*?\*/','',names,flags=re.S).split(','):
   n=n.strip().lstrip('\\');assert n and not any(c.isspace() for c in n)
   ranges[n]=(int(a),int(b))
 def expand(t):
  if t[0]=='net' and t[2] is None and t[1] in ranges:
   a,b=ranges[t[1]];step=-1 if a>b else 1
   return [('net',t[1],i) for i in range(a,b+step,step)]
  return [t]
 full={n:(k,{p:[v for t in ts for v in expand(t)] for p,ts in pins.items()}) for n,(k,pins) in g.items()}
 dr,loads=connections(full)
 exterior=CELL.sub('',text)
 exterior=re.sub(r'\(\*.*?\*\)','',exterior,flags=re.S)
 exterior=re.sub(r'^\s*(?:defparam|wire)\b[^;]*;','',exterior,flags=re.M)
 exterior=re.sub(r'/\*.*?\*/|//[^\n]*','',exterior,flags=re.S)
 external={v for m in REF.finditer(exterior) for t in lexical_tokens(m[0]) for v in expand(t) if v[0]=='net'}
 targets={n for n in owned if full[n][1]['O'][0] in external or any(s not in owned for s,p in loads[full[n][1]['O'][0]])}
 inputs={t for n in owned for p,ts in full[n][1].items() if p!='O' for t in ts if t[0]=='net' and dr.get(t) not in owned}
 return targets,inputs,dict(vector_declarations=len(ranges),external_reference_bits=len(external))
