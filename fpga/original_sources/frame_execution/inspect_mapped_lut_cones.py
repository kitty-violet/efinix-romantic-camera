"""Read-only flattened mapped-netlist LUT cone attribution, not a fit gate."""
import argparse,collections,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--net',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
body=a.net.read_text();cells=re.findall(r'^\s*(EFX_[A-Z0-9_]+)\s+(.+?)\s*\((.*?)\)\s*/\*',body,re.M|re.S)
normalize=lambda s:re.sub(r'\s+','',s).replace('\\','')
netrefs=lambda s:[normalize(x) for x in re.findall(r'\\[^\s,{}]+\s*(?:\[\d+\])?|(?<![\w\'])\b[a-zA-Z_][\w$]*(?:\[\d+\])?',s)]
drivers={};roots=collections.defaultdict(set)
for kind,name,connections in cells:
 ports={k:netrefs(v) for k,v in re.findall(r'\.(\w+)\(([^()]*)\)',connections)}
 if kind=='EFX_LUT4':
  out=ports.get('O',[])
  if len(out)==1:drivers[out[0]]=set(n for k,v in ports.items() if k.startswith('I') for n in v)
 elif kind in ('EFX_FF','EFX_DSP24','EFX_DSP48'):
  label=normalize(name).split('/')[0]
  keys=('D',) if kind=='EFX_FF' else ('A','B','C','OP')
  roots[label].update(n for k in keys for n in ports.get(k,[]))
def cone(initial):
 seen=set();stack=list(initial)
 while stack:
  n=stack.pop()
  if n in seen or n not in drivers:continue
  seen.add(n);stack.extend(drivers[n])
 return seen
rows=[]
for label,nets in roots.items():
 c=cone(nets);rows.append(dict(label=label,luts=len(c),roots=len(nets)))
rows.sort(key=lambda x:-x['luts']);result=dict(scope='Overlapping combinational cones ending at FF/DSP data inputs; not additive',lut_cells=len(drivers),cells=len(cells),rows=rows)
a.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(result,rows=rows[:20])))
