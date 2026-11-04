"""Attribute mapped LUT cones to arithmetic data/control outputs, without RTL estimates."""
import argparse,collections,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('netlist',type=Path);a=p.parse_args()
t=a.netlist.read_text()
def norm(s):return re.sub(r'\s+','',s)
def nets(s):
 out=[]
 for v in s.replace('{','').replace('}','').split(','):
  v=norm(v)
  if not v or "'" in v:continue
  m=re.fullmatch(r'(.+)\[(\d+):(\d+)\]',v)
  if m:
   name,hi,lo=m.groups();out += [name+'['+str(i)+']' for i in range(int(lo),int(hi)+1)]
  else:out.append(v)
 return out
lut={};roots=collections.defaultdict(set);lutcells={}
for m in re.finditer(r'^\s+EFX_(LUT4|DSP24|DSP48|FF)\s+(\S+)\s+\((.*?)\)\s*/\*',t,re.M|re.S):
 kind,name,pins=m.groups();pins=dict(re.findall(r'\.(\w+)\((.*?)\)',pins,re.S))
 if kind=='LUT4':
  o=norm(pins['O']);lut[o]=sum([nets(pins[k]) for k in ['I0','I1','I2','I3'] if k in pins],[]);lutcells[o]=name
 if kind in ['DSP24','DSP48']:
  for k in ['A','B','CE','RST','OP']:
   if k in pins:roots[kind+'_'+k].update(nets(pins[k]))
 if kind=='FF':
  bucket='OTHER'
  for x in ['reader/','dispatch/','skip_fuse/','pool/']:
   if x in name:bucket=x[:-1];break
  roots['FF_'+bucket].update(nets(pins.get('D','')))
def cone(todo):
 found=set();pending=list(todo)
 while pending:
  v=pending.pop()
  if v in lut and v not in found:found.add(v);pending+=lut[v]
 return found
cones={k:cone(v) for k,v in roots.items()}
print(json.dumps(dict(lut_count=len(lut),cones={k:len(v) for k,v in cones.items()},union_arithmetic_inputs=len(set().union(*[v for k,v in cones.items() if k.startswith('DSP')]))),indent=2))
