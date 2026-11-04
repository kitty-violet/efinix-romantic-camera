"""Partition mapped LUT cells by combinational output cones, stopping at state.

Overlaps are explicit; these counts are not separate addable implementations.
"""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha
p=argparse.ArgumentParser();p.add_argument('--netlist',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
text=a.netlist.read_text();cells={};driver={};aliases={}
def net(s):return s.strip().removeprefix('\\')
for m in re.finditer(r'^\s*EFX_(\w+)\s+(\S+)\s*\((.*?)\)\s*/\*.*?;',text,re.M|re.S):
 kind,name,body=m.groups();ports={k:net(v) for k,v in re.findall(r'\.(\w+)\(([^()]*)\)',body)}
 cells[name]=(kind,ports)
 for pin in ('O','Q','Q7','CO'):
  if ports.get(pin):driver[ports[pin]]=name
for m in re.finditer(r'^\s*assign\s+(\S+)\s*=\s*([^;]+);',text,re.M):
 lhs,rhs=m.groups()
 if not any(c in rhs for c in '{}~&|^?:'):aliases[net(lhs)]=net(rhs)
def cone(signals):
 seen=set();todo=list(signals);visited=set()
 while todo:
  n=todo.pop()
  if n in visited:continue
  visited.add(n)
  if n in aliases:todo.append(aliases[n])
  name=driver.get(n)
  if name is None:continue
  kind,ports=cells[name]
  if kind not in ('LUT4','COMB4','ADD'):continue
  seen.add(name)
  todo.extend(v for k,v in ports.items() if k not in ('O','Q','Q7','CO'))
 return seen
sets={}
for group,prefixes in [('activation',['pool_data']),('weight',['pool_weights']),('metadata',['pool_tag','pool_group','pool_terms','pool_mask']),('handshake',['pool_valid','pool_first','pool_last','in_ready','weight_ready','done_valid','busy','error','phase_advance'])]:
 sets[group]=cone(n for n in driver|aliases if any(n==x or n.startswith(x+'[') for x in prefixes))
counts={k:dict(collections.Counter(cells[n][0] for n in v)) for k,v in sets.items()}
exclusive=collections.Counter()
for n,(kind,ports) in cells.items():
 if kind not in ('LUT4','COMB4','ADD'):continue
 groups=tuple(k for k,v in sets.items() if n in v)
 exclusive[(','.join(groups) or 'state_or_other',kind)]+=1
out=dict(netlist=str(a.netlist.resolve()),sha256=sha(a.netlist),cells=len(cells),cones=counts,
 exclusive=[dict(cones=k[0],kind=k[1],count=v) for k,v in exclusive.most_common()],
 note='Combinational cones stop at registers/SRL/DSP. Cross-output overlaps are explicit; no area savings claim.')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
