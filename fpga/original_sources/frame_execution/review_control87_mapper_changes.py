"""Compare changed mapped input functions at identical named state outputs.

Only recursively proven structural functions may become shared BDD boundaries.
Named state outputs and identical protected-cell interface ports are primary
inputs to this local proof; no protected-cell internals are proved here.
Auto-net names are never equated. This is local binary next-input evidence,
not whole-chip sequential/4-state equivalence or physical release.
"""
import collections,functools,json,pickle,re,time
from pathlib import Path
from run_units import ROOT,sha
from camera_joint_native import OUT
from native_local_drivers import lexical_tokens

def main():
 start=time.monotonic();base=ROOT/'tmp/frame-execution/connected-board-video-control84-map1';root=ROOT/'tmp/frame-execution/connected-board-video-control87-map1'
 auditfile=root/'control87-candidate-audit.json';audit=json.loads(auditfile.read_text())
 maps=[pickle.loads((p/'integration-parse.pkl').read_bytes()) for p in (base,root)]
 assert maps[1][0]==audit['map_sha256']
 interfaces=[];top_inputs=[]
 for folder in (base,root):
  text=(folder/'outflow/ti60f225_oob.map.v').read_text()
  match=re.search(r'protected_cell\s+protected_inst\s*\((.*?)\);',text,re.S);assert match
  interfaces.append({n.lstrip('\\'):lexical_tokens(v) for n,v in re.findall(r'\.([^\s(]+)\s*\(([^()]*)\)',match[1])})
  header=re.sub(r'/\*.*?\*/|//[^\n]*','',text.split('\n    wire',1)[0],flags=re.S)
  primary=set()
  for declaration in re.findall(r'^\s*input\s+(.*?);',header,re.M|re.S):
   m=re.fullmatch(r'\s*(?:\[(\d+):(\d+)\])?\s*([^\s,]+)\s*',declaration);assert m,declaration
   name=m[3].lstrip('\\')
   if m[1] is None:primary.add(('net',name,None))
   else:
    high,low=int(m[1]),int(m[2]);primary.update(('net',name,bit) for bit in range(min(high,low),max(high,low)+1))
  top_inputs.append(primary)
 assert interfaces[0]==interfaces[1] and len(interfaces[0])==1201
 assert top_inputs[0]==top_inputs[1] and top_inputs[0]
 interface_tokens=collections.defaultdict(list)
 for pin,ts in interfaces[0].items():
  for bit,t in enumerate(ts):
   if t[0]=='net':interface_tokens[t].append((pin,bit))
 external_inputs=set()
 targets=audit['unclassified_nonlut_changes'];intern={};structures={};seen=[set(),set()];origins={}
 def unique(key):
  if key not in intern:intern[key]=len(intern)+2;structures[intern[key]]=key
  return intern[key]
 def literal(token):
  m=re.fullmatch(r"\d+'([bhd])([0-9a-f]+)",token[1],re.I);assert m,token
  value=int(m[2],{'b':2,'h':16,'d':10}[m[1].lower()]);assert value in (0,1),token
  return value
 @functools.lru_cache(None)
 def signature(side,token):
  assert time.monotonic()-start<120,'bounded review exceeded120s'
  if token[0]=='literal':return literal(token)
  _,g,params,drivers,_=maps[side];name=drivers.get(token)
  if name is None:
   assert (token in interface_tokens or token in top_inputs[0]) and maps[1-side][3].get(token) is None,('unresolved input',side,token)
   external_inputs.add(token)
   key=('top-level-input',token) if token in top_inputs[0] else ('protected-interface-input',tuple(interface_tokens[token]))
   result=unique(key)
   seen[side].add(result);return result
  kind,pins=g[name];ports=OUT[kind];output=[(p,i) for p in ports for i,t in enumerate(pins.get(p,[])) if t==token]
  assert len(output)==1,(name,token,output)
  if kind not in ('EFX_LUT4','EFX_ADD'):
   assert name in maps[1-side][1] and maps[1-side][1][name][0]==kind
   assert params.get(name,{})==maps[1-side][2].get(name,{})
   key=('state-output',kind,name,output[0])
  else:
   inputs=tuple((p,tuple(signature(side,t) for t in ts)) for p,ts in sorted(pins.items()) if p not in ports)
   key=('comb',kind,output[0],tuple(sorted(params.get(name,{}).items())),inputs)
  result=unique(key);seen[side].add(result);origins[side,token]=name;return result
 pairs=[]
 for row in targets:
  assert len(row['before'])==len(row['after'])
  for bit,(a,b) in enumerate(zip(row['before'],row['after'])):
   if a==b:continue
   a,b=tuple(a),tuple(b);pairs.append((row['cell'],row['pin'],bit,a,b,signature(0,a),signature(1,b)))
 common=seen[0]&seen[1];nodes={};bdd_unique={};boundaries=set();expanded=[set(),set()]
 def node(v,lo,hi):
  assert time.monotonic()-start<120,'bounded review exceeded120s'
  if lo==hi:return lo
  key=(v,lo,hi)
  if key not in bdd_unique:
   assert len(nodes)<200000,'bounded BDD limit'
   bdd_unique[key]=len(nodes)+2;nodes[bdd_unique[key]]=key
  return bdd_unique[key]
 @functools.lru_cache(None)
 def ite(c,t,f):
  if c<2:return t if c else f
  if t==f:return t
  if t==1 and f==0:return c
  v=min(nodes[n][0] for n in (c,t,f) if n>=2)
  def part(n,high):return nodes[n][2 if high else 1] if n>=2 and nodes[n][0]==v else n
  return node(v,ite(*(part(n,False) for n in (c,t,f))),ite(*(part(n,True) for n in (c,t,f))))
 @functools.lru_cache(None)
 def value(side,token,expand=False):
  s=signature(side,token)
  if s<2:return s
  if s in common and (not expand or structures[s][0]!='comb' or structures[s][1]!='EFX_LUT4'):
   boundaries.add(s);return node(s,0,1)
  _,g,params,drivers,_=maps[side];name=drivers[token];kind,pins=g[name]
  assert kind=='EFX_LUT4',('unmatched non-LUT boundary',side,name,kind)
  expanded[side].add(name);mask=params[name]['LUTMASK'][1]
  def build(i,offset):
   if i==4:return (mask>>offset)&1
   lo,hi=build(i+1,offset),build(i+1,offset|(1<<i))
   if lo==hi:return lo
   assert f'I{i}' in pins,('dependent unconnected pin',name,i)
   return ite(value(side,pins[f'I{i}'][0],expand),hi,lo)
  return build(0,0)
 rows=[]
 for name,pin,bit,a,b,x,y in pairs:
  same=x==y
  proof='recursive structure'
  if not same:
   proof='BDD at proven common functions'
   if value(0,a)!=value(1,b):
    # Independent cuts can lose correlations, giving a false difference.
    # Expand every LUT to named state/IP/primary and unchanged ADD functions.
    assert value(0,a,True)==value(1,b,True),('changed input function',name,pin,bit)
    proof='BDD with shared LUTs expanded'
  rows.append(dict(cell=name,pin=pin,bit=bit,proof=proof))
 # Newly omitted LUT inputs must be irrelevant to every binary truth-table row.
 missing=[]
 for n in audit['mapped_added']+audit['new_cone']:
  kind,pins=maps[1][1][n]
  if kind!='EFX_LUT4':continue
  mask=maps[1][2][n]['LUTMASK'][1]
  for i in range(4):
   if f'I{i}' in pins:continue
   assert all(((mask>>v)^(mask>>(v^(1<<i))))&1==0 for v in range(16)),(n,i)
   missing.append([n,f'I{i}'])
 result=dict(status='CONTROL87_CHANGED_INPUT_BINARY_FUNCTIONS_PASS_NOT_FULL_RELEASE',changed_nonlut_pins=len(targets),scalar_changed_tokens=len(pairs),proof_counts=dict(collections.Counter(r['proof'] for r in rows)),rows=rows,
  recursive_structures=len(structures),common_structures=len(common),bdd_nodes=len(nodes),bdd_boundaries=len(boundaries),expanded_luts=[len(s) for s in expanded],unused_input_checks=missing,
  protected_interface_pins_identical=1201,top_level_input_bits_identical=len(top_inputs[0]),external_boundary_tokens=sorted(external_inputs,key=str),
  old_map_sha256=maps[0][0],map_sha256=maps[1][0],seconds=time.monotonic()-start,
  scope='Only listed changed input functions, binary identical named state-output and protected-interface assignment. All1201 protected interface connections match; protected internals not proved. Common cuts recursively proved from these primary boundaries and identical primitive parameters, never auto-net identity. Not complete sequential or4-state proof.',
  evidence={str(f):sha(f) for f in (Path(__file__).resolve(),auditfile,base/'outflow/ti60f225_oob.map.v',root/'outflow/ti60f225_oob.map.v')},hardware_access=False,board_release=False)
 (root/'mapper-input-function-proof.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:result[k] for k in ('status','changed_nonlut_pins','scalar_changed_tokens','proof_counts','recursive_structures','bdd_nodes','seconds')}))
if __name__=='__main__':main()
