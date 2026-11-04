"""Compositional public LUT-cone equivalence, including mapper regrouping."""
import collections,functools,itertools,json,pickle,re,sys
from pathlib import Path
from run_units import ROOT,sha

OP={'EFX_FF':('Q',),'EFX_LUT4':('O',),'EFX_ADD':('O','CO'),'EFX_RAM10':('RDATA',),'EFX_DPRAM10':('RDATAA','RDATAB'),'EFX_SRL8':('Q',),'EFX_DSP24':('O','CASCOUT'),'EFX_DSP48':('O','CASCOUT')}
OLD=ROOT/'tmp/frame-execution/connected-board-video-bandwidth83-map2';NEW=ROOT/'tmp/frame-execution/connected-board-video-control84-map1'

class Nodes:
 def __init__(self):self.rows=[('constant',0),('constant',1)];self.ids={v:i for i,v in enumerate(self.rows)}
 def intern(self,row):
  if row not in self.ids:self.ids[row]=len(self.rows);self.rows.append(row)
  return self.ids[row]
 def lut(self,mask,ins):
  children=tuple(sorted(set(ins)-{0,1}));truth=0
  for code in range(1<<len(children)):
   vals={n:(code>>i)&1 for i,n in enumerate(children)};index=sum((n if n in (0,1) else vals[n])<<i for i,n in enumerate(ins));truth|=((mask>>index)&1)<<code
  # Canonicalize duplicate/constant/unobserved inputs and pin permutation.
  for i in reversed(range(len(children))):
   if all(((truth>>x)&1)==((truth>>(x^(1<<i)))&1) for x in range(1<<len(children))):
    truth=sum(((truth>>(x&((1<<i)-1)|((x>>i)<<(i+1))))&1)<<x for x in range(1<<(len(children)-1)))
    children=children[:i]+children[i+1:]
  if not children:return truth&1
  if len(children)==1 and truth==2:return children[0]
  return self.intern(('lut',truth,children))

class BDD:
 def __init__(self):self.nodes=[None,None];self.unique={};self.cache={}
 def node(self,var,low,high):
  if low==high:return low
  key=(var,low,high)
  if key not in self.unique:self.unique[key]=len(self.nodes);self.nodes.append(key)
  assert len(self.nodes)<1000000,'BDD budget exceeded; no proof'
  return self.unique[key]
 def apply(self,op,a,b):
  if a>b:a,b=b,a
  key=(op,a,b)
  if key in self.cache:return self.cache[key]
  if a<2 and b<2:return (a&b) if op=='and' else (a|b) if op=='or' else (a^b)
  v=min(self.nodes[x][0] for x in (a,b) if x>=2)
  def branch(x,i):return self.nodes[x][i] if x>=2 and self.nodes[x][0]==v else x
  out=self.node(v,self.apply(op,branch(a,1),branch(b,1)),self.apply(op,branch(a,2),branch(b,2)));self.cache[key]=out;return out
 def lut(self,mask,args):
  out=0
  for code in range(1<<len(args)):
   if not (mask>>code)&1:continue
   term=1
   for i,x in enumerate(args):term=self.apply('and',term,x if (code>>i)&1 else self.apply('xor',x,1))
   out=self.apply('or',out,term)
  return out

def expand(ts):
 out=[]
 for t in ts:
  if t[0]!='literal':out.append(t);continue
  m=re.fullmatch(r"(\d+)'s?([hbd])([0-9a-fA-F]+)",t[1]);assert m,t
  width=int(m[1]);value=int(m[3],{'b':2,'d':10,'h':16}[m[2]])
  out.extend(('const',(value>>i)&1,None) for i in reversed(range(width)))
 return out

def main():
 sys.setrecursionlimit(10000);factory=Nodes();graphs=[];parameters=[];hashes=[];views=[];touched=[]
 proposal=json.loads((ROOT/'tmp/frame-execution/video-bandwidth83-canonical1/control-factor-proposal.json').read_text())
 maps=[pickle.loads((b/'integration-parse.pkl').read_bytes()) for b in (OLD,NEW)]
 for data,rows in zip(maps,(proposal['original'],proposal['candidate'])):
  for row in rows:
   n=row['name'];ports={'O':[('net',row['output'],None)]};ports.update({f'I{i}':[('net',v,None)] for i,v in enumerate(row['inputs']) if v is not None})
   assert data[1][n]==('EFX_LUT4',ports) and data[2][n]['LUTMASK']==(16,row['mask']),n
 def local(rows,env):
  env=dict(env)
  for row in rows:env[row['output']]=(row['mask']>>sum((env[v] if v else 0)<<i for i,v in enumerate(row['inputs'])))&1
  return env['service63_ddr_33']
 for bits in itertools.product((0,1),repeat=9):
  env=dict(zip(proposal['boundary'],bits));assert local(proposal['original'],env)==local(proposal['candidate'],env)
 cut=('net','service63_ddr_33',None);boundary_values=[]
 for board in (OLD,NEW):
  h,g,ps,_,_=pickle.loads((board/'integration-parse.pkl').read_bytes());assert h==sha(board/'outflow/ti60f225_oob.map.v')
  graphs.append(g);parameters.append(ps);hashes.append(h)
  pins={n:(k,{p:expand(v) for p,v in ports.items()}) for n,(k,ports) in g.items()};driver={};seen=set();active=set()
  for n,(k,ports) in pins.items():
   assert k in OP,k
   for p in OP[k]:
    for i,t in enumerate(ports.get(p,[])):
     if t[0]=='const':continue
     assert t not in driver,(t,n);driver[t]=(n,p,i)
  cache={}
  def expr(t):
   if t in cache:return cache[t]
   assert t not in active,('combinational loop',t);active.add(t)
   if t==cut:out=factory.intern(('proved_512_cut',t))
   elif t[0]=='const':out=t[1]
   elif t not in driver:out=factory.intern(('external',t))
   else:
    n,p,i=driver[t];k,ports=pins[n]
    if k!='EFX_LUT4':out=factory.intern(('cut',n,p,i))
    else:
     mask=ps[n]['LUTMASK'][1];inputs=[]
     for j in range(4):
      values=ports.get(f'I{j}')
      if values is None:
       assert all(((mask>>v)&1)==((mask>>(v^(1<<j)))&1) for v in range(16)),(n,j)
       inputs.append(0)
      else:assert len(values)==1;inputs.append(expr(values[0]))
     out=factory.lut(mask,inputs)
   active.remove(t);cache[t]=out;seen.add(out);return out
  v={}
  for n,(k,ports) in pins.items():
   if k=='EFX_LUT4':continue
   for p,ts in ports.items():
    if p not in OP[k]:v[n,p]=tuple(expr(t) for t in ts)
  # Include all public LUT outputs, even those terminating at protected IP
  # or top-level ports that the public primitive parser does not enumerate.
  outputs={}
  for n,(k,ports) in pins.items():
   if k=='EFX_LUT4':
    ts=ports['O'];assert len(ts)==1;outputs[ts[0]]=expr(ts[0])
  boundary_values.append({name:expr(('net',name,None)) for name in proposal['boundary']})
  views.append((v,outputs));touched.append(seen)
 # The512-combination cut is sound only if all nine actual input functions
 # match. Require exact canonical DAG equality, not same wire spelling.
 assert boundary_values[0]==boundary_values[1],'Factored-cone boundary functions changed'
 old,new=graphs
 assert {n for n,(k,_) in old.items() if k!='EFX_LUT4'}=={n for n,(k,_) in new.items() if k!='EFX_LUT4'}
 for n,(k,p) in old.items():
  if k=='EFX_LUT4':continue
  assert new[n][0]==k and parameters[0].get(n,{})==parameters[1].get(n,{}),n
  assert set(p)==set(new[n][1]),n
  for pin in OP[k]:assert len(expand(p.get(pin,[])))==len(expand(new[n][1].get(pin,[]))), (n,pin)
 # Fully expand LUT cones for ROBDD comparison. Cutting every common DAG
 # node would lose relationships between a composite and its descendants.
 bdd=BDD();memo={};proofs=[]
 def formal(node):
  if node<2:return node
  if node in memo:return memo[node]
  row=factory.rows[node]
  if row[0]!='lut':out=bdd.node(node,0,1)
  else:out=bdd.lut(row[1],[formal(x) for x in row[2]])
  memo[node]=out;return out
 left,right=views[0][0],views[1][0];assert set(left)==set(right)
 checked=0
 for endpoint in sorted(left):
  a,b=left[endpoint],right[endpoint];assert len(a)==len(b),endpoint
  for bit,(x,y) in enumerate(zip(a,b)):
   checked+=1
   if x==y:continue
   assert formal(x)==formal(y),('non-equivalent',endpoint,bit,x,y)
   proofs.append(dict(cell=endpoint[0],pin=endpoint[1],bit=bit,method='ROBDD'))
 # Bind every retained native public LUT output; auto-generated nNN aliases
 # can be renamed/reused, so named auto nets are not semantic identities.
 from native_local_drivers import graph
 src=ROOT/'tmp/frame-execution/video-bandwidth83-input1/native_source.v'
 native=graph(src.read_text());native_outputs=[]
 for n,(k,ports) in native.items():
  if k!='EFX_LUT4' or n=='SERVICE63_DDR_29':continue
  ts=ports['O'];assert len(ts)==1;t=ts[0]
  a=views[0][1].get(t);b=views[1][1].get(t)
  assert a is not None and b is not None,(n,t)
  assert a==b or formal(a)==formal(b),('native_output',n)
  native_outputs.append(n)
 result=dict(status='CONTROL84_PUBLIC_COMBINATIONAL_BOUNDARIES_EQUIVALENT',map_sha256=hashes[1],parent_map_sha256=hashes[0],
  input_bits_compared=checked,roBDD_different_structure_bits=proofs,native_lut_outputs_compared=len(native_outputs),canonical_nodes=len(factory.rows),bdd_nodes=len(bdd.nodes),factored_cut_combinations=512,nine_boundary_functions_identical=True,
  nonlut_parameters_and_port_widths_identical=True,protected_equivalence_claim=False,
  evidence={str(f):sha(f) for f in (Path(__file__),src,OLD/'inputs.json',NEW/'inputs.json',OLD/'outflow/ti60f225_oob.map.v',NEW/'outflow/ti60f225_oob.map.v')},
  limits=['Binary public combinational cone equivalence at every non-LUT input and retained native LUT output; unchanged primitive parameters and equal output widths.','RAM, DSP and ADD behavior is composed through matching cell/port cut points; no claim to formal proof of encrypted IP.','Physical timing/reset pulse integrity still requires new implementation checks.'],hardware_access=False)
 (NEW/'public-comb-equivalence.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','input_bits_compared','native_lut_outputs_compared','canonical_nodes','bdd_nodes')}))

if __name__=='__main__':main()
