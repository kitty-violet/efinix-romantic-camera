"""Prove mapper LUT renaming/factoring at every retained primitive boundary."""
import functools,re
from camera_joint_native import OUT

def compare(a,b,pa,pb):
 nonlut={n for n,(k,_) in a.items() if k!='EFX_LUT4'}
 assert nonlut=={n for n,(k,_) in b.items() if k!='EFX_LUT4'}
 for n in nonlut:
  assert a[n][0]==b[n][0] and pa[n]==pb[n],('retained primitive parameters',n)
 common={n for n in a.keys()&b.keys() if a[n][0]==b[n][0]=='EFX_LUT4'}
 drivers=[{t:(n,p,i) for n,(k,pins) in g.items() for p in OUT[k] for i,t in enumerate(pins.get(p,[])) if t[0]=='net'} for g in (a,b)]
 expanded=set();rounds=[]
 for iteration in range(20):
  anchors=nonlut|common;nodes={};unique={}
  def node(v,lo,hi):
   if lo==hi:return lo
   key=(v,lo,hi)
   if key not in unique:
    assert len(nodes)<1500000,'bounded BDD node budget'
    unique[key]=len(nodes)+2;nodes[unique[key]]=key
   return unique[key]
  @functools.lru_cache(None)
  def ite(c,t,f):
   if c<2:return t if c else f
   if t==f:return t
   if t==1 and f==0:return c
   v=min(nodes[x][0] for x in (c,t,f) if x>=2)
   def part(x,hi):return nodes[x][2 if hi else 1] if x>=2 and nodes[x][0]==v else x
   return node(v,ite(*(part(x,False) for x in (c,t,f))),ite(*(part(x,True) for x in (c,t,f))))
  @functools.lru_cache(None)
  def value(side,t):
   if t[0]=='literal':
    m=re.fullmatch(r"\d+'([bhd])([0-9a-f]+)",t[1],re.I);assert m,t
    v=int(m[2],{'b':2,'h':16,'d':10}[m[1].lower()]);assert v in (0,1),t;return v
   info=drivers[side].get(t)
   if info is None:return node(f'{t[2] if isinstance(t[2],int) else -1:06d}:external:'+repr(t),0,1)
   n,p,i=info
   bits=re.findall(r'\[(\d+)\]',n)
   bit=int(bits[-1]) if bits else i
   if n in anchors:return node(f'{bit:06d}:cell:'+repr((n,p,i)),0,1)
   assert (a,b)[side][n][0]=='EFX_LUT4',n
   return function(side,n)
  @functools.lru_cache(None)
  def function(side,n):
   pins=(a,b)[side][n][1];mask=(pa,pb)[side][n]['LUTMASK'][1]
   ins=[pins.get('I'+str(i),[('literal',"1'b0",None)])[0] for i in range(4)]
   def build(i,offset):
    if i==4:return mask>>offset&1
    lo,hi=build(i+1,offset),build(i+1,offset|(1<<i))
    return lo if lo==hi else ite(value(side,ins[i]),hi,lo)
   return build(0,0)
  bad={n for n in common if function(0,n)!=function(1,n)}
  rounds.append(dict(iteration=iteration,anchor_luts=len(common),expanded_this_round=len(bad),bdd_nodes=len(nodes)))
  print('COMBINATIONAL_EQUIVALENCE',rounds[-1],flush=True)
  if bad:
   expanded.update(bad);common-=bad;continue
  checks=0;forward={}
  for n in sorted(nonlut):
   kind,x=a[n];y=b[n][1];assert x.keys()==y.keys(),('primitive ports',n)
   for p in x:
    assert len(x[p])==len(y[p]),(n,p)
    for t,u in zip(x[p],y[p]):
     if p not in OUT[kind]:assert value(0,t)==value(1,u),('non-LUT input function',n,p,t,u)
     if t[0]=='net' and u[0]=='net':
      assert t not in forward or forward[t]==u,('ambiguous forward boundary',n,p,t,u)
      forward[t]=u
     checks+=1
  for n in common:
   x=a[n][1]['O'][0];y=b[n][1]['O'][0]
   assert x not in forward or forward[x]==y
   forward[x]=y
  # Explicitly report structural differences; do not call renamed LUTs
  # byte-identical or reuse old physical timing.
  proof=dict(status='PUBLIC_COMBINATIONAL_BOUNDARIES_BOOLEAN_EQUIVALENT',retained_nonlut_cells=len(nonlut),
   equivalent_named_luts=len(common),expanded_common_luts=sorted(expanded),removed_luts=sorted(a.keys()-b.keys()),added_luts=sorted(b.keys()-a.keys()),
   primitive_pin_checks=checks,rounds=rounds,scope='All Boolean inputs at retained FF/RAM/DSP/ADD/SRL boundaries; no arbitrary-X or routed-timing claim.')
  return forward,proof
 raise AssertionError('LUT boundary refinement budget exhausted')
