"""Exact four-state proofs for mapper-pruned LUT identities and duplicates."""
import itertools
from camera_joint_native import OUT
from audit_video_service_rq_fast import lut
ZERO=('literal',"1'b0",None)
def normalize_frontier_aliases(a,b,pa):
 missing=set(a)-set(b);assert all(a[n][0]=='EFX_LUT4' for n in missing),missing
 alias={};rows=[]
 def root(t):
  seen=set()
  while t in alias:assert t not in seen;seen.add(t);t=alias[t]
  return t
 def ins(n):return [root(a[n][1].get('I'+str(i),[ZERO])[0]) for i in range(4)]
 def signature(n):
  ts=ins(n);leaves=sorted({t for t in ts if t[0]=='net'},key=str);values=[]
  for vv in itertools.product((0,1,'x','z'),repeat=len(leaves)):values.append(lut(pa[n]['LUTMASK'][1],ts,dict(zip(leaves,vv))))
  return tuple(leaves),tuple(values)
 # Only actual retained source nodes can stand in for a removed duplicate.
 kept_by_leaves={};kept={}
 for n in sorted(set(a)&set(b)):
  if a[n][0]=='EFX_LUT4':
   leaves=tuple(sorted({t for t in ins(n) if t[0]=='net'},key=str));kept_by_leaves.setdefault(leaves,[]).append(n)
 for n in sorted(missing):
  ts=ins(n);sig=signature(n);dst=a[n][1]['O'][0];leaves,values=sig
  if sig not in kept:
   for candidate in kept_by_leaves.get(leaves,[]):
    cs=signature(candidate);kept.setdefault(cs,candidate)
    if cs==sig:break
  if len(set(values))==1 and values[0] in (0,1):
   src=('literal',"1'b"+str(values[0]),None);reason='constant_for_all_four_state_inputs';cases=len(values)
  elif sig in kept:
   other=kept[sig];src=a[other][1]['O'][0];reason='same_complete_four_state_function_as_retained_'+other;cases=len(values)
  elif len(leaves)==1:
   src=leaves[0];drivers=[k for k,(kind,pins) in a.items() if any(src in z for p,z in pins.items() if p in OUT[kind])]
   assert len(drivers)==1 and a[drivers[0]][0]=='EFX_LUT4',(n,'identity input can be Z',src)
   assert all(lut(pa[n]['LUTMASK'][1],ts,{src:v})==v for v in (0,1,'x')),(n,ts,pa[n])
   reason='identity_of_LUT_output_which_never_drives_Z';cases=3
  else:raise AssertionError(('unproved_mapper_removal',n,sig))
  assert src!=dst;alias[dst]=src;rows.append(dict(cell=n,input=src,output=dst,reason=reason,four_state_cases=cases))
 for n in missing:a.pop(n)
 for kind,pins in a.values():
  for pin,ts in pins.items():
   if pin not in OUT[kind]:pins[pin]=[root(t) for t in ts]
 return dict(cells=sorted(missing),alias_pairs=[(k,root(v)) for k,v in alias.items()],all_aliases_four_state_proved=True,rows=rows,upstream_never_z=True)
