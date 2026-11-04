"""Joint exact combinational remapping and geographically grouped drivers.

No protocol, pipeline latency, reset, clock, pin or encrypted PHY change.
The reference is the fully mapped capture-prefetch board, never old RTL.
"""
import argparse,collections,functools,json,re,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import CELL,graph,raw_cells,render_token,rewrite_expression
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from fixed_control_ddr_reuse_locations import locations

PARENT=ROOT/'tmp/frame-execution/connected-board-camera-prefetch1'
PHYSICAL=ROOT/'tmp/frame-execution/camera-prefetch-bound1'
OUT={'EFX_LUT4':['O'],'EFX_ADD':['O','CO'],'EFX_FF':['Q'],'EFX_SRL8':['Q'],
     'EFX_DPRAM10':['RDATA'],'EFX_RAM10':['RDATA'],'EFX_DSP48':['O'],'EFX_DSP24':['O']}
P='video_processing_pipeline/pipeline/cnn/core/dut/loader/'
ZERO=('literal',"1'b0",None)

def used_inputs(kind,p,par):
 if kind=='EFX_LUT4':
  mask=par['LUTMASK'][1]
  return {q:ts for q,ts in p.items() if q.startswith('I') and any(((mask>>x)^(mask>>(x^(1<<int(q[1:])))))&1 for x in range(16))}
 return {q:ts for q,ts in p.items() if q not in OUT.get(kind,[])}

def lut_value(mask,ins,allbits):
 out=0
 for m in range(16):
  if mask>>m&1:
   term=allbits
   for i,v in enumerate(ins):term &= v if m>>i&1 else allbits^v
   out |= term
 return out

def divider_plan(g,ps,loads):
 roots=[('net',P.removesuffix('loader/')+'load_offset',i+3) for i in range(3)]+[('net',P+'first_word',i) for i in range(3,13)]
 known=set(roots);pure=set()
 while True:
  prev=len(pure)
  for n,(k,p) in g.items():
   if k not in ('EFX_LUT4','EFX_ADD') or n in pure:continue
   ins=[t for ts in used_inputs(k,p,ps[n]).values() for t in ts]
   if any(t in known for t in ins) and all(t[0]=='literal' or t in known for t in ins):
    assert not any(t in roots for q,ts in p.items() if q in OUT[k] for t in ts)
    pure.add(n);known.update(t for q,ts in p.items() if q in OUT[k] for t in ts)
  if len(pure)==prev:break
 assert len(pure)==50 and all(g[n][0]=='EFX_LUT4' for n in pure)
 boundary={t for n in pure for q,ts in g[n][1].items() if q in OUT[g[n][0]] for t in ts if any(s not in pure for s,p in loads[t])}
 assert len(boundary)==12
 allbits=(1<<8192)-1;values={t:sum(1<<x for x in range(8192) if x>>i&1) for i,t in enumerate(roots)}
 pending=set(pure);depth={t:0 for t in roots}
 while pending:
  progress=False
  for n in sorted(pending):
   k,p=g[n];ins=[p.get(f'I{i}',[ZERO])[0] for i in range(4)]
   if any(t[0]!='literal' and t not in values for t in ins):continue
   v=[allbits if t[0]=='literal' and t[1].endswith('1') else 0 if t[0]=='literal' else values[t] for t in ins]
   output=p['O'][0];values[output]=lut_value(ps[n]['LUTMASK'][1],v,allbits)
   depth[output]=1+max([depth.get(t,0) for ts in used_inputs(k,p,ps[n]).values() for t in ts] or [0])
   pending.remove(n);progress=True
  assert progress
 return roots,pure,{t:values[t] for t in boundary},max(depth[t] for t in boundary)

def transform(text,placement):
 started=time.monotonic()
 g=graph(text);raw=raw_cells(text);ps=params(text);xy=locations(placement)
 print('PARSED',len(g),round(time.monotonic()-started,2),flush=True)
 loads=collections.defaultdict(list);driver={}
 for n,(k,p) in g.items():
  for pin,ts in p.items():
   for t in set(ts):
    if pin in OUT.get(k,[]):driver[t]=n
    else:loads[t].append((n,pin))
 roots,removed,functions,old_depth=divider_plan(g,ps,loads)
 print('DIVIDER_CONE',len(removed),old_depth,round(time.monotonic()-started,2),flush=True)
 added={};pars={};depths={t:0 for t in roots};truth_cache={};memo={};constants={0:ZERO,1:('literal',"1'b1",None)}
 def make_lut(ins,mask):
  # Canonicalize constant and repeated pins through a complete small truth table.
  unique=list(dict.fromkeys(t for t in ins if t[0]!='literal'))
  tab=0
  for x in range(1<<len(unique)):
   idx=sum((int(t[1].endswith('1')) if t[0]=='literal' else ((x>>unique.index(t))&1))<<i for i,t in enumerate(ins))
   tab |= ((mask>>idx)&1)<<x
  for i in reversed(range(len(unique))):
   if all(((tab>>x)^(tab>>(x^(1<<i))))&1==0 for x in range(1<<len(unique))):
    tab=sum(((tab>>((x&((1<<i)-1))|((x>>i)<<(i+1))))&1)<<x for x in range(1<<(len(unique)-1)))
    unique.pop(i)
  if not unique:return constants[tab&1]
  if len(unique)==1 and tab==2:return unique[0]
  key=(tuple(unique),tab)
  if key in truth_cache:return truth_cache[key]
  nm=f'JOINT_DIV_{len(added):03d}';tok=('net',f'joint_div_{len(added):03d}',None)
  full=sum(((tab>>(x&((1<<len(unique))-1)))&1)<<x for x in range(16))
  pins={f'I{i}':[unique[i] if i<len(unique) else ZERO] for i in range(4)};pins['O']=[tok]
  added[nm]=('EFX_LUT4',pins);pars[nm]={'LUTMASK':(16,full)};depths[tok]=1+max(depths.get(t,0) for t in unique)
  truth_cache[key]=tok;return tok
 def table(vars,tab):
  key=(tuple(vars),tab)
  if key in memo:return memo[key]
  full=(1<<(1<<len(vars)))-1
  if tab in (0,full):ans=constants[int(tab!=0)]
  elif len(vars)<=4:
   mask=sum(((tab>>(i&((1<<len(vars))-1)))&1)<<i for i in range(16))
   ans=make_lut(list(vars)+[ZERO]*(4-len(vars)),mask)
  else:
   half=1<<(len(vars)-1);low=tab&((1<<half)-1);high=tab>>half
   if low==high:ans=table(vars[:-1],low)
   else:
    a,b=table(vars[:-1],low),table(vars[:-1],high)
    ans=make_lut([a,b,vars[-1],ZERO],0xcaca)
  memo[key]=ans;return ans
 def multi(vars,tabs):
  if len(vars)<=7:return [table(vars,t) for t in tabs]
  # Functional decomposition: retain the complete cofactor, including all
  # otherwise unreachable inputs, rather than assuming ideal arithmetic.
  width=4;bits=1<<width;hi_count=1<<(len(vars)-width);mask=(1<<bits)-1
  cof=[[t>>(h*bits)&mask for h in range(hi_count)] for t in tabs]
  static=[i for i,c in enumerate(cof) if all(v in (0,mask) for v in c)]
  dynamic=[i for i in range(len(tabs)) if i not in static]
  sigs=[tuple(cof[i][h] for i in dynamic) for h in range(hi_count)]
  classes=list(dict.fromkeys(sigs));nb=(len(classes)-1).bit_length()
  if nb>3:return [table(vars,t) for t in tabs]
  ids={s:i for i,s in enumerate(classes)}
  hi_tabs=[sum((cof[i][h]==mask)<<h for h in range(hi_count)) for i in static]
  hi_tabs += [sum(((ids[sigs[h]]>>b)&1)<<h for h in range(hi_count)) for b in range(nb)]
  upper=multi(vars[width:],hi_tabs);answer=[None]*len(tabs)
  for i,t in zip(static,upper):answer[i]=t
  for j,i in enumerate(dynamic):
   tt=sum((classes[c][j] if c<len(classes) else 0)<<(c*bits) for c in range(1<<nb))
   answer[i]=table(vars[:width]+upper[len(static):],tt)
  return answer
 ordered=sorted(functions.items(),key=lambda kv:str(kv[0]))
 div_outputs=dict(zip([t for t,fn in ordered],multi(roots,[fn for t,fn in ordered])))
 new_depth=max(depths.get(t,0) for t in div_outputs.values())
 print('DIVIDER_REMAP',new_depth,len(added),round(time.monotonic()-started,2),flush=True)
 assert new_depth<old_depth and len(added)<=220,(old_depth,new_depth,len(added))
 # Independently evaluate emitted LUTs for every possible 13-bit cut input.
 allbits=(1<<8192)-1;values={t:sum(1<<x for x in range(8192) if x>>i&1) for i,t in enumerate(roots)}
 for n,(k,p) in added.items():
  ins=[p[f'I{i}'][0] for i in range(4)]
  values[p['O'][0]]=lut_value(pars[n]['LUTMASK'][1],[allbits if t[0]=='literal' and t[1].endswith('1') else 0 if t[0]=='literal' else values[t] for t in ins],allbits)
 assert all(values[new]==functions[old] for old,new in div_outputs.items())
 edits=collections.defaultdict(dict)
 for old,new in div_outputs.items():
  for n,pin in loads[old]:
   if n not in removed:edits[n].setdefault(pin,{})[old]=new
 # Group long, broad data/control nets; never touch clock/reset/CDC/protected endpoints.
 clones=[];alias={};clone_orig={};original_clone_count=len(added)
 def box(group):
  pp=[xy[n][:2] for n,p in group];return [min(p[d] for p in pp) for d in (0,1)]+[max(p[d] for p in pp) for d in (0,1)]
 def split(group,limit=32,span=70):
  b=box(group)
  if len(group)<=limit and max(b[2]-b[0],b[3]-b[1])<=span:return [group]
  if len(group)<=2:return [group]
  axis=0 if b[2]-b[0]>=b[3]-b[1] else 1
  s=sorted(group,key=lambda a:(xy[a[0]][axis],a));mid=len(s)//2
  return split(s[:mid],limit,span)+split(s[mid:],limit,span)
 # Both reported route variants contribute, including SRL data and ADD carry output.
 hot=set()
 for idx in (1,2):
  report=(ROOT/f'tmp/frame-execution/camera-global-survey{idx}b/outflow/canonical_setup.rpt').read_text()
  for line in report.splitlines():
   m=re.match(r'\s*net \(fo=(\d+)\)\s+([\d.]+)\s+[\d.]+\s+(\S+)',line)
   if m and float(m[2])>=0.65:hot.add(m[3])
 def label(t):return t[1]+(f'[{t[2]}]' if t[2] is not None else '')
 candidates=[]
 for tok,n in driver.items():
  kind,p=g[n];all_loads=loads[tok]
  if n in removed or n not in xy or kind not in ('EFX_LUT4','EFX_FF','EFX_SRL8','EFX_ADD'):continue
  if n.startswith('protected_inst/'):continue
  if kind=='EFX_ADD' and p.get('O')!=[tok]:continue
  if kind in ('EFX_FF','EFX_SRL8'):
   if p.get('CLK')!=[('net','process_clk_150',None)]:continue
   if kind=='EFX_FF' and ps[n].get('SR_SYNC')!=(1,1):continue
  ls=[(s,pin) for s,pin in all_loads if s not in removed and s in xy and not s.startswith('protected_inst/') and not re.search(r'CLK(?!E)|RESET|RST|^SR$',pin)]
  if len(ls)<3:continue
  # Retain original reset/clock/protected loads on the original driver. The
  # critical ordinary branches get separate identical drivers, not new resets.
  if label(tok) not in hot and (tok[1].startswith('cpu/') or sum(s.startswith('cpu/') for s,pin in ls)>len(ls)//2):continue
  b=box(ls);extent=max(b[2]-b[0],b[3]-b[1])
  distance=max(abs(xy[s][0]-xy[n][0])+abs(xy[s][1]-xy[n][1]) for s,pin in ls)
  if not ((len(ls)>=128 and extent>=70) or (label(tok) in hot and (extent>=45 or distance>=70))):continue
  groups=split(ls,24 if kind=='EFX_LUT4' else 16,55 if kind=='EFX_SRL8' else 70)
  fixed=[x for x in all_loads if x not in ls]
  if len(groups)>1 or fixed:candidates.append((n,tok,groups,fixed))
 # Prevent exploding input fanout: only a bounded number of clone cells globally.
 print('DRIVER_CANDIDATES',len(candidates),round(time.monotonic()-started,2),flush=True)
 for n,tok,groups,fixed in sorted(candidates,key=lambda r:(-len(loads[r[1]]),r[0])):
  groups.sort(key=lambda a:sum(abs(xy[s][0]-xy[n][0])+abs(xy[s][1]-xy[n][1]) for s,p in a)/len(a))
  row=dict(driver=n,kind=g[n][0],net=tok,original_loads=fixed if fixed else groups[0],clones=[])
  for group in (groups if fixed else groups[1:]):
   local={}
   def copy(name):
    if name in local:return local[name]
    k,p=g[name];assert k in ('EFX_LUT4','EFX_FF','EFX_SRL8','EFX_ADD')
    np={pin:list(ts) for pin,ts in p.items()}
    # Copy whole dedicated carry prefix, never sever its cascade.
    if k=='EFX_ADD':
     ct=p.get('CI',[ZERO])[0];parent=driver.get(ct)
     if parent and g[parent][0]=='EFX_ADD':
      copy(parent);np['CI']=[local[parent][ct]]
    number=len(clone_orig);new=f'JOINT_LOCAL_{number:04d}';mapping={}
    for pin in OUT[k]:
     for original in p.get(pin,[]):
      replacement=('net',f'joint_local_{number:04d}_{pin.lower()}',None);mapping[original]=replacement;alias[replacement]=original
     if pin in p:np[pin]=[mapping[t] for t in p[pin]]
    added[new]=(k,np);pars[new]=ps[name];clone_orig[new]=name;local[name]=mapping
    return mapping
   mapping=copy(n)
   for s,pin in group:edits[s].setdefault(pin,{})[tok]=mapping[tok]
   row['clones'].append(dict(output=mapping[tok],loads=group,bbox=box(group),copied=list(local)))
  clones.append(row)
 assert 0<len(clone_orig)<=1200,len(clone_orig)
 print('REPLICAS',len(clone_orig),round(time.monotonic()-started,2),flush=True)
 # Preserve whole-chip graph except the proved divider and exact replicas.
 def change(m):
  name=m[2].lstrip('\\')
  if name in removed:return ''
  if name not in edits:return m[0]
  body=re.sub(r'\.(\w+)\(([^()]*)\)',lambda z:'.'+z[1]+'('+rewrite_expression(z[2],edits[name].get(z[1],{}))+')',m[3])
  return m[0][:m.start(3)-m.start()]+body+m[0][m.end(3)-m.start():]
 changed=CELL.sub(change,text)
 changed=re.sub(r'^\s*defparam\s+(\\\S+|\w+)\s*\.\w+\s*=[^;]+;[^\n]*',lambda m:'' if m[1].lstrip('\\') in removed else m[0],changed,flags=re.M)
 lines=[]
 for n,(k,p) in added.items():
  for pin in OUT[k]:
   for t in p.get(pin,[]):
    # Carry links must remain dedicated cascades. Keeping an intermediate
    # CO net forces Efinity to export/reimport it through auxiliary adders.
    keep='' if k=='EFX_ADD' and pin=='CO' else '(* syn_keep="true" *) '
    lines.append('    '+keep+'wire '+render_token(t)+';')
  lines.append('    '+k+' '+n+' ('+', '.join('.'+pin+'('+ (render_token(ts[0]) if len(ts)==1 else '{'+','.join(render_token(t) for t in ts)+'}') +')' for pin,ts in p.items())+') /* proved joint native */ ;')
  lines.extend(f"    defparam {n}.{key}={width}'h{value:x};" for key,(width,value) in pars[n].items())
 anchor=changed.index('    protected_cell protected_inst (');changed=changed[:anchor]+'\n'.join(lines)+'\n'+changed[anchor:]
 ng=graph(changed);nps=params(changed);assert protected_blocks(changed)==protected_blocks(text)
 div_inverse={v:k for k,v in div_outputs.items()};assert len(div_inverse)==len(div_outputs)
 def collapse(ts):return [alias.get(t,div_inverse.get(t,t)) for t in ts]
 for n,(k,p) in g.items():
  if n in removed:continue
  assert ng[n][0]==k and {pin:collapse(ts) for pin,ts in ng[n][1].items()}==p,n
  assert nps[n]==ps[n]
 for n,parent in clone_orig.items():
  assert ng[n][0]==g[parent][0] and {pin:collapse(ts) for pin,ts in ng[n][1].items()}==g[parent][1],(n,parent)
  assert nps[n]==ps[parent]
 assert set(ng)==(set(g)-removed)|set(added)
 evidence=dict(status='JOINT_NATIVE_EXACT_TRANSFORM_PASS',removed_divider_cells=sorted(removed),divider_inputs=roots,
  divider_outputs=[dict(old=k,new=v) for k,v in div_outputs.items()],divider_exhaustive_patterns=8192,
  old_divider_depth=old_depth,new_divider_depth=new_depth,divider_added_cells=original_clone_count,
  replicas=clones,clone_orig=clone_orig,added_kinds=dict(collections.Counter(k for k,p in added.values())),
  removed_kinds={'EFX_LUT4':len(removed)},unchanged_retained_cells=len(g)-len(removed),
  cycles_added=0,protected_blocks_byte_equal=True,hardware_access=False,board_release=False)
 return changed,evidence

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 src=PARENT/'outflow/ti60f225_oob.map.v';place=PHYSICAL/'outflow/ti60f225_oob.out.placeloc'
 changed,r=transform(src.read_text(),place);(root/'native_source.v').write_text(changed,encoding='utf-8')
 r.update(source_sha256=sha(root/'native_source.v'),inputs={str(f):sha(f) for f in (src,place,Path(__file__))})
 for i in (1,2):
  f=ROOT/f'tmp/frame-execution/camera-global-survey{i}b/outflow/canonical_setup.rpt';r['inputs'][str(f)]=sha(f)
 (root/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in r.items() if k not in ('replicas','clone_orig','removed_divider_cells','divider_inputs','divider_outputs','inputs')}),flush=True)
if __name__=='__main__':main()
