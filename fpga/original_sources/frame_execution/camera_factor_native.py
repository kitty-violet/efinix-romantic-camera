"""Exact late-control factorization; no sequential replication or new cycles.

Keeps the exhaustively proved loader cut remap. Shannon factorization moves
the late error/address decisions to the last LUT, reusing unchanged cones.
Only selected CE loads change. All resets/clocks/data/state/PHY stay intact.
"""
import argparse,collections,functools,json,re,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import CELL,graph,render_token,rewrite_expression
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from camera_joint_native import PARENT,PHYSICAL,OUT,ZERO
BASE=HERE/'camera_joint_native.py'
P='video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/'

def divider_only(text,placement):
 code=BASE.read_text()
 old='for n,tok,groups,fixed in sorted(candidates,key=lambda r:(-len(loads[r[1]]),r[0])):'
 assert code.count(old)==1
 code=code.replace(old,'for n,tok,groups,fixed in []:')
 code=code.replace('assert 0<len(clone_orig)<=1200,len(clone_orig)','assert len(clone_orig)==0')
 ns={'__name__':'divider_only','__file__':str(BASE)}
 exec(compile(code,str(BASE)+':no-replicas','exec'),ns)
 return ns['transform'](text,placement)

def normalize(xs,mask):
 unique=list(dict.fromkeys(x for x in xs if not isinstance(x,int)))
 vals=[]
 for bits in range(1<<len(unique)):
  i=sum((x if isinstance(x,int) else ((bits>>unique.index(x))&1))<<j for j,x in enumerate(xs))
  vals.append((mask>>i)&1)
 for i in reversed(range(len(unique))):
  if all(vals[j]==vals[j|(1<<i)] for j in range(len(vals)) if not j>>i&1):
   vals=[v for j,v in enumerate(vals) if not j>>i&1];unique.pop(i)
 if not unique:return vals[0]
 if len(unique)==1 and vals==[0,1]:return unique[0]
 return ('lut',tuple(unique),sum(v<<i for i,v in enumerate(vals)))

class Cone:
 def __init__(self,text,root,stops=()):
  self.g=graph(text);self.ps=params(text);self.root=root
  self.lut={pins['O'][0]:(n,pins) for n,(k,pins) in self.g.items() if k=='EFX_LUT4'};self.stops=set(stops)
 @functools.lru_cache(None)
 def expr(self,t,value=None):
  if t==self.root and value is not None:return value
  if t[0]=='literal':
   assert t[1] in ("1'b0","1'b1");return int(t[1][-1])
  if t in self.stops or t not in self.lut:return ('leaf',t)
  n,pins=self.lut[t]
  return normalize([self.expr(pins.get('I'+str(i),[ZERO])[0],value) for i in range(4)],self.ps[n]['LUTMASK'][1])
 @functools.lru_cache(None)
 def root_depth(self,t):
  if t==self.root:return 0
  if t not in self.lut:return -10000
  n,pins=self.lut[t];mask=self.ps[n]['LUTMASK'][1]
  ins=[pins.get('I'+str(i),[ZERO])[0] for i in range(4) if any(((mask>>v)^(mask>>(v^(1<<i))))&1 for v in range(16))]
  return 1+max([self.root_depth(t) for t in ins] or [-10000])

def factor(text,root,scope,prefix):
 c=Cone(text,root);g=c.g;ps=c.ps
 c.stops={t for t in c.lut if c.root_depth(t)<0}
 targets={}
 for n,(kind,pins) in g.items():
  if kind not in ('EFX_FF','EFX_SRL8','EFX_DSP24','EFX_DSP48') or not n.startswith(scope):continue
  if 'CE' not in pins:continue
  t=pins['CE'][0]
  if c.root_depth(t)<=1 or c.expr(t,0)==c.expr(t,1):continue
  targets.setdefault(t,[]).append((n,'CE'))
 assert targets
 # Reuse an existing expression only when its root-independent structure is
 # exactly identical; avoid preserving a hidden late-root path through a gate.
 oldexpr={('leaf',t):t for t in c.stops}
 added={};pars={};built={};edits=collections.defaultdict(dict);rows=[]
 def emit(expr):
  if isinstance(expr,int):return ('literal',"1'b"+str(expr),None)
  if expr[0]=='leaf':return expr[1]
  if expr in oldexpr:return oldexpr[expr]
  if expr in built:return built[expr]
  xs=[emit(x) for x in expr[1]];v=expr[2];number=len(added)
  name=prefix+f'_{number:03d}';tok=('net',name.lower(),None)
  mask=sum(((v>>(x&((1<<len(xs))-1)))&1)<<x for x in range(16))
  pins={'I'+str(i):[xs[i] if i<len(xs) else ZERO] for i in range(4)};pins['O']=[tok]
  added[name]=('EFX_LUT4',pins);pars[name]={'LUTMASK':(16,mask)};built[expr]=tok
  return tok
 for t,loads in sorted(targets.items(),key=lambda x:str(x[0])):
  a,b=c.expr(t,0),c.expr(t,1)
  low,high=emit(a),emit(b)
  def leaf(t):return int(t[1][-1]) if t[0]=='literal' else ('leaf',t)
  out=emit(normalize([leaf(low),leaf(high),('leaf',root),0],0xcaca))
  for n,pin in loads:edits[n][pin]={t:out}
  rows.append(dict(old=t,new=out,loads=loads,old_late_levels=c.root_depth(t),low_constant=a if isinstance(a,int) else None,high_constant=b if isinstance(b,int) else None))
 def change(m):
  n=m[2].lstrip('\\')
  if n not in edits:return m[0]
  body=re.sub(r'\.(\w+)\(([^()]*)\)',lambda z:'.'+z[1]+'('+rewrite_expression(z[2],edits[n].get(z[1],{}))+')',m[3])
  return m[0][:m.start(3)-m.start()]+body+m[0][m.end(3)-m.start():]
 changed=CELL.sub(change,text);lines=[]
 for n,(k,pins) in added.items():
  lines.append('    (* syn_keep="true" *) wire '+render_token(pins['O'][0])+';')
  lines.append('    EFX_LUT4 '+n+' ('+', '.join('.'+pin+'('+render_token(ts[0])+')' for pin,ts in pins.items())+') /* exact late-control cofactor */ ;')
  lines.append(f"    defparam {n}.LUTMASK=16'h{pars[n]['LUTMASK'][1]:04x};")
 anchor=changed.index('    protected_cell protected_inst (');changed=changed[:anchor]+'\n'.join(lines)+'\n'+changed[anchor:]
 ng=graph(changed);np=params(changed)
 # Distinct old CE nets may have identical cofactors. Their new common
 # function is checked separately for every old net below.
 for n,(k,pins) in g.items():
  assert ng[n][0]==k and np[n]==ps[n]
  expected={pin:[edits.get(n,{}).get(pin,{}).get(t,t) for t in ts] for pin,ts in pins.items()}
  assert ng[n][1]==expected,n
 assert set(ng)-set(g)==set(added) and set(g)<=set(ng)
 assert protected_blocks(text)==protected_blocks(changed)
 # Independent reconstruction from emitted cell graph, then compare both
 # cofactors. Shannon identity proves every binary leaf combination, not a
 # sampled or reachable-only subset. Sequential boundaries are unchanged.
 nc=Cone(changed,root,c.stops)
 for r in rows:
  for v in (0,1):assert c.expr(tuple(r['old']),v)==nc.expr(tuple(r['new']),v),r
  assert nc.root_depth(tuple(r['new']))==1,r
 return changed,dict(root=root,scope=scope,outputs=rows,cut_boundaries=sorted(c.stops,key=str),added_cells=sorted(added),added_luts=len(added),shannon_cases=2*len(rows),unchanged_retained_cells=len(g),max_old_late_levels=max(r['old_late_levels'] for r in rows),new_late_levels=1,proof='Complete Boolean cofactors reconstructed from emitted LUT graph; independent original cones are shared boundaries. No state or latency changes. Four-state simulation separate.')

def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True,type=Path);args=a.parse_args();out=args.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir()
 src=PARENT/'outflow/ti60f225_oob.map.v';place=PHYSICAL/'outflow/ti60f225_oob.out.placeloc'
 text,proof=divider_only(src.read_text(),place)
 assert not proof['clone_orig']
 stages=[]
 for idx,(root,scope,prefix) in enumerate([
  (('net',P+'werror',None),P,'FACTOR_ERROR'),
  (('net','video_processing_pipeline/pipeline/cnn/core/aa',37),'video_processing_pipeline/pipeline/cnn/core/dut/engine/','FACTOR_ADDRESS')]):
  (out/f'stage{idx}_before.v').write_text(text)
  text,r=factor(text,root,scope,prefix);(out/f'stage{idx}_after.v').write_text(text);stages.append(r)
 (out/'native_source.v').write_text(text,encoding='utf8')
 proof.update(status='FACTOR_NATIVE_EXACT_TRANSFORM_PASS',factor_stages=stages,clock_loads_added=0,reset_loads_added=0)
 proof['added_kinds']['EFX_LUT4']+=sum(r['added_luts'] for r in stages)
 proof.update(source_sha256=sha(out/'native_source.v'),inputs={str(f):sha(f) for f in (src,place,BASE,Path(__file__))})
 (out/'result.json').write_text(json.dumps(proof,indent=2))
 print(json.dumps({k:v for k,v in proof.items() if k not in ('inputs','factor_stages','divider_inputs','divider_outputs','removed_divider_cells','replicas','clone_orig')}))
 for r in stages:print(r['scope'],r['added_luts'],len(r['outputs']),r['max_old_late_levels'],r['new_late_levels'])
if __name__=='__main__':main()
