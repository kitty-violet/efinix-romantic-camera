"""Program-only audit experiment: reserve source rows earlier in existing SRAM.
No extra bank, tensor, weight, arithmetic or RTL. Recheck every SRAM access.
"""
from pathlib import Path
from collections import Counter,defaultdict
import ast,copy,hashlib,heapq,json,time
from analyze_trace import load_functions
R=Path(__file__).resolve().parent
accesses=load_functions('compile_schedule.py',{'units','accesses'})['accesses']
ns=dict(accesses=accesses,heapq=heapq)
tree=ast.parse((R/'reorder_dma.py').read_text('utf8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('graph','cost','run','check')],type_ignores=[]),'reorder_dma.py','exec'),ns)

def reallocate(meta,advance):
 new={};peaks=[]
 for t in meta:
  rows=copy.deepcopy(t['records']);high=[0]*6
  for row in rows:
   row['reserved_from']=max(0,row['born']-(advance.get(t['group'],0)if row['node']=='source'else 0))
  for bank in range(6):
   active=[]
   for row in sorted((x for x in rows if x['bank']==bank),key=lambda x:(x['reserved_from'],x['born'])):
    active=[x for x in active if x['last']>=row['reserved_from']]
    gaps=[];at=0
    for x in sorted(active,key=lambda x:x['base'])+[dict(base=1024,length=0)]:
     if x['base']-at>=row['length']:gaps.append((x['base']-at,at))
     at=max(at,x['base']+x['length'])
    if not gaps:return None,dict(tile=t['tile'],bank=bank,node=row['node'],y=row['y'],length=row['length'])
    row['base']=min(gaps)[1];active.append(row);high[bank]=max(high[bank],row['base']+row['length'])
    new[t['tile'],row['node'],row['y']]=row
  peaks.append(high)
 return new,peaks

def remap(commands,old,new):
 output=copy.deepcopy(commands)
 for c in output:
  for d in([c['row']]if c['kind']!='compute'else c['inputs']+c['outputs']+c['skip']):
   key=(c['tile'],d['node'],d['y']);d['base']+=new[key]['base']-old[key]['base']
 return output

def main():
 st=time.monotonic();current=json.loads((R/'schedule/program.json').read_text('utf8'))
 original=sorted(copy.deepcopy(current),key=lambda c:c['original_command'])
 for c in original:c['command']=c.pop('original_command')
 raw=json.dumps(original,separators=(',',':')).encode()
 assert hashlib.sha256(raw).hexdigest()==json.loads((R/'schedule/result.json').read_text())['parent_schedule_sha256']
 meta=json.loads((R/'schedule/allocations.json').read_text('utf8'));old={(t['tile'],r['node'],r['y']):r for t in meta for r in t['records']}
 duration=json.loads((R/'trace1/durations.json').read_text('utf8'));measured={};idx=[0,0]
 for c in current:
  e=int(c['kind']!='compute');v=duration[e].get(str(idx[e]));idx[e]+=1
  if v is not None:measured[c['original_command']]=v+1
 assert len(measured)==len(current),'Need complete actual trace before candidate ranking'
 ns['cost']=lambda c:measured[c['command']]
 trials=[];best=None
 options=[dict(encoder=0,residual=0,decoder=0),dict(encoder=3,residual=0,decoder=0),dict(encoder=6,residual=0,decoder=0),
  dict(encoder=0,residual=3,decoder=0),dict(encoder=0,residual=6,decoder=0),dict(encoder=0,residual=0,decoder=6),
  dict(encoder=6,residual=3,decoder=6),dict(encoder=12,residual=6,decoder=12)]
 for advance in options:
  new,peaks=reallocate(meta,advance)
  if new is None:
   result=dict(advance=advance,status='REJECT_CAPACITY',detail=peaks);trials.append(result);print(json.dumps(result),flush=True);continue
  cmds=remap(original,old,new);ns['check'](cmds);pred,succ,rank=ns['graph'](cmds)
  order,streams,model=ns['run'](cmds,pred,succ,rank,True)
  out=[]
  for j,i in enumerate(order):
   c=dict(cmds[i]);c['original_command']=i;c['command']=j;out.append(c)
  checks=ns['check'](out)
  rgb=lambda cc:[(c['tile'],c['row']['y'],c['row']['origin'])for c in cc if c['kind']=='store'and c['channels']==3]
  assert rgb(out)==rgb(original)
  result=dict(advance=advance,status='STATIC_PASS_MODEL_ONLY',high=max(map(max,peaks)),semantic_reads=checks,model=model)
  trials.append(result);print(json.dumps(result),flush=True)
  if best is None or model['model_ticks']<best[0]:best=(model['model_ticks'],out,result)
  (R/'allocation-exploration.json').write_text(json.dumps(dict(trials=trials,seconds=time.monotonic()-st),indent=2),'utf8')
 assert best
 dest=R/'allocation-candidate';dest.mkdir(exist_ok=True)
 raw=json.dumps(best[1],separators=(',',':')).encode();(dest/'program.json').write_bytes(raw)
 (dest/'result.json').write_text(json.dumps(dict(**best[2],program_sha256=hashlib.sha256(raw).hexdigest(),hardware_access=False,rtl_pass=False),indent=2),'utf8')
 print('SELECTED',best[0],best[2]['advance'],flush=True)

if __name__=='__main__':main()
