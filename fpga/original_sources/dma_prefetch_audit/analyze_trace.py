"""Evidence-only analysis of exact descriptor dependencies and observed events.
No hardware access, no writes outside this isolated audit workspace.
"""
from pathlib import Path
from collections import Counter,defaultdict
import ast,json,hashlib,sys
R=Path(__file__).resolve().parent

def load_functions(filename,names):
 tree=ast.parse((R/filename).read_text('utf8'))
 selected=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
 assert len(selected)==len(names)
 ns={};exec(compile(ast.Module(body=selected,type_ignores=[]),filename,'exec'),ns)
 return ns

def static(schedule='schedule',vectors='vectors',outfile='dependency-audit.json'):
 raw=(R/schedule/'program.json').read_bytes();commands=json.loads(raw)
 accesses=load_functions('compile_schedule.py',{'units','accesses'})['accesses']
 latest_w=[None]*12288;latest_r=[[None]*12288 for _ in range(2)]
 streams=[[],[]];counts=[0,0];totals=Counter()
 bins=[(R/vectors/'compute.bin').read_bytes(),(R/vectors/'dma.bin').read_bytes()]
 for c in commands:
  rr,ww,pad=accesses(c);rs={u for u,_ in rr};ws={u for u,_ in ww}|pad
  e=int(c['kind']!='compute');peer=1-e;ordinal=counts[e];deps=defaultdict(Counter);examples={}
  def dep(v,typ,u):
   if v is not None and v[0]==peer:
    deps[v[1]][typ]+=1;examples.setdefault((v[1],typ),dict(unit=u,previous_command=v[2]))
  for u in rs:dep(latest_w[u],'RAW',u)
  for u in ws:
   dep(latest_w[u],'WAW',u)
   dep(latest_r[peer][u],'WAR',u)
  required=max(deps,default=0);size=64 if e==0 else 16
  encoded=int.from_bytes(bins[e][ordinal*size:(ordinal+1)*size],'little')&16383
  assert required==encoded,('DEPENDENCY_MISMATCH',c['command'],required,encoded)
  item=dict(command=c['command'],kind=c['kind'],node=c.get('node',c.get('tensor')),tile=c['tile'],group=c['group'],ordinal=ordinal,
   peer_completed=required,dependencies={str(k):dict(v)for k,v in deps.items()},max_dependency_examples={typ:examples[required,typ]for typ in deps[required]} if required else{})
  if c['kind']=='compute':item.update(pixels=c['pixels'],rows=c['rows'])
  else:item.update(y=c['row']['y'],width=c['row']['width'],bank=c['row']['bank'],base=c['row']['base'])
  streams[e].append(item);counts[e]+=1
  for u in rs:latest_r[e][u]=(e,counts[e],c['command'])
  for u in ws:latest_w[u]=(e,counts[e],c['command'])
  if required:totals[str(e)+':'+','.join(sorted(deps[required]))]+=1
 result=dict(schedule_sha256=hashlib.sha256(raw).hexdigest(),counts=counts,max_dependency_classes=dict(totals),streams=streams)
 (R/outfile).write_text(json.dumps(result,separators=(',',':')),'utf8')
 print(json.dumps({k:v for k,v in result.items()if k!='streams'},indent=2))
 return result

def analyze(trace,depfile='dependency-audit.json'):
 data=json.loads((R/depfile).read_text('utf8'));cs,ds=data['streams']
 events=[]
 for line in(trace/'events.txt').read_text('utf8').splitlines():
  p=line.split();need=6 if p[0]=='WAIT'else 5 if p[0]=='CS'else 4 if p[0]=='DS'else 3
  if len(p)!=need:continue
  events.append((p[0],*[int(x)for x in p[1:]]))
 waits=Counter();by_node=Counter();targets=Counter();worst=[];durations=[{},{}];starts=[{},{}];tile_spans={}
 for row in events:
  kind=row[0]
  if kind=='WAIT':
   _,a,b,ci,di,req=row;n=b-a;c=cs[ci];d=ds[di];target=ds[req-1]
   assert req==c['peer_completed']
   direct=c['dependencies'].get(str(di+1));typ=','.join(sorted(direct))if direct else'prefix_not_direct'
   waits[(d['kind'],typ)]+=n;by_node[c['node']]+=n
   targets[','.join(sorted(c['dependencies'][str(req)]))]+=n
   worst.append(dict(cycles=n,start=a,end=b,compute=c,dma_in_front=d,final_dma_dependency=target,category=typ))
  elif kind in('CS','DS'):
   e=int(kind=='DS');starts[e][row[2]]=row[1]
   c=(cs,ds)[e][row[2]];span=tile_spans.setdefault(c['tile'],dict(group=c['group'],start=row[1],end=row[1]))
   span['start']=min(span['start'],row[1])
  elif kind in('CE','DE'):
   e=int(kind=='DE');idx=row[2]
   assert idx in starts[e]
   durations[e][idx]=row[1]-starts[e][idx]
   tile_spans[(cs,ds)[e][idx]['tile']]['end']=row[1]
 group_cycles=Counter()
 for span in tile_spans.values():group_cycles[span['group']]+=span['end']-span['start']
 summary=dict(complete=(trace/'result.json').exists(),event_count=len(events),completed=[len(x)for x in durations],
  compute_dependency_wait_cycles=sum(waits.values()),wait_current_dma_and_reason={':'.join(k):v for k,v in waits.items()},
  wait_compute_node=dict(by_node),wait_final_dependency_reason=dict(targets),
  max_wait_examples=sorted(worst,key=lambda x:x['cycles'],reverse=True)[:12],
  measured_duration_sums=[sum(x.values())for x in durations],
  observed_tile_spans=tile_spans,group_span_cycles=dict(group_cycles),
  warning='Read AXI buckets from trace1 observe merged fetch/DMA port and are excluded. Prefix_not_direct is not automatically removable: indirect dependencies remain.')
 (trace/'analysis.json').write_text(json.dumps(summary,indent=2),'utf8')
 (trace/'durations.json').write_text(json.dumps(durations,separators=(',',':')),'utf8')
 print(json.dumps({k:v for k,v in summary.items()if k!='max_wait_examples'},indent=2))
 return summary

if __name__=='__main__':
 candidate=len(sys.argv)>1 and sys.argv[1].startswith('candidate')
 depfile='candidate-dependency-audit.json'if candidate else'dependency-audit.json'
 if not(R/depfile).exists()or '--static'in sys.argv:
  static('allocation-candidate','candidate-vectors',depfile)if candidate else static()
 if len(sys.argv)>1 and sys.argv[1]!='--static':analyze(R/sys.argv[1],depfile)
