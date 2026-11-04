"""Demand-driven S4R1 row schedule and exact six-bank live allocation.

This is an executable address/lifetime proof, not a hardware FPS estimate.
Stride4 requests its actual three source rows; unused stem rows are not
computed. Nearest4 is an address view and has no intermediate DDR tensor.
"""
from pathlib import Path
from collections import defaultdict,Counter
import copy,json,time,sys
from coefficients import R,INPUTS,oracle,sha

def grouped(ys,limit):
 chunk=[]
 for y in sorted(ys):
  if chunk and (y!=chunk[-1]+1 or len(chunk)==limit):yield chunk;chunk=[]
  chunk.append(y)
 if chunk:yield chunk

def tile_graph(graph,group,left,right):
 shapes={'input':(720,1280,3)};nodes={}
 for n0 in graph['nodes']:
  n=dict(n0);h,w,c=shapes[n['inputs'][0]]
  if n['kind']=='nearest':h*=n['factor'];w*=n['factor']
  else:s=n.get('stride',1);h=(h+s-1)//s;w=(w+s-1)//s
  n['shape']=(h,w,n['channels']);shapes[n['id']]=n['shape'];nodes[n['id']]=n
 names=({'encoder':['stem','down_dw','down_pw'],
  'residual':[n for n in nodes if n.startswith(('res1_','res2_'))],
  'decoder':['up1_low_dw','up1_low_pw','up1_nearest','up1_high','rgb']})[group]
 source=nodes[names[0]]['inputs'][0]
 local={'source':dict(id='source',kind='source',inputs=[],shape=shapes[source])}
 for name in names:
  n=copy.deepcopy(nodes[name]);n['inputs']=[p if p in local else 'source' for p in n['inputs']]
  n['parameter_node']=name
  if n['kind']=='add':
   pw=local.pop(n['inputs'][0]);n.update(kind='pw_fused',parameter_node=pw['id'],cin=pw['cin'],kernel=1,stride=1,
    activation='none',inputs=[pw['inputs'][0],n['inputs'][1]])
  local[name]=n
 final=names[-1];ranges={final:(left,right)}
 for name,n in reversed(list(local.items())):
  if name=='source':continue
  a,b=ranges[name];p=n['inputs'][0];width=local[p]['shape'][1]
  if n['kind']=='nearest':a//=4;b//=4
  elif n['kind'] in ('conv3x3','dw3x3'):
   a=max(0,a*n['stride']-1);b=min(width-1,b*n['stride']+1)
  for parent in n['inputs']:
   lo,hi=ranges.get(parent,(a,b));ranges[parent]=(min(lo,a),max(hi,b))
 assert set(ranges)==set(local)
 # The retained RGB DMA accepts four-byte-aligned byte addresses. RGB24
 # therefore starts at a multiple of four pixels. Keep two extra halo pixels
 # where necessary; the compute source origin selects exactly the same tensor.
 if local['source']['shape'][2]==3:
  lo,hi=ranges['source'];ranges['source']=(lo//4*4,hi)
 return local,ranges,final

def schedule(local,final):
 done={n:set()for n in local};tasks=[]
 def physical(n,y):
  h=local[n]['shape'][0];y=max(0,min(h-1,y))
  return physical(local[n]['inputs'][0],y//4) if local[n]['kind']=='nearest' else (n,y)
 def ensure(name,ys):
  node=local[name];ys={max(0,min(node['shape'][0]-1,y))for y in ys}-done[name]
  if not ys:return
  if node['kind']=='nearest':
   ensure(node['inputs'][0],{y//4 for y in ys});done[name].update(ys);return
  quantum=1 if node['kind']=='source' or node.get('stride',1)==4 else 3
  for chunk in grouped(ys,quantum):
   deps=[]
   for pi,parent in enumerate(node['inputs']):
    if pi==0 and node['kind']in ('conv3x3','dw3x3'):py={y*node['stride']+dy for y in chunk for dy in (-1,0,1)}
    else:py=set(chunk)
    py={max(0,min(local[parent]['shape'][0]-1,y))for y in py}
    ensure(parent,py);deps.extend(physical(parent,y)for y in py)
   tasks.append(dict(node=name,ys=chunk,dependencies=sorted(set(deps)),outputs=[(name,y)for y in chunk]))
   done[name].update(chunk)
 h=local[final]['shape'][0]
 for ys in grouped(range(h),3):
  ensure(final,set(ys))
  for y in ys:tasks.append(dict(node='commit',ys=[y],dependencies=[(final,y)],outputs=[]))
 return tasks,physical,done

def allocate(local,ranges,tasks,offsets=None):
 records={}
 if offsets is None:offsets={n:i%6 for i,n in enumerate(local)if local[n]['kind']!='nearest'}
 for ti,task in enumerate(tasks):
  for n,y in task['outputs']:
   lo,hi=ranges[n];g=(local[n]['shape'][2]+7)//8
   phase=lo%2 if n=='source' and g==4 else 0
   records[n,y]=dict(node=n,y=y,origin=lo,width=hi-lo+1,groups=g,phase=phase,
    bank=(y+offsets[n])%6,length=(hi-lo+1+phase+1)//2*g,born=ti,last=ti)
  for key in task['dependencies']:
   assert key in records;records[key]['last']=ti
 high=[0]*6;peak=[0]*6
 for bank in range(6):
  active=[]
  for key,r in sorted(records.items(),key=lambda kv:kv[1]['born']):
   if r['bank']!=bank:continue
   active=[q for q in active if q['last']>=r['born']]
   gaps=[];at=0
   for q in sorted(active,key=lambda q:q['base'])+[dict(base=1024,length=0)]:
    if q['base']-at>=r['length']:gaps.append((q['base']-at,at))
    at=max(at,q['base']+q['length'])
   if not gaps:raise MemoryError(dict(bank=bank,row=key,length=r['length'],active=sum(q['length']for q in active)))
   r['base']=min(gaps)[1];active.append(r)
   high[bank]=max(high[bank],r['base']+r['length']);peak[bank]=max(peak[bank],sum(q['length']for q in active))
 return records,dict(high=high,peak_live=peak,offsets=offsets)

def compile_tile(graph,group,left,right,tile_id):
 local,ranges,final=tile_graph(graph,group,left,right);tasks,physical,done=schedule(local,final)
 candidates=[None]
 if group=='residual':
  for pattern in [(2,0,1,0),(3,0,2,0),(1,3,5,0),(2,4,0,0)]:
   offsets={'source':0}
   for i,n in enumerate(list(local)[1:]):offsets[n]=pattern[i%4]
   candidates.append(offsets)
 failures=[]
 for offsets in candidates:
  try:records,allocation=allocate(local,ranges,tasks,offsets);break
  except MemoryError as exc:failures.append(str(exc))
 else:raise MemoryError(dict(group=group,left=left,attempts=failures))
 allocation['rejected_offset_attempts']=failures;commands=[]
 for task in tasks:
  name=task['node']
  if name in ('source','commit'):
   for key in task['outputs']if name=='source'else task['dependencies']:
    row=dict(records[tuple(key)]);commands.append(dict(kind='load'if name=='source'else'store',row=row,channels=local[key[0]]['shape'][2],global_width=local[key[0]]['shape'][1],tensor=key[0]))
   continue
  node=local[name];stride=node.get('stride',1);lo,hi=ranges[name];max_width=128 if stride==4 else 320
  parent=node['inputs'][0];nn=local[parent]['kind']=='nearest';pw=node['kind']in ('pw1x1','pw_fused')
  for x in range(lo,hi+1,max_width):
   count=min(max_width,hi-x+1);y=task['ys'][0];rows=len(task['ys'])
   py=list(range(y,y+rows))if pw else list(range(y*stride-1,(y+rows-1)*stride+2))
   inputs=[dict(records[physical(parent,yy)])for yy in py];assert len(inputs)<=6
   outs=[]
   for yy in task['ys']:
    d=dict(records[name,yy]);delta=x-d['origin'];d['base']+=((delta+d['phase'])//2)*d['groups'];d['phase']=(delta+d['phase'])%2;d['origin']=x;d['width']=count;outs.append(d)
   skips=[dict(records[physical(node['inputs'][1],yy)])for yy in task['ys']]if node['kind']=='pw_fused'else[]
   for d in skips:d['delta']=x-d['origin']
   assert all(d['delta'] in (0,2)for d in skips),(group,name,[d['delta']for d in skips])
   aliases={}
   for d in inputs:
    key=d['node'],d['y'],d['base'];assert aliases.get(d['bank'],key)==key;aliases[d['bank']]=key
   commands.append(dict(kind='compute',node=name,parameter_node=node['parameter_node'],operator=node['kind'],pixels=count,rows=rows,stride=stride,nn_factor=4 if nn else 1,
    output_x=x*stride,source_width=local[parent]['shape'][1],inputs=inputs,outputs=outs,skip=skips,input_groups=(node['cin']+7)//8,output_groups=(node['channels']+7)//8,mask=(1<<min(8,node['channels']))-1))
 for c in commands:c['tile']=tile_id;c['group']=group;c['final']=final
 return commands,dict(tile=tile_id,group=group,left=left,right=right,allocation=allocation,
  computed_rows={n:len(ys)for n,ys in done.items()},records=list(records.values()))

def units(d,start,count):
 assert 0<=start and start+count<=d['width'],(d,start,count)
 return [(d['bank']*2048+2*(d['base']+((x+d['phase'])//2)*d['groups']+g)+(x+d['phase'])%2,
          (d['node'],d['y'],d['origin']+x,g))for x in range(start,start+count)for g in range(d['groups'])]

def accesses(c):
 reads=[];writes=[];padding=set()
 if c['kind']!='compute':
  d=c['row'];(writes if c['kind']=='load'else reads).extend(units(d,0,d['width']))
  if c['kind']=='load'and d['groups']==4:padding=set(range(d['bank']*2048+2*d['base'],d['bank']*2048+2*(d['base']+d['length'])))
 else:
  pw=c['operator']in ('pw1x1','pw_fused');lo=max(0,c['output_x']-(0 if pw else 1));hi=min(c['source_width']-1,c['output_x']+(c['pixels']-1)*c['stride']+(0 if pw else 1))
  lo//=c['nn_factor'];hi//=c['nn_factor']
  for d in c['inputs']:reads.extend(units(d,lo-d['origin'],hi-lo+1))
  for d in c['skip']:reads.extend(units(d,d['delta'],c['pixels']))
  for d in c['outputs']:writes.extend(units(d,0,c['pixels']))
 assert all(0<=u<12288 for u,_ in reads+writes)
 return reads,writes,padding

def main():
 start=time.monotonic();root=R/'data'/('schedule'+(sys.argv[1]if len(sys.argv)>1 else'1'));root.mkdir(parents=True,exist_ok=False)
 graph,_=oracle.load(Path(INPUTS['selected'][0]['folder']));cmds=[];tiles=[]
 widths=(int(sys.argv[2]),int(sys.argv[3]))if len(sys.argv)>3 else(64,80)
 for group,width,total in [('encoder',widths[0],320),('residual',widths[1],320),('decoder',256,1280)]:
  for left in range(0,total,width):
   cc,meta=compile_tile(graph,group,left,min(total-1,left+width-1),len(tiles));cmds.extend(cc);tiles.append(meta)
 owner=[None]*12288;reads_count=writes_count=0
 for i,c in enumerate(cmds):
  c['command']=i;reads,writes,padding=accesses(c)
  for u,identity in reads:
   assert owner[u]==(c['tile'],*identity),('STALE_TENSOR',i,u,owner[u],identity);reads_count+=1
  for u in padding:owner[u]=('padding',i)
  for u,identity in writes:owner[u]=(c['tile'],*identity);writes_count+=1
 program=json.dumps(cmds,separators=(',',':')).encode();(root/'program.json').write_bytes(program)
 (root/'allocations.json').write_text(json.dumps(tiles,separators=(',',':')),'utf8')
 result=dict(status='S4R1_STATIC_ROW_LIFETIME_AND_ADDRESS_PASS_NOT_RTL_RELEASE',commands=dict(Counter(c['kind']for c in cmds)),total_commands=len(cmds),
  tile_count=len(tiles),max_bank_words=max(max(t['allocation']['high'])for t in tiles),
  semantic_reads=reads_count,semantic_writes=writes_count,program_sha256=sha(program),seconds=time.monotonic()-start,
  compute_graph_nodes=len(graph['nodes']),tile_widths=dict(encoder=widths[0],residual=widths[1],decoder=256),rtl_executed=False,throughput_claim=False,
  encoder_stem_rows_per_tile=[t['computed_rows']['stem']for t in tiles if t['group']=='encoder'])
 (root/'result.json').write_text(json.dumps(result,indent=2),'utf8');print(json.dumps(result))
if __name__=='__main__':main()
