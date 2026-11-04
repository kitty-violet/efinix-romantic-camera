"""Explain physical reuse fences against conservative logical-row dependencies.

Duration replay is a ranking model anchored to measured aggregate work, never
an RTL timing result. Logical-row dependencies may overconstrain sub-row work.
"""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha
from screen_row_pair_layout import problem
from model_contract import compile_contract,load_contract
from fourway_frame_program import hw

def timed(streams,durations):
 done=[0,0];running=[None,None];tick=0;blocked=collections.Counter()
 while done!=list(map(len,streams)):
  for e in (0,1):
   if running[e] and running[e][1]==tick:done[e]+=1;running[e]=None
  for e in (0,1):
   if running[e] or done[e]>=len(streams[e]):continue
   c=streams[e][done[e]]
   if c['dep']<=done[1-e]:running[e]=(c['index'],tick+durations[c['index']])
  if done==list(map(len,streams)):return tick,blocked
  future=[r[1] for r in running if r];assert future,'dependency deadlock'
  nxt=min(future)
  for e in (0,1):
   if running[e] is None and done[e]<len(streams[e]):blocked[streams[e][done[e]]['index']]+=nxt-tick
  tick=nxt
 return tick,blocked

def main():
 p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--descriptors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 cmds=json.loads((a.program/'program.json').read_text());pr=json.loads((a.program/'result.json').read_text());assert sha(a.program/'program.json')==pr['program_sha256']
 dr=json.loads((a.descriptors/'result.json').read_text());assert dr['program_sha256']==pr['program_sha256']
 plan=compile_contract(load_contract());locals={};ti=0
 for gname,width in [('encoder',256),('residual',80),('decoder',256)]:
  group=next(g for g in hw.build_tile_manifest(width)['groups'] if g['name']==gname)
  if gname=='residual':group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+width,320)-1)) for i,x in enumerate(range(0,320,width))])
  for tile in group['tiles']:locals[ti]=problem(plan,group,tile)[0];ti+=1
 phy=[json.loads((a.descriptors/(n+'_dependencies.json')).read_text()) for n in ('compute','dma')]
 phy=[[dict(index=r['index'],dep=r['peer_completed']) for r in rows] for rows in phy]
 physical={r['index']:r['dep'] for rows in phy for r in rows}
 writes={};reads=[{},{}];counts=[0,0];logical=[[],[]];extra=[]
 for c in cmds:
  e=int(c['kind']!='compute');local=locals[c['tile']]
  def key(n,d):
   if local[n]['kind']=='nearest_x2':n=local[n]['parents'][0]
   return c['tile'],n,d['y']
  if c['kind']=='load':r=set();w={key('source',c['row'])}
  elif c['kind']=='store':r={key(c['final'],c['row'])};w=set()
  else:
   parents=local[c['node']]['parents'];r={key(parents[0],d) for d in c['inputs']}
   r.update(key(parents[1],d) for d in c['skip']);w={key(c['node'],d) for d in c['outputs']}
  dep=0
  for u in r|w:
   prev=writes.get(u)
   if prev and prev[0]!=e:dep=max(dep,prev[1])
   if u in w:dep=max(dep,reads[1-e].get(u,0))
  counts[e]+=1
  for u in r:reads[e][u]=counts[e]
  for u in w:writes[u]=(e,counts[e])
  logical[e].append(dict(index=c['command'],dep=dep))
  if physical[c['command']]>dep:extra.append(dict(index=c['command'],kind=c['kind'],group=c['group'],node=c.get('node'),logical=dep,physical=physical[c['command']]))
 # A single calibrated duration vector compares scheduling only. This is not
 # a claim that each command actually took its group average in hardware.
 frame=ROOT/'tmp/frame-execution/row-pair-production-frame1/result.json';fr=json.loads(frame.read_text())
 vectors=ROOT/'tmp/frame-execution/joint-reorder-vectors1/frame_manifest.json';vm=json.loads(vectors.read_text())
 idmap={n['id']:i for i,n in enumerate(vm['nodes'])}
 cc=collections.Counter(c['node'] for c in cmds if c['kind']=='compute')
 durations={};raw={}
 for c in cmds:
  if c['kind']=='compute':durations[c['command']]=fr['nodes'][idmap[c['node']]]['cycles_with_header_control']/cc[c['node']]
  else:raw[c['command']]=24+c['row']['width']*c['channels']/16
 factor=fr['dma_overlap'][0]['active']/sum(raw.values());durations.update({i:v*factor for i,v in raw.items()})
 pt,pb=timed(phy,durations);lt,lb=timed(logical,durations)
 delays=[];by_group=collections.Counter()
 for i,v in pb.items():
  c=cmds[i];delta=v-lb.get(i,0)
  by_group[c['group']+'/'+c['kind']]+=delta
  delays.append(dict(index=i,group=c['group'],kind=c['kind'],node=c.get('node'),extra_wait=delta,physical_wait=v,logical_wait=lb.get(i,0)))
 result=dict(status='DEPENDENCY_REUSE_RANKING_NOT_RTL_TIME',program_sha256=pr['program_sha256'],descriptor_sha256=sha(a.descriptors/'result.json'),
  extra_fences=len(extra),by_kind=dict(collections.Counter(c['kind'] for c in extra)),by_group_kind=dict(collections.Counter(c['group']+'/'+c['kind'] for c in extra)),
  estimated_same_duration_physical_cycles=pt,estimated_same_duration_logical_cycles=lt,
  extra_wait_by_group=dict(by_group),top_delays=sorted(delays,key=lambda x:x['extra_wait'],reverse=True)[:40],
  extra=extra,logical_streams=logical,actual_measurement_sha256=sha(frame),script_sha256=sha(__file__),not_hardware_timing=True)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('extra','logical_streams','top_delays')}))

if __name__=='__main__':main()
