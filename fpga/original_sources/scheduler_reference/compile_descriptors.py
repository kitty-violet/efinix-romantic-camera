"""S4R1 0x55 descriptor ABI; exact completed-operation dependencies.

The old 0x52 layout is retained through bit446. Bits447/448 explicitly add
stride4/nearest4; all other reserved bits remain zero. No no-op padding.
"""
from pathlib import Path
import ast,json,random,struct,time,zlib,sys
from coefficients import R,INPUTS,oracle,sha
from compile_schedule import accesses
P=Path('E:/fpga')
def schemas():
 src=P/'hw/efinity/prototypes/frame_execution/compile_production_descriptors.py';widths={}
 for n in ast.parse(src.read_text('utf8')).body:
  if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('COMPUTE_WIDTHS','DMA_WIDTHS'):widths[n.targets[0].id]=ast.literal_eval(n.value)
 cw=widths['COMPUTE_WIDTHS'];dw=widths['DMA_WIDTHS'];cw.update({10:18,13:12,20:12});dw[4]=3
 return [[('peer_completed',14),('program_index',15)]+[('f'+str(k),v)for k,v in cw.items()]+[('storage_phase',1),('out_phase',4),('stride4',1),('nearest4',1)],
 [('peer_completed',14)]+[('f'+str(k),v)for k,v in dw.items()]]
def replay(streams,regions,seed):
 rng=random.Random(seed);done=[0,0];running=[None,None];tick=overlaps=0
 while done!=list(map(len,streams)):
  for e in (0,1):
   if running[e]and running[e][1]==tick:done[e]+=1;running[e]=None
  for e in (0,1):
   if running[e]or done[e]>=len(streams[e]):continue
   row=streams[e][done[e]]
   if row['peer_completed']>done[1-e]:continue
   i=row['program_index'];r,w=regions[i]
   if running[1-e]:
    rr,ww=regions[running[1-e][0]];assert not(w&(rr|ww)or r&ww),('LIVE_OVERLAP',i,running[1-e][0]);overlaps+=1
   running[e]=(i,tick+rng.randrange(3,120))
  if done==list(map(len,streams)):break
  assert any(running),('DEADLOCK',done)
  tick=min(p[1]for p in running if p)
 assert overlaps>0
 return dict(seed=seed,completed=done,overlapped_starts=overlaps,abstract_ticks=tick)
def pack(values,schema,version,size):
 word=offset=0
 for name,width in schema:
  v=values[name];assert 0<=v<1<<width,(name,v,width)
  word|=v<<offset;offset+=width
 word|=version<<(size*8-(8 if size==64 else 4))
 off=0
 for name,width in schema:
  assert(word>>off)&((1<<width)-1)==values[name];off+=width
 return word.to_bytes(size,'little')
def main():
 start=time.monotonic();folder=R/'data'/('schedule'+(sys.argv[2]if len(sys.argv)>2 else'1'));proof=json.loads((folder/'result.json').read_text('utf8'))
 raw=(folder/'program.json').read_bytes();assert sha(raw)==proof['program_sha256'];commands=json.loads(raw)
 graph,_=oracle.load(Path(INPUTS['selected'][0]['folder']));node_ids={n['id']:i for i,n in enumerate(graph['nodes'])}
 schema=schemas();root=R/'data'/('descriptors'+(sys.argv[1]if len(sys.argv)>1 else'1'));root.mkdir(parents=True,exist_ok=False)
 latest_write=[None]*12288;latest_read=[[0]*12288 for _ in range(2)];counts=[0,0];streams=[[],[]];regions=[]
 for c in commands:
  r,w,padding=accesses(c);rs={u for u,_ in r};ws={u for u,_ in w}|padding;regions.append((rs,ws));e=int(c['kind']!='compute');peer=1-e;dep=0
  for u in rs|ws:
   if latest_write[u]and latest_write[u][0]==peer:dep=max(dep,latest_write[u][1])
   if u in ws:dep=max(dep,latest_read[peer][u])
  counts[e]+=1
  for u in rs:latest_read[e][u]=counts[e]
  for u in ws:latest_write[u]=(e,counts[e])
  streams[e].append(dict(peer_completed=dep,program_index=c['command']))
 trials=[replay(streams,regions,s)for s in (71,391,1027)]
 styles=[];dma_reference=None
 for style in INPUTS['selected']:
  layout=json.loads((R/f"data/resident1/slot{style['slot']}/resident-layout.json").read_text('utf8'));params={r['node']:r for r in layout['operators']}
  output=[bytearray(),bytearray()];indices=[0,0]
  for c in commands:
   e=int(c['kind']!='compute');values=dict(streams[e][indices[e]]);indices[e]+=1
   fields={i:0 for i in range(32)};fields[30]=c['tile'];fields[31]=c['command']
   if e==0:
    node=c['node'];p=params[c['parameter_node']];pw=c['operator']in ('pw1x1','pw_fused');stem=c['operator']=='conv3x3';fused=c['operator']=='pw_fused'
    residual=layout['residual'].get(node,dict(main_scaled=False,multiplier=0,post_multiplier=0))
    fields.update({1:node_ids[node],2:c['pixels'],3:c['rows'],4:c['input_groups'],5:c['output_groups'],
     6:int(pw)|(int(stem)<<2)|(int(c['stride']==2)<<3)|(int(fused)<<4)|(int(residual['main_scaled'])<<5),
     7:c['output_x'],8:c['source_width'],9:c['inputs'][0]['origin'],16:p['weight']['offset'],17:p['bias']['offset'],18:p['multiplier']['offset'],19:p['relu']['offset'],
     23:c['skip'][0]['delta']if fused else 0,24:residual['multiplier']&0x3ffff,25:residual['post_multiplier']&0x3ffff,28:c['mask'],29:1<<node_ids[node]})
    if fused:fields[29]|=1<<node_ids[c['parameter_node']]
    if c['nn_factor']==4:fields[29]|=1<<node_ids['up1_nearest']
    for rows,bankfield,lowfield,highfield in [(c['inputs'],10,11,12),(c['outputs'],13,14,15),(c['skip'],20,21,22)]:
     fields[bankfield]=sum(d['bank']<<(3*y)for y,d in enumerate(rows));b=sum(d['base']<<(10*y)for y,d in enumerate(rows));fields[lowfield]=b&0xffffffff;fields[highfield]=b>>32
    phases={d['phase']for d in c['inputs']};assert len(phases)==1
    values.update(storage_phase=phases.pop(),out_phase=sum(d['phase']<<y for y,d in enumerate(c['outputs'])),stride4=int(c['stride']==4),nearest4=int(c['nn_factor']==4))
   else:
    d=c['row'];store=c['kind']=='store';rgb=c['channels']==3
    addresses={'encoder':(0,0x02000000),'residual':(0x02000000,0x03000000),'decoder':(0x03000000,0x00800000)}
    base=addresses[c['group']][int(store)];address=base+(d['y']*c['global_width']+d['origin'])*(3 if rgb else c['channels'])
    # Mirror the actual row_pair_dma acceptance contract, not merely semantic
    # tensor ranges. A rejected descriptor cannot be rescued by bank lifetimes.
    assert address%4==0,('DMA_RGB_ALIGNMENT',c['command'],address)
    assert rgb and d['groups']==1 or not rgb and d['groups']==4
    assert rgb or address%32==0
    assert d['base']+((d['width']+1)//2 if rgb else ((d['width']+d['phase']+1)//2)*4)<=1024
    fields.update({0:int(store),1:node_ids[c['final']]if store else 0,2:d['width'],3:d['groups'],4:d['bank'],5:d['base'],6:int(rgb),7:address,
     8:int(store and rgb),9:d['y']if store and rgb else 0,10:d['origin']if store and rgb else 0})
    assert d['width']<=511 and address+(d['width']*(3 if rgb else c['channels']))<=base+0x400000
    assert rgb and d['phase']==0 or not rgb and d['phase']==(address//32)%2
    if store and rgb:assert d['width']==256
   values.update({'f'+str(k):v for k,v in fields.items()});output[e].extend(pack(values,schema[e],0x55 if e==0 else 3,64 if e==0 else 16))
  sd=root/f"slot{style['slot']}";sd.mkdir();(sd/'compute.bin').write_bytes(output[0])
  if dma_reference is None:dma_reference=output[1];(root/'dma.bin').write_bytes(output[1])
  else:assert output[1]==dma_reference
  styles.append(dict(slot=style['slot'],compute_sha256=sha(output[0]),compute_crc32=f'{zlib.crc32(output[0]):08x}'))
 result=dict(status='S4R1_DESCRIPTOR_ABI_DEPENDENCY_PASS_NOT_RTL_RELEASE',compute_records=counts[0],dma_records=counts[1],compute_bytes=counts[0]*64,dma_bytes=counts[1]*16,
  schedule_sha256=sha(raw),styles=styles,dma_sha256=sha(dma_reference),dma_crc32=f'{zlib.crc32(dma_reference):08x}',replays=trials,
  expected_node_mask=(1<<len(graph['nodes']))-1,compute_version=0x55,new_bits=dict(stride4=447,nearest4=448),source_bytes=sum(c['row']['width']*c['channels']for c in commands if c['kind']=='load'),
  result_bytes=sum(c['row']['width']*c['channels']for c in commands if c['kind']=='store'),seconds=time.monotonic()-start,rtl_decoder_verified=False,hardware_access=False)
 (root/'schema.json').write_text(json.dumps(schema,indent=2),'utf8');(root/'result.json').write_text(json.dumps(result,indent=2),'utf8');print(json.dumps(result))
if __name__=='__main__':main()
