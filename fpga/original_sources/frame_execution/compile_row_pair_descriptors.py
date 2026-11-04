"""Versioned row-pair ABI and physical cross-engine hazards for the full graph.

The scheduler must wait for completion counts, not requests. This compiler
rebuilds dependencies on actual 64-bit halves in the six-bank placement; old
plane dependencies are deliberately not reused. No cycle/FPS claim is made.
"""
import argparse,ast,collections,json,random,time,zlib
from pathlib import Path
from run_units import ROOT,HERE,sha

def access(c,paired_feature_ddr=False):
 reads=set();writes=set()
 def units(d,x0,width):
  assert x0>=0 and x0+width<=d['width'],(c['command'],d,x0,width)
  return {d['bank']*2048+2*(d['base']+((x+d['phase'])//2)*d['groups']+g)+(x+d['phase'])%2
          for x in range(x0,x0+width) for g in range(d['groups'])}
 if c['kind']!='compute':
  d=c['row'];(reads if c['kind']=='store' else writes).update(units(d,0,d['width']))
  if paired_feature_ddr and c['kind']=='load' and d['groups']==4:
   writes.update(range(d['bank']*2048+2*d['base'],d['bank']*2048+2*d['base']+((d['width']+d['phase']+1)//2)*8))
 else:
  pw=c['operator'] in ('pw1x1','pw_fused');lo=c['output_x']-(0 if pw else 1)
  hi=c['output_x']+(c['pixels']-1 if pw else 2*c['pixels']-1 if c['stride2'] else c['pixels'])
  lo=max(0,min(c['source_width']-1,lo));hi=max(0,min(c['source_width']-1,hi))
  if c['nn']:lo//=2;hi//=2
  for d in c['inputs']:reads.update(units(d,lo-d['origin'],hi-lo+1))
  for d in c['skip']:reads.update(units(d,d.get('delta',0),c['pixels']))
  for d in c['outputs']:
   # Output descriptors have already advanced their base/phase to output_x.
   # Their width still describes the underlying row, so only this block writes.
   writes.update(units(d,0,c['pixels']))
 assert all(0<=x<12288 for x in reads|writes)
 return reads,writes

def replay(streams,regions,seed):
 rng=random.Random(seed);done=[0,0];running=[None,None];tick=overlap=0
 while done!=list(map(len,streams)):
  for e in (0,1):
   if running[e] and running[e][1]==tick:done[e]+=1;running[e]=None
  for e in (0,1):
   if running[e] or done[e]>=len(streams[e]):continue
   row=streams[e][done[e]]
   if row['dep']>done[1-e]:continue
   index=row['index'];r,w=regions[index]
   if running[1-e]:
    rr,ww=regions[running[1-e][0]];assert not(w & (rr|ww) or r & ww),('overlapping live operations',index,running[1-e][0])
    overlap+=1
   running[e]=(index,tick+rng.randrange(3,20) if e==0 else tick+rng.randrange(2,12))
  future=[x[1] for x in running if x]
  if done==list(map(len,streams)):break
  assert future,('dependency deadlock',done)
  tick=min(future)
 assert overlap>0
 return dict(seed=seed,completed=done,overlapped_starts=overlap,abstract_ticks=tick,not_hardware_cycles=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--program',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir();start=time.monotonic()
 pr=json.loads((a.program/'result.json').read_text());assert pr['status']=='ROW_PAIR_FULL_GRAPH_ADDRESS_PASS'
 assert pr.get('lifetimes_from_actual_production_order') and pr.get('semantic_last_writer_reads',0)>0
 assert sha(a.program/'program.json')==pr['program_sha256'];cmds=json.loads((a.program/'program.json').read_text())
 paired=pr.get('paired_feature_ddr',False);dma_version=3 if paired else 2
 vectors=ROOT/'tmp/frame-execution/joint-reorder-vectors1';vm=json.loads((vectors/'frame_manifest.json').read_text())
 assert sha(vectors/'program.hex')==vm['files']['program.hex'];old=[int(x,16) for x in (vectors/'program.hex').read_text().splitlines()]
 assert len(old)==len(cmds)==19229
 old_schema=HERE/'compile_production_descriptors.py';widths={}
 for n in ast.parse(old_schema.read_text()).body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('COMPUTE_WIDTHS','DMA_WIDTHS'):widths[n.targets[0].id]=ast.literal_eval(n.value)
 cw=widths['COMPUTE_WIDTHS'];dw=widths['DMA_WIDTHS'];cw.update({10:18,13:12,20:12});dw[4]=3
 schemas=[ [('peer_completed',14),('program_index',15)]+[('f'+str(k),v) for k,v in cw.items()]+[('storage_phase',1),('out_phase',4)],
           [('peer_completed',14)]+[('f'+str(k),v) for k,v in dw.items()] ]
 streams=[[],[]];regions=[];last_write=[None]*12288;last_read=[[0]*12288 for _ in range(2)];counts=[0,0]
 patch_fields={10,11,12,13,14,15,20,21,22}
 for i,c in enumerate(cmds):
  assert c['command']==i
  e=0 if c['kind']=='compute' else 1;r,w=access(c,paired);regions.append((r,w));dep=0
  for u in r|w:
   prev=last_write[u]
   if prev and prev[0]!=e:dep=max(dep,prev[1])
   if u in w:dep=max(dep,last_read[1-e][u])
  counts[e]+=1
  for u in r:last_read[e][u]=counts[e]
  for u in w:last_write[u]=(e,counts[e])
  fields=[(old[i]>>(32*k))&0xffffffff for k in range(32)];original=list(fields)
  extra={}
  if e==0:
   def pack(ds,key,n):return sum(d[key]<<(n*y) for y,d in enumerate(ds))
   fields[10]=pack(c['inputs'],'bank',3);b=pack(c['inputs'],'base',10);fields[11]=b&0xffffffff;fields[12]=b>>32
   fields[13]=pack(c['outputs'],'bank',3);b=pack(c['outputs'],'base',10);fields[14]=b&0xffffffff;fields[15]=b>>32
   fields[20]=pack(c['skip'],'bank',3);b=pack(c['skip'],'base',10);fields[21]=b&0xffffffff;fields[22]=b>>32
   phases={d['phase'] for d in c['inputs']};assert len(phases)==1
   extra=dict(storage_phase=phases.pop(),out_phase=pack(c['outputs'],'phase',1))
   assert all(fields[k]==original[k] for k in range(32) if k not in patch_fields)
  else:
   assert c['row']['phase']==(c['row']['origin']%2 if paired and c['row']['groups']==4 else 0)
   if paired and c['row']['groups']==4:
    assert fields[7]%32==0 and (fields[7]//32)%2==c['row']['phase']
   fields[4]=c['row']['bank'];fields[5]=c['row']['base']
   assert all(fields[k]==original[k] for k in range(32) if k not in (4,5))
  values=dict(peer_completed=dep,program_index=i,**{'f'+str(k):v for k,v in enumerate(fields)},**extra)
  packed=0;offset=0
  for name,bits in schemas[e]:
   val=values[name];assert 0<=val<(1<<bits),(i,name,val,bits);packed|=val<<offset;offset+=bits
  assert offset==(447 if e==0 else 121)
  packed|=(0x52<<504) if e==0 else (dma_version<<124)
  decoded={};offset=0
  for name,bits in schemas[e]:decoded[name]=(packed>>offset)&((1<<bits)-1);offset+=bits
  assert decoded=={name:values[name] for name,_ in schemas[e]}
  streams[e].append(dict(index=i,dep=dep,packed=packed))
 assert counts==[8789,10440]
 trials=[replay(streams,regions,s) for s in (71,391,1027)]
 layouts={};files={}
 for e,name in enumerate(('compute','dma')):
  size=64 if e==0 else 16;binary=b''.join(c['packed'].to_bytes(size,'little') for c in streams[e])
  path=root/(name+'.bin');path.write_bytes(binary);files[path.name]=sha(path)
  off=0;layout=[]
  for field,bits in schemas[e]:layout.append(dict(field=field,offset=off,width=bits));off+=bits
  layouts[name]=dict(fields=layout,record_bytes=size,used_bits=off,version_offset=504 if e==0 else 124,version_width=8 if e==0 else 4,version=0x52 if e==0 else dma_version,crc32=f'{zlib.crc32(binary):08x}',records=len(streams[e]))
  (root/(name+'_dependencies.json')).write_text(json.dumps([dict(index=c['index'],peer_completed=c['dep']) for c in streams[e]],separators=(',',':'))+'\n')
 (root/'schema.json').write_text(json.dumps(layouts,indent=2)+'\n')
 result=dict(status='ROW_PAIR_DESCRIPTOR_ABI_AND_PHYSICAL_DEPENDENCIES_PASS',compute_records=8789,dma_records=10440,
  record_bytes=[64,16],total_bytes=8789*64+10440*16,used_bits=[447,121],files=files,schema_sha256=sha(root/'schema.json'),
  program_path=str(a.program.resolve()),program_sha256=pr['program_sha256'],old_vector_manifest_sha256=sha(vectors/'frame_manifest.json'),old_schema_sha256=sha(old_schema),network_contract_sha256=sha(ROOT/'network_ab/model_b_contract.json'),
  reference_address_units='six banks of2048 64-bit halves',paired_feature_ddr=paired,dma_version=dma_version,dependency_replays=trials,seconds=time.monotonic()-start,script_sha256=sha(__file__),
  rtl_decoder_verified=False,full_frame_numeric=False,whole_board=False,throughput_claim=False)
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('dependency_replays','files')},flush=True)
if __name__=='__main__':main()
