"""Translate the frozen full-frame command list to a validated row-pair ABI.

Only local SRAM placement changes. Network, tile width, DDR row traffic,
compute command order and arithmetic are preserved. This is not an RTL run.
"""
import argparse,collections,json,time
from pathlib import Path
from screen_row_pair_layout import problem
from pack_row_pair_backtrack import solve
from model_contract import compile_contract,load_contract
from fourway_frame_program import hw
from run_units import ROOT,sha

def packets(c):
 g=c['input_groups'];og=c['output_groups'];w=c['pixels'];ox=c['output_x']
 rgb=c['node']=='rgb_head';paired=rgb or c['stride2'];pw=c['operator'] in ('pw1x1','pw_fused')
 if paired:
  total=(w+1)//2+7 if rgb else w+8
  for i in range(total):
   if i>=((w+1)//2) if rgb else i>w:continue
   x=ox+2*i-(0 if rgb else 1)
   for q in ([0] if rgb else range(g)):yield x,x+1,q
 elif pw:
  period=8*g*og;total=((w+7)//8)*period+((w-1)%8)
  for i in range(total):
   x=ox+(i//period)*8+i%8
   if x<=ox+w-1:yield x,x,(i//8)%g
 else:
  for q in range(g):
   yield ox-1,ox-1,q
   total=((w+7)//8)*9+((w-1)%8)+1
   for i in range(total-1):
    phase=i%9;x=ox+(i//9)*8+max(0,phase-1)
    if x<=ox+w:yield x,x,q

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--screen',type=Path,required=True);p.add_argument('--tail-proof',type=Path);p.add_argument('--load-lookahead',type=int,default=0);p.add_argument('--store-lookahead',type=int,default=0);p.add_argument('--lookahead-group',choices=('encoder','residual','decoder'),action='append');p.add_argument('--paired-feature-ddr',action='store_true');a=p.parse_args()
 assert 0<=a.load_lookahead<=8
 assert 0<=a.store_lookahead<=16
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 start=time.monotonic();source=ROOT/'tmp/frame-execution/joint-reorder-vectors1/program.json';old=json.loads(source.read_text())
 screen=json.loads(a.screen.read_text());plan=compile_contract(load_contract());layouts={};tile_index=0;manifest=[];global_records={};tile_meta=[]
 # Reserve each incoming row before its original load slot, allowing the
 # hardware DMA engine to overlap the preceding compute instead of reusing
 # its still-live address. This only EXTENDS proven lifetimes; never removes
 # a real hazard or changes the mathematical/DDR command order.
 load_slots=[c['command'] for c in old if c['kind']=='load']
 early_load={i:load_slots[max(0,k-a.load_lookahead)] for k,i in enumerate(load_slots)}
 store_slots=[c['command'] for c in old if c['kind']=='store']
 late_store={i:store_slots[min(len(store_slots)-1,k+a.store_lookahead)] for k,i in enumerate(store_slots)}
 for gname,width in [('encoder',256),('residual',80),('decoder',256)]:
  group=next(g for g in hw.build_tile_manifest(width)['groups'] if g['name']==gname)
  if gname=='residual':group=dict(group,tiles=[dict(tile_index=i,commit_interval=dict(x0=x,x1=min(x+width,320)-1)) for i,x in enumerate(range(0,320,width))])
  offsets=next(c['offsets'] for c in screen['cases'] if c['group']==gname)
  for tile in group['tiles']:
   local,tasks,records=problem(plan,group,tile);addresses={};proof=[]
   if a.paired_feature_ddr:
    for c in old:
     if c['tile']==tile_index and c['kind']=='load' and c['row']['groups']==4:
      d=c['row'];r=records['source',d['y']]
      r['phase']=d['origin']%2
      r['length']=((d['width']+r['phase']+1)//2)*4
   # The production program has reordered DMA/compute tasks. Lifetimes from
   # the earlier scheduler are unsafe even when every address is in range.
   spans={}
   def touch(n,d,i):
    if local[n]['kind']=='nearest_x2':n=local[n]['parents'][0]
    key=(n,d['y']);assert key in records
    if key not in spans:spans[key]=[i,i]
    else:spans[key][1]=i
   for c in old:
    if c['tile']!=tile_index:continue
    i=c['command']
    if c['kind']!='compute':touch('source' if c['kind']=='load' else c['final'],c['row'],i)
    else:
     parents=local[c['node']]['parents']
     for d in c['inputs']:touch(parents[0],d,i)
     for d in c['outputs']:touch(c['node'],d,i)
     for d in c['skip']:touch(parents[1],d,i)
   assert set(spans)==set(records),(tile_index,len(spans),len(records))
   for key,r in records.items():r['born'],r['last']=spans[key]
   if a.load_lookahead and (not a.lookahead_group or gname in a.lookahead_group):
    for c in old:
     if c['tile']==tile_index and c['kind']=='load':
      r=records['source',c['row']['y']];r['born']=min(r['born'],early_load[c['command']])
   if a.store_lookahead and (not a.lookahead_group or gname in a.lookahead_group):
    for c in old:
     if c['tile']==tile_index and c['kind']=='store':
      r=records[c['final'],c['row']['y']];r['last']=max(r['last'],late_store[c['command']])
   for (n,y),r in records.items():r['bank']=(y+offsets[n])%6
   layouts[tile_index]=(local,addresses)
   # Reordered commands may preload the next tile before the previous tile
   # retires. Place ALL tiles on one common timeline, never independent arenas.
   for (n,y),r in records.items():global_records[f'{tile_index}:{n}',y]=r
   tile_meta.append(dict(tile=tile_index,group=gname,width=width))
   tile_index+=1
 global_proof=[]
 peaks=[]
 for bank in range(6):
  events=collections.defaultdict(int)
  for r in global_records.values():
   if r['bank']==bank:events[r['born']]+=r['length'];events[r['last']+1]-=r['length']
  at=peak=when=0
  for tick,delta in sorted(events.items()):
   at+=delta
   if at>peak:peak=at;when=tick
  live=[dict(node=k[0],row=k[1],length=r['length'],born=r['born'],last=r['last']) for k,r in global_records.items() if r['bank']==bank and r['born']<=when<=r['last']]
  peaks.append(dict(bank=bank,live_words=peak,at_command=when,rows=live))
 (root/'bank_peak_demand.json').write_text(json.dumps(dict(load_lookahead=a.load_lookahead,store_lookahead=a.store_lookahead,groups=a.lookahead_group,peaks=peaks),indent=2)+'\n')
 for bank in range(6):
  if peaks[bank]['live_words']>1024:
   failure=dict(status='ROW_PAIR_LIVE_CAPACITY_EXCEEDED',bank=bank,peak=peaks[bank],limit=1024)
   (root/'layout_failure.json').write_text(json.dumps(failure,indent=2)+'\n');raise ValueError(failure['status'])
  solved=solve(global_records,bank)
  if not solved['passed']:
   (root/'layout_failure.json').write_text(json.dumps(dict(status='ROW_PAIR_BOUNDED_SEARCH_EXHAUSTED',bank=bank,nodes=solved['nodes'],seconds=solved['seconds'],peaks=peaks),indent=2)+'\n')
   raise ValueError(('global lifetime placement',bank,solved['nodes']))
  global_proof.append(dict(bank=bank,nodes=solved['nodes'],high=max(r['base']+r['length'] for r in solved['allocations'])))
  for r in solved['allocations']:
   t,n=r['node'].split(':',1);r=dict(r,node=n);layouts[int(t)][1][n,r['row']]=r
  print(dict(bank=bank,rows=len(solved['allocations']),high=global_proof[-1]['high']),flush=True)
 for m in tile_meta:
  addresses=layouts[m['tile']][1]
  manifest.append(dict(**m,banks=global_proof,allocations=list(addresses.values())))
 converted=[];read_checks=write_checks=pairs_checked=padding_reads=0
 for c in old:
  local,addresses=layouts[c['tile']];out=dict(c)
  def physical(n):return local[n]['parents'][0] if local[n]['kind']=='nearest_x2' else n
  def desc(n,d,output=False):
   n=physical(n);r=addresses[n,d['y']];v={k:value for k,value in d.items() if k not in ('plane','base')};v.update(bank=r['bank'],base=r['base'],phase=r['phase'])
   if output:
    x=c['output_x']//(2 if c['stride2'] else 1);delta=x-d['origin'];assert delta>=0
    v['base']+=((delta+v['phase'])//2)*v['groups'];v['phase']=(delta+v['phase'])%2;v['origin']=x
   assert 0<=v['base']<1024
   return v
  if c['kind']!='compute':
   out['row']=desc('source' if c['kind']=='load' else c['final'],c['row'])
   if a.paired_feature_ddr and out['row']['groups']==4:
    d=out['row'];assert d['phase']==d['origin']%2
    if c['kind']=='store':assert d['phase']==0 and d['width']%2==0
  else:
   parents=local[c['node']]['parents'];out['inputs']=[desc(parents[0],d) for d in c['inputs']]
   out['outputs']=[desc(c['node'],d,True) for d in c['outputs']]
   out['skip']=[desc(parents[1],d) for d in c['skip']]
   assert len(set(d['phase'] for d in out['inputs']))==1
   # Every physical bank can have only one row address per packet. Adjacent
   # clamped/nearest aliases are legal only when they name the same row.
   bank_rows={}
   for d in out['inputs']:
    row=(d['y'],d['base']);assert d['bank'] not in bank_rows or bank_rows[d['bank']]==row
    bank_rows[d['bank']]=row
   assert len({d['bank'] for d in out['outputs']})==len(out['outputs'])
   for raw_a,raw_b,q in packets(c):
    xa=max(0,min(c['source_width']-1,raw_a));xb=max(0,min(c['source_width']-1,raw_b))
    if c['nn']:xa//=2;xb//=2
    for d in out['inputs']:
     la=xa-d['origin'];lb=xb-d['origin']
     assert 0<=la<d['width'],('first half outside mathematical row',c['command'],la,d)
     for x in (la,lb):
      if not 0<=x<d['width']:
       # The old stride2 stream emits one trailing pair. Its unused second
       # half may be exactly one pixel past the mathematical tile. It must
       # still lie in reserved padding; initialization/consumer masking is
       # an explicit integration obligation, never an out-of-range bypass.
       assert c['stride2'] and x==lb==d['width'] and raw_b==c['output_x']+2*c['pixels'],('source range',c['command'],x,d)
       padding_reads+=1
     aa=d['base']+((la+d['phase'])//2)*d['groups']+q;ab=d['base']+((lb+d['phase'])//2)*d['groups']+q
     assert 0<=aa<1024 and aa==ab,('pair split/address',c['command'],aa,ab,d)
     reserved=((d['width']+d['phase']+1)//2)*d['groups']
     assert aa<d['base']+reserved and ab<d['base']+reserved,('unreserved padding',c['command'],aa,ab,d)
     # Pair lane inversion reconstructs the original unpadded physical x.
     for x,addr in ((xa,aa),(xb,ab)):
      half=(x-d['origin']+d['phase'])%2
      decoded=2*((addr-d['base'])//d['groups'])+half-d['phase']+d['origin'];assert decoded==x
     read_checks+=2
    pairs_checked+=1
  targets=[out['row']] if c['kind']!='compute' else out['outputs']+out['skip']
  for d in targets:
   delta=d.get('delta',0) if d in out.get('skip',[]) else 0
   width=c['pixels'] if c['kind']=='compute' and d in out['skip'] else d['width']
   for x in (delta,delta+width-1):
    addr=d['base']+((x+d['phase'])//2)*d['groups']+d['groups']-1
    assert 0<=addr<1024,('writer/DMA/skip range',c['command'],addr,d)
    write_checks+=1
  converted.append(out)
 # Independent semantic last-writer check in the actual production order.
 # In-range addresses and hazard dependencies alone cannot detect a load that
 # has already overwritten the tensor which a later command intends to read.
 owners=[None]*12288;semantic_reads=semantic_writes=0
 for c in converted:
  local,_=layouts[c['tile']]
  def owner_access(n,d,x0,count,write):
   nonlocal semantic_reads,semantic_writes
   if local[n]['kind']=='nearest_x2':n=local[n]['parents'][0]
   for x in range(x0,x0+count):
    assert 0<=x<d['width'],(c['command'],n,x,d)
    for g in range(d['groups']):
     u=d['bank']*2048+2*(d['base']+((x+d['phase'])//2)*d['groups']+g)+(x+d['phase'])%2
     identity=(c['tile'],n,d['y'],d['origin']+x,g)
     if write:owners[u]=identity;semantic_writes+=1
     else:
      assert owners[u]==identity,('semantic SRAM overwrite/uninitialized',c['command'],u,identity,owners[u])
      semantic_reads+=1
  if c['kind']!='compute':
   if a.paired_feature_ddr and c['kind']=='load' and c['row']['groups']==4:
    # The straight DMA also writes the rounded edge halves. Mark them as
    # padding in the independent owner oracle: a later logical read must
    # never consume these halves as a different still-live tensor.
    d=c['row'];pairs=(d['width']+d['phase']+1)//2
    for offset in range(pairs*8):owners[d['bank']*2048+2*d['base']+offset]=('aligned DMA padding',c['command'])
   owner_access('source' if c['kind']=='load' else c['final'],c['row'],0,c['row']['width'],c['kind']=='load')
  else:
   parents=local[c['node']]['parents'];pw=c['operator'] in ('pw1x1','pw_fused')
   lo=max(0,c['output_x']-(0 if pw else 1))
   hi=min(c['source_width']-1,c['output_x']+(c['pixels']-1 if pw else 2*c['pixels']-1 if c['stride2'] else c['pixels']))
   if c['nn']:lo//=2;hi//=2
   for d in c['inputs']:owner_access(parents[0],d,lo-d['origin'],hi-lo+1,False)
   for d in c['skip']:owner_access(parents[1],d,d.get('delta',0),c['pixels'],False)
   for d in c['outputs']:owner_access(c['node'],d,0,c['pixels'],True)
 # All nonlocal compute/DDR semantics must be byte-for-byte unchanged.
 for before,after in zip(old,converted):
  for k,v in before.items():
   if k not in ('row','inputs','outputs','skip'):assert after[k]==v
 program=root/'program.json';program.write_text(json.dumps(converted,separators=(',',':'))+'\n')
 (root/'allocations.json').write_text(json.dumps(manifest,separators=(',',':'))+'\n')
 proof=None
 if a.tail_proof:
  proof=json.loads((a.tail_proof/'result.json').read_text());assert proof['status']=='COMPACT_DISPATCH_FOUR_STATE_CYCLE_AB_PASS' and proof['poison_trailing_half']
  records=json.loads((a.tail_proof/'inputs.json').read_text())['inputs']
  candidate=next(r for r in records if Path(r['snapshot']).name=='candidate.sv')
  expected_source=Path(__file__).parent/'variants/ternary_stride/unified_column_dispatch32.sv'
  assert sha(candidate['snapshot'])==candidate['sha256'] and sha(expected_source)==candidate['source_sha256']
  assert Path(candidate['snapshot']).read_text()==expected_source.read_text()
  assert sha(a.tail_proof/'simulation.log')==proof['log_sha256']
 result=dict(status='ROW_PAIR_FULL_GRAPH_ADDRESS_PASS' if proof else 'ROW_PAIR_FULL_GRAPH_ADDRESS_PASS_PADDING_OBLIGATION_OPEN',commands=len(converted),tiles=tile_index,read_scalar_addresses_checked=read_checks,packet_coordinates_checked=pairs_checked,writer_dma_skip_bounds_checked=write_checks,
  issued_trailing_padding_halves=padding_reads,padding_initialization_or_consumer_mask_proved=proof is not None,
  reader_profile='full-rate paired stride; intended ternary_stride dispatch, not compact-feed decimated stride',
  tail_proof_sha256=sha(a.tail_proof/'result.json') if proof else None,
  original_command_order_preserved=True,original_tile_widths_preserved=True,original_ddr_traffic_preserved=not a.paired_feature_ddr,paired_feature_ddr=a.paired_feature_ddr,ram10=84,bytes=98304,
  feature_load_padding_bytes=sum((((c['row']['width']+c['row']['phase']+1)//2)*64-c['row']['width']*32) for c in converted if c['kind']=='load' and c['row']['groups']==4) if a.paired_feature_ddr else 0,
  lifetimes_from_actual_production_order=True,cross_tile_lifetimes=True,semantic_last_writer_reads=semantic_reads,semantic_writes=semantic_writes,
  load_lookahead_reserved=a.load_lookahead,
  store_lookahead_reserved=a.store_lookahead,
  load_lookahead_groups=a.lookahead_group,
  program_sha256=sha(program),allocations_sha256=sha(root/'allocations.json'),reference_program_sha256=sha(source),screen_sha256=sha(a.screen),script_sha256=sha(__file__),seconds=time.monotonic()-start,
  rtl_executed=False,full_frame_30fps=False,board_capacity_pass=False)
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
if __name__=='__main__':main()
