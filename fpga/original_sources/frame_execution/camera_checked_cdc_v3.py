"""Classify the retained read-only done crossing by its exact mapped cone.

report_timing -nworst 1 may select any of the six step bits. Do not assume
the least significant bit must be the worst path after every placement.
All other CDC classification, endpoint counts and settling limits stay intact.
"""
import inspect,itertools
from run_units import sha
from native_local_drivers import graph
from probe_full_native_split_map import params
import camera_checked_cdc_v2 as prior

def done_cone(text):
 g=graph(text);ps=params(text)
 ff={p['Q'][0]:n for n,(k,p) in g.items() if k=='EFX_FF'}
 lut={p['O'][0]:n for n,(k,p) in g.items() if k=='EFX_LUT4'}
 end='cell_probe/done_sync[0]~FF';k,p=g[end];assert k=='EFX_FF'
 target=p['D'][0];seen=set();sources=set();order=[]
 def visit(t):
  if t[0]=='literal':assert t[1] in ("1'b0","1'b1");return
  if t in ff:sources.add(ff[t]);return
  assert t in lut,('unknown done cone driver',t)
  n=lut[t]
  if n in seen:return
  seen.add(n)
  for pin,ts in g[n][1].items():
   if pin!='O':
    for s in ts:visit(s)
  order.append(n)
 visit(target)
 names=[f'cell_probe/ddr_probe[{i}]~FF' for i in range(48,54)]
 assert sources==set(names),('unexpected retained done sources',sorted(sources))
 for n in names:
  k,p=g[n];assert k=='EFX_FF'
  assert p['CLK']==[('net','sdram_clk',None)] and p['CE']==[('literal',"1'b1",None)]
  assert p['SR']==[('net','cell_probe/ddr_probe',55)]
  for a,v in dict(CLK_POLARITY=1,CE_POLARITY=1,SR_POLARITY=0,SR_SYNC=1,SR_VALUE=0).items():assert ps[n][a]==(1,v),(n,a)
 invert=1-ps[end]['D_POLARITY'][1]
 for step in range(64):
  vals={g[n][1]['Q'][0]:(step>>i)&1 for i,n in enumerate(names)}
  def val(t):return int(t[1][-1]) if t[0]=='literal' else vals[t]
  for n in order:
   pins=g[n][1];idx=sum(val(pins.get('I'+str(i),[('literal',"1'b0",None)])[0])<<i for i in range(4))
   vals[pins['O'][0]]=ps[n]['LUTMASK'][1]>>idx&1
  assert (val(target)^invert)==int(step==32),('wrong retained done decode',step)
 return dict(done_starts=[n+'|CLK' for n in names],done_cone_cells=order,done_decode_truth_cases=64,
   done_decode='step[5:0] == 32',observation_only=True)

def retained_probe(board):
 result=prior.retained_probe(board)
 result.pop('done_start')
 result.update(done_cone((board/'outflow/ti60f225_oob.map.v').read_text()))
 return result

def inventory(board):
 code=inspect.getsource(prior.inventory)
 needle="start==retained['done_start']";assert code.count(needle)==1
 code=code.replace(needle,"start in retained['done_starts']")
 env=dict(prior.__dict__,retained_probe=retained_probe,__file__=__file__)
 exec(compile(code,__file__+':exact-cone-classifier','exec'),env)
 result=env['inventory'](board)
 result['retained_done_classifier_sha256']=sha(__file__)
 result['retained_done_classifier_parent_sha256']=sha(prior.__file__)
 return result

map_structure=prior.map_structure
