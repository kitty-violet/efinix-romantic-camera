"""Current camera-only synchronizers and finite held-data crossings."""
import inspect,itertools,json,re
from pathlib import Path
from run_units import sha
from native_local_drivers import graph
from audit_snapshot_cells import cells
from probe_full_native_split_map import params

def map_structure(board):
 net=board/'outflow/ti60f225_oob.map.v';text=net.read_text();g=graph(text);cc=cells(text);ps=params(text)
 ff_by_q={tuple(p['Q']):(n,p) for n,(k,p) in g.items() if k=='EFX_FF'}
 chains=[]
 for name,width,stages in [('cfg',2,3),('valid',2,3),('status',15,2)]:
  for i in range(width):
   names=[f'camera_observer/{name}_meta[{i}]~FF']
   for stage in range(1,stages):
    previous=g[names[-1]][1]['Q']
    following=[n for n,(k,p) in g.items() if k=='EFX_FF' and p['D']==previous and 'async_reg=' in cc[n][2]]
    assert len(following)==1,(name,i,stage,following);names+=following
   for n in names:
    k,p=g[n];assert k=='EFX_FF' and p['CLK']==[('net','process_clk_150',None)]
    assert p['CE']==[('literal',"1'b1",None)] and p['SR']==[('net','w_arstn',None)]
    assert 'async_reg=' in cc[n][2] and ps[n]['SR_SYNC']==(1,0)
   chains.append(dict(name=name,bit=i,cells=names))
 held=[]
 for bit in range(96):
  t=('net','camera_diagnostic',bit)
  if (t,) not in ff_by_q:
   assert bit==59;continue
  n,p=ff_by_q[(t,)];assert p['CLK']==[('net','process_clk_150',None)]
  assert p['SR']==[('net','w_arstn',None)] and p['CE']!=[('literal',"1'b1",None)]
  source=ff_by_q[tuple(p['D'])];expected='CLK_5M' if bit<64 else 'i_mipi_rx_pclk'
  assert source[1]['CLK']==[('net',expected,None)]
  held.append(dict(bit=bit,start=source[0]+'|CLK',end=n+'|D',launch=expected,capture='process_clk_150'))
 assert len(held)==95
 # Verify each mapped capture enable consumes only the third valid stage and
 # its own sticky latch, including the one-shot latch's D/CE and reset value.
 lut_q={tuple(p['O']):(n,p) for n,(k,p) in g.items() if k=='EFX_LUT4'}
 checks=0
 for lo,hi,flag,valid in ((0,64,117,[('cfg',0),('cfg',1)]),(64,80,118,[('valid',0)]),(80,96,119,[('valid',1)])):
  latch=('net','camera_diagnostic',flag);ln,lp=ff_by_q[(latch,)]
  assert lp['D']==[('literal',"1'b1",None)] and ps[ln]['SR_VALUE']==(1,0)
  ends=[x for x in held if lo<=x['bit']<hi]
  assert all(g[x['end'][:-2]][1]['CE']==lp['CE'] for x in ends)
  signals=[latch]+[g[next(c for c in chains if c['name']==n and c['bit']==i)['cells'][-1]][1]['Q'][0] for n,i in valid]
  def ev(t,values):
   if t in values:return values[t]
   if t[0]=='literal':assert t[1] in ("1'b0","1'b1");return int(t[1][-1])
   n,p=lut_q[(t,)];idx=sum(ev(p.get('I'+str(i),[('literal',"1'b0",None)])[0],values)<<i for i in range(4))
   return ps[n]['LUTMASK'][1]>>idx&1
  for values in itertools.product((0,1),repeat=len(signals)):
   assert ev(lp['CE'][0],dict(zip(signals,values)))==int(not values[0] and any(values[1:]));checks+=1
 result=dict(status='CAMERA_MAPPED_SYNC_AND_HELD_STRUCTURE_PASS',chains=chains,held=held,capture_enable_truth_checks=checks,
  map_sha256=sha(net),auditor_sha256=sha(__file__),physical_settling_verified=False)
 return result

def inventory(board):
 # Retain the original strict DDR crossing/Gray classification. The old
 # primitive-probe cone is proven dead and removed; its paths no longer exist.
 import fault_probe_clock_cdc as original
 import snapshot_clock_cdc as old
 source=inspect.getsource(old.inventory)
 source=source.replace("end.startswith('u_ddr3_top/controller/')","re.match(r'^(?:protected_inst/)*u_ddr3_top/controller/',end)")
 source=source.replace("start.startswith('u_ddr3_top/controller/')","re.match(r'^(?:protected_inst/)*u_ddr3_top/controller/',start)")
 before="if re.fullmatch(r'bringup_ddr_meta\\[\\d+\\]~FF\\|D', end):"
 after="if re.fullmatch(r'clock_probe/meta\\[[0-5]\\]~FF\\|D', end):\n                assert capture == 'process_clk_150'\n                kind = 'clock_probe_first_stage'\n            el"+before
 assert source.count(before)==1;source=source.replace(before,after)
 before="('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required')"
 source=source.replace(before,"('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required','clock_probe_first_stage')")
 scope=dict(old.__dict__);exec(compile(source,__file__+':DDR','exec'),scope)
 result=scope['inventory'](board);assert result['classes']['clock_probe_first_stage']==6
 result['clock_probe_sync']=original.map_pairs((board/'outflow/ti60f225_oob.map.v').read_text())
 result['camera_structure']=map_structure(board);result['observer_auditor_sha256']=sha(__file__)
 expected={x['end']:x for x in result['camera_structure']['held']};rows=[];reports={}
 for stem in ('camera_cfg_to_system','camera_pixel_to_system'):
  f=board/'outflow'/('ti60f225_oob.'+stem+'.rpt');body=f.read_text();reports[str(f)]=sha(f)
  assert '-npaths 10000 -nworst 1' in body
  paths=body.split('++++ Path ')[1:];assert 0<len(paths)<10000
  for path in paths:
   def field(s):return re.search(r'^'+s+r'\s*:\s*(.+)$',path,re.M)[1].strip()
   end=field('Path End')
   if end not in expected:continue
   row=expected[end];assert field('Path Start')==row['start']
   assert field('Launch Clock').split()[0]==row['launch'] and field('Capture Clock').split()[0]=='process_clk_150'
   delay=float(field('Data Path Delay').split()[0]);assert delay<13.332,(end,delay)
   rows.append(dict(start=row['start'],end=end,delay_ns=delay))
 assert len(rows)==95 and {x['end'] for x in rows}==set(expected)
 result['camera_bundled_settling']=dict(endpoints=95,max_delay_ns=max(x['delay_ns'] for x in rows),conservative_limit_ns=13.332,
  earliest_capture_receiving_periods=3,stable_until_reset=True,reports=reports,paths=rows)
 return result
