"""Finite added camera crossings, retaining the existing full DDR/Gray audit."""
import itertools,json,re
from run_units import sha
from native_local_drivers import graph
from audit_snapshot_cells import cells
from probe_full_native_split_map import params
import camera_checked_cdc_v3 as prior

def map_structure(board):
 net=board/'outflow/ti60f225_oob.map.v';text=net.read_text();g=graph(text);ps=params(text);attrs=cells(text)
 ff={p['Q'][0]:(n,p) for n,(k,p) in g.items() if k=='EFX_FF'}
 lut={p['O'][0]:(n,p) for n,(k,p) in g.items() if k=='EFX_LUT4'}
 def n(name,i=None):return ('net',name,i)
 one=('literal',"1'b1",None);zero=('literal',"1'b0",None)
 prefix='camera_stream_observer/';chains=[]
 def check_ff(name,clock,reset,check_ce=True):
  k,p=g[name];assert k=='EFX_FF' and p['CLK']==[n(clock)] and p['SR']==[n(reset)],name
  required=dict(CLK_POLARITY=1,SR_POLARITY=0,SR_SYNC=0,SR_VALUE=0)
  if check_ce:required['CE_POLARITY']=1
  for key,value in required.items():assert ps[name][key]==(1,value),(name,key)
  return p
 for bit in range(3):
  names=[prefix+f'valid_{stage}[{bit}]~FF' for stage in ('meta','sync','settle')]
  for i,name in enumerate(names):
   p=check_ff(name,'process_clk_150','w_arstn');assert p['CE']==[one] and 'async_reg=' in attrs[name][2]
   if i:assert p['D']==g[names[i-1]][1]['Q']
  source=ff[g[names[0]][1]['D'][0]];check_ff(source[0],'i_mipi_rx_pclk','raw_frontend/ingress/n10')
  chains.append(dict(bit=bit,cells=names,source=source[0],end=names[0]+'|D'))
 events=[prefix+'event_meta~FF',prefix+'event_sync~FF']
 for i,name in enumerate(events):
  p=check_ff(name,'process_clk_150','w_arstn');assert p['CE']==[one] and 'async_reg=' in attrs[name][2]
  if i:assert p['D']==g[events[i-1]][1]['Q']
 toggle=ff[g[events[0]][1]['D'][0]];check_ff(toggle[0],'i_mipi_rx_pclk','raw_frontend/ingress/n10')
 chains.append(dict(name='frame_event_toggle',cells=events,source=toggle[0],end=events[0]+'|D',activity_only=True))
 def ev(t,v):
  if t in v:return v[t]
  if t[0]=='literal':assert t in (one,zero);return int(t==one)
  name,p=lut[t];idx=sum(ev(p.get('I'+str(i),[zero])[0],v)<<i for i in range(4));return ps[name]['LUTMASK'][1]>>idx&1
 def leaves(t):
  if t[0]=='literal':return set()
  if t in ff or t not in lut or t in (n('n1345'),n('raw_frontend/ingress/n10')):return {t}
  _,p=lut[t];return set().union(*(leaves(x) for pin,ts in p.items() if pin!='O' for x in ts))
 held=[];checks=0
 for group,bits in enumerate((range(32),range(32,60),range(64,84))):
  endflag=ff[n('camera_stream_faults',92+group)];check_ff(endflag[0],'process_clk_150','w_arstn');assert endflag[1]['D']==[one]
  stable=g[chains[group]['cells'][-1]][1]['Q'][0];latched=endflag[1]['Q'][0]
  for a,b in itertools.product((0,1),repeat=2):assert ev(endflag[1]['CE'][0],{latched:a,stable:b})==((not a) and b);checks+=1
  sourceflag=g[chains[group]['source']][1]['Q'][0]
  event=n('camera_boundary_error') if group==0 else n('raw_stream_errors',7) if group==1 else n('camera_frame_end')
  for bit in bits:
   if bit in (19,20,51,52):assert n('camera_stream_faults',bit) not in ff;continue
   name,p=ff[n('camera_stream_faults',bit)];check_ff(name,'process_clk_150','w_arstn');assert p['CE']==endflag[1]['CE']
   sn,sp=ff[p['D'][0]];check_ff(sn,'i_mipi_rx_pclk','raw_frontend/ingress/n10')
   # The mapper absorbs the combinational frame_end expression into CE.
   # Enumerate its actual leaves, rather than treating that derived event
   # as an independent cutpoint which no longer exists in every path.
   tokens=sorted(leaves(sp['CE'][0])|leaves(event)|{sourceflag},key=str);assert len(tokens)<=10,(sn,tokens)
   for values in itertools.product((0,1),repeat=len(tokens)):
    v=dict(zip(tokens,values));assert ev(sp['CE'][0],v)==((not v[sourceflag]) and ev(event,v)),(sn,v);checks+=1
   held.append(dict(bit=bit,start=sn+'|CLK',end=name+'|D',launch='i_mipi_rx_pclk',capture='process_clk_150'))
 assert len(held)==76
 # All boundary and observer pixel FFs share the real packer release, not
 # the earlier video delay. The four release bits are actual pixel FFs.
 reset=n('raw_frontend/ingress/n10');rn,rp=lut[reset];assert ps[rn]['LUTMASK']==(16,0x8000)
 for i in range(4):
  token=n('raw_frontend/ingress/camera_release',i);assert rp['I'+str(i)]==[token]
  sn,sp=ff[token];assert sp['CLK']==[n('i_mipi_rx_pclk')]
 pixel=[]
 for name,(kind,p) in g.items():
  if kind=='EFX_FF' and (name.startswith(prefix) or name.startswith('camera_recovery_boundary/') or name in ('camera_frame_active~FF','camera_polarity_known~FF','camera_active_vsync_level~FF','camera_boundary_error~FF')) and p['CLK']==[n('i_mipi_rx_pclk')]:
   check_ff(name,'i_mipi_rx_pclk','raw_frontend/ingress/n10',check_ce=False);pixel.append(name)
 assert len(pixel)>70
 return dict(status='CAMERA_RECOVERY_SYNC_HELD_RESET_STRUCTURE_PASS',map_sha256=sha(net),chains=chains,held=held,
  truth_checks=checks,pixel_reset_cells=pixel,held_until_reset=True,auditor_sha256=sha(__file__))

def inventory(board):
 result=prior.inventory(board);structure=map_structure(board);expected={x['end']:x for x in structure['held']}
 first={x['end']:x for x in structure['chains']};rows=[];heads=[]
 report=board/'outflow/ti60f225_oob.camera_pixel_to_system.rpt';body=report.read_text();assert '-npaths 10000 -nworst 1' in body
 paths=body.split('++++ Path ')[1:];assert 0<len(paths)<10000
 for path in paths:
  def field(s):return re.search(r'^'+s+r'\s*:\s*(.+)$',path,re.M)[1].strip()
  end=field('Path End');start=field('Path Start')
  if end in expected:
   assert start==expected[end]['start'] and field('Launch Clock').split()[0]=='i_mipi_rx_pclk' and field('Capture Clock').split()[0]=='process_clk_150'
   delay=float(field('Data Path Delay').split()[0]);assert delay<13.332,(end,delay);rows.append(dict(start=start,end=end,delay_ns=delay))
  elif end in first:
   assert start==first[end]['source']+'|CLK';heads.append(end)
  else:assert not end.startswith('camera_stream_observer/'),('unclassified camera crossing',start,end)
 assert len(rows)==76 and {x['end'] for x in rows}==set(expected)
 assert len(heads)==4 and set(heads)==set(first)
 result['camera_recovery_structure']=structure
 result['camera_recovery_settling']=dict(endpoints=76,max_delay_ns=max(x['delay_ns'] for x in rows),conservative_limit_ns=13.332,
  earliest_capture_receiving_periods=3,first_sync_endpoints=heads,report=str(report),report_sha256=sha(report),paths=rows)
 return result

if __name__=='__main__':
 import argparse
 from pathlib import Path
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 r=map_structure(a.board.resolve());assert not a.out.exists();a.out.write_text(json.dumps(r,indent=2));print(r['status'],len(r['held']),r['truth_checks'])
