"""Actual compact capture structure and finite routed held-data settling."""
import re,functools
from run_units import sha
from audit_snapshot_cells import cells,audit as audit_sync
from probe_full_native_split_map import params
from native_local_drivers import CELL,graph
from video_service_sample_ram import storage_structure
P='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'

@functools.lru_cache(maxsize=1)
def parsed(text):
 return cells(text),params(text)

@functools.lru_cache(maxsize=1)
def structure(text):
 c,ps=parsed(text)
 fragments=[m[0] for m in CELL.finditer(text) if m[2].lstrip('\\').startswith((P,'service_serializer/storage/','service_pixel_snapshot/storage/'))]
 # The unchanged sync checker references only its exact 20 named FFs.
 # Reusing their full primitive declarations avoids parsing 61k cells again.
 sync=audit_sync('\n'.join(fragments),P,'process_clk_150','jtag_inst1_TCK')
 assert not any(n.startswith((P+'held_payload[',P+'snapshot[',P+'shifter[')) for n in c)
 if any(n.startswith('service_serializer/storage/banks[') for n in c):
  ramtext='\n'.join(s for s in fragments if s.lstrip().startswith('EFX_RAM10'))
  storage=storage_structure(graph(ramtext),ps)
  for r in storage:
   if r['owner']!='service_serializer':continue
   assert r['write_enable']==('net',P+'n5',None)
   assert r['read_enable']==('net',P+'n2000',None)
  assert c[P+'acknowledge~FF'][1]['CE']==P+'n5'
  return dict(sync=sync,pairs=[],held_data_endpoints=0,dualclock_sample_ram=storage,
   transfer='Fixed-address dual-clock RAM; source write and acknowledge share capture edge; read enable follows two acknowledgement FFs. Protocol equivalence is a separate mandatory test.')
 held={n for n,(k,p,a) in c.items() if k=='EFX_FF' and n.startswith('service_serializer/held[')}
 published={n for n,(k,p,a) in c.items() if k=='EFX_FF' and n.startswith('service_serializer/published[')}
 assert len(held)==len(published)==147
 pairs=[]
 for n in sorted(published):
  s=n.replace('/published[','/held[');assert s in held
  sp,ep=c[s][1],c[n][1]
  assert sp['CLK']=='process_clk_150' and ep['CLK']=='jtag_inst1_TCK'
  assert sp['CE']==P+'n5' and ep['CE']==P+'n2000'
  assert sp['SR']==ep['SR']=="1'b0" and ep['D']==sp['Q']
  for x in (s,n):
   for key in ('CLK_POLARITY','CE_POLARITY','SR_POLARITY'):assert ps[x][key]==(1,1),(x,key,ps[x])
  assert ps[n]['D_POLARITY']==(1,1)
  pairs.append(dict(start=s+'|CLK',end=n+'|D'))
 # Sampling and acknowledgement are driven on the same source edge.
 assert c[P+'acknowledge~FF'][1]['CE']==P+'n5'
 return dict(sync=sync,pairs=pairs,held_data_endpoints=len(pairs))

def audit(root):
 path=root/'outflow/ti60f225_oob.map.v';proof=structure(path.read_text());expected={r['end']:r['start'] for r in proof['pairs']}
 reports={};rows=[];seen=set()
 for stem in ('system_to_jtag','jtag_to_system','ddr_to_jtag'):
  p=root/f'outflow/ti60f225_oob.{stem}.rpt'
  if not p.exists():
   log=root/'sta.log';block=re.search('BEGIN_'+stem+r'\s+(.*?)END_'+stem,log.read_text(),re.S)
   assert block and 'No timing path is found' in block[1],('missing report',stem)
   reports[str(log)]=sha(log);continue
  reports[str(p)]=sha(p);body=p.read_text();assert '-npaths 10000 -nworst 1' in body
  blocks=body.split('++++ Path ')[1:];assert len(blocks)<10000
  for b in blocks:
   def f(label):
    m=re.search(r'^'+re.escape(label)+r'\s*:\s*(.+)$',b,re.M);assert m,label
    return m[1].strip()
   start,end=f('Path Start'),f('Path End')
   if not end.startswith((P,'service_serializer/')) and end!='service_serial_tdo~FF|D':continue
   assert end not in seen,('duplicate',end);seen.add(end)
   delay=float(f('Data Path Delay').split()[0]);charged=None
   if end in expected:
    assert start==expected[end]
    assert f('Launch Clock').split()[0]=='process_clk_150' and f('Capture Clock').split()[0]=='jtag_inst1_TCK'
    charged=delay+abs(float(f('Launch Clock Path Delay'))-float(f('+ Capture Clock Path Delay')))+abs(float(f('- Clock Uncertainty')))+abs(float(f('- Setup')))+.010
    assert charged<100,(start,end,charged)
    kind='compact_held_data_after_two_ack_stages'
   elif end in (P+'request_meta~FF|D',P+'ack_meta~FF|D'):kind='toggle_first_sync_ff'
   elif re.fullmatch(re.escape(P)+r'raw_meta\[\d\]~FF\|D',end):kind='raw_status_first_sync_ff'
   else:raise AssertionError(('unknown compact JTAG crossing',start,end))
   rows.append(dict(start=start,end=end,kind=kind,delay_ns=delay,charged_ns=charged))
 assert {r['end'] for r in rows if r['kind'].startswith('compact_held')}==set(expected)
 assert {r['end'] for r in rows if r['kind']=='toggle_first_sync_ff'}=={P+'request_meta~FF|D',P+'ack_meta~FF|D'}
 return dict(status='COMPACT_JTAG_ROUTED_STRUCTURE_AND_SETTLING_PASS',map_sha256=sha(path),structure=proof,held_data_endpoints=len(expected),max_held_charged_ns=max((r['charged_ns'] for r in rows if r['charged_ns'] is not None),default=None),conservative_settling_limit_ns=100 if expected else None,tck_constraint_ns=100,endpoints=rows,reports=reports,checker_sha256=sha(__file__),analogue_metastability_proven=False,board_release=False)
