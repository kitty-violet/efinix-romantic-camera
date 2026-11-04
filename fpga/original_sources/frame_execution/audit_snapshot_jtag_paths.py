"""Audit routed new snapshot crossings, including held-data settling budget."""
import re
from run_units import sha
from audit_snapshot_cells import cells,audit as audit_sync
PREFIX='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'

def audit(root):
 net=root/'outflow/ti60f225_oob.map.v';text=net.read_text();c=cells(text)
 sync=audit_sync(text,PREFIX,'process_clk_150','jtag_inst1_TCK')
 held={n+'|D' for n,(k,p,a) in c.items() if n.startswith(PREFIX+'snapshot[') and k=='EFX_FF'}
 assert len(held)>160
 reports={};rows=[]
 for stem in ('system_to_jtag','jtag_to_system','ddr_to_jtag'):
  p=root/f'outflow/ti60f225_oob.{stem}.rpt'
  if not p.exists():
   log=root/'sta.log';body=log.read_text()
   block=re.search('BEGIN_'+stem+r'\s+(.*?)END_'+stem,body,re.S)
   assert block and 'No timing path is found' in block[1],('Missing report without explicit empty-path evidence',stem)
   reports[str(log)]=sha(log);continue
  reports[str(p)]=sha(p)
  body=p.read_text();assert '-npaths 10000 -nworst 1' in body
  paths=body.split('++++ Path ')[1:];assert len(paths)<10000
  for path in paths:
   def field(name):return re.search(r'^'+name+r'\s*:\s*(.+)$',path,re.M).group(1).strip()
   start,end=field('Path Start'),field('Path End')
   if not end.startswith(PREFIX):continue
   delay=float(field('Data Path Delay').split()[0])
   if end in held:
    assert start.startswith(PREFIX+'held_payload[') and start.endswith('~FF|CLK'),(start,end)
    kind='held_data_after_two_tck_ack_stages'
    # At 10 MHz constraint the acknowledgement waits two synchronizer stages
    # before capture. Require < ONE TCK (100 ns), stricter than two-stage wait.
    assert delay<100,(start,end,delay)
   elif end in (PREFIX+'request_meta~FF|D',PREFIX+'ack_meta~FF|D'):
    kind='toggle_first_sync_ff'
   elif re.fullmatch(re.escape(PREFIX)+r'raw_meta\[\d\]~FF\|D',end):
    kind='raw_status_first_sync_ff'
   else:raise AssertionError(('Unknown new JTAG CDC endpoint',start,end))
   rows.append(dict(start=start,end=end,kind=kind,delay_ns=delay))
 assert {r['end'] for r in rows if r['kind'].startswith('held_data')}==held
 assert {r['end'] for r in rows if r['kind']=='toggle_first_sync_ff'}=={PREFIX+'request_meta~FF|D',PREFIX+'ack_meta~FF|D'}
 return dict(status='ROUTED_SNAPSHOT_CDC_STRUCTURE_AND_SETTLING_PASS',map_sha256=sha(net),
  sync=sync,held_data_endpoints=len(held),max_held_data_delay_ns=max(r['delay_ns'] for r in rows if r['end'] in held),
  conservative_settling_limit_ns=100,tck_constraint_ns=100,unknown_new_endpoints=0,
  reports=reports,endpoints=rows,analogue_metastability_proven=False,board_release=False)
