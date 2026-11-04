"""Native dual-clock RAM and unchanged two-FF OFP5 handshake checks.

The old 960-bit held-bus-to-TCK FF crossing no longer exists. Payload transfer
uses the RAM write/read clocks; handshake publication occurs only after the
final inactive-bank write. Original acceptance SDC is never modified here.
"""
import re
from run_units import sha
from audit_snapshot_cells import cells,audit as audit_sync
from audit_local_reset_implementation import norm
PREFIX='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'

def structure(board):
 net=board/'outflow/ti60f225_oob.map.v';text=net.read_text();cc=cells(text)
 params={}
 for n,k,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text,re.M):
  params.setdefault(norm(n),{})[k]=v.strip()
 sync=audit_sync(text,PREFIX,'process_clk_150','jtag_inst1_TCK')
 memories={n:(k,p,a) for n,(k,p,a) in cc.items() if n.startswith(PREFIX) and k in ('EFX_RAM10','EFX_DPRAM10')}
 assert len(memories)==1,memories
 name,(kind,p,attrs)=next(iter(memories.items()))
 assert kind=='EFX_RAM10'
 expected=dict(READ_WIDTH='1',WRITE_WIDTH='16',OUTPUT_REG="1'b0",WRITE_MODE='"READ_UNKNOWN"')
 for key,value in expected.items():
  actual=params[name][key]
  assert actual==value or (key=='OUTPUT_REG' and actual=='0'),(key,actual,value)
 initial={k:v for k,v in params[name].items() if k.startswith('INIT_')}
 assert len(initial)==40 and all(re.fullmatch(r"256'b0{256}",v) for v in initial.values())
 assert p['WCLK']=='process_clk_150' and p['RCLK']=='jtag_inst1_TCK'
 for pin in ('RST',):assert p[pin]=="1'b0",(pin,p[pin])
 for pin in ('WADDREN','RADDREN'):assert p[pin]=="1'b1",(pin,p[pin])
 assert not any(n.startswith((PREFIX+'snapshot[',PREFIX+'shifter[')) for n in cc)
 registers={n:p for n,(k,p,a) in cc.items() if n.startswith(PREFIX) and k=='EFX_FF'}
 assert len(registers)<1800,(len(registers),'Wide duplicate stores remain')
 assert any(n.startswith(PREFIX+'held_payload[') for n in registers)
 synchronous_clears=[]
 for n,ports in registers.items():
  if ports.get('SR') in ("1'b0","1'b1"):
   assert ports['SR']!=params[n]['SR_POLARITY'],(n,'Constant asserted reset')
  else:
   # Synthesis may implement a local synchronous data clear with the FF SR
   # pin. That is distinct from adding an asynchronous functional reset.
   assert params[n]['SR_SYNC']=="1'b1",(n,ports,params[n])
   synchronous_clears.append(dict(name=n,signal=ports['SR']))
 assert registers[PREFIX+'acknowledge~FF']['CLK']=='process_clk_150'
 assert registers[PREFIX+'scan_bank~FF']['CLK']=='jtag_inst1_TCK'
 return dict(status='OFP5_NATIVE_DUAL_CLOCK_RAM_AND_SYNC_STRUCTURE_PASS',map_sha256=sha(net),
  memory=dict(name=name,kind=kind,pins=p,parameters=params[name]),sync=sync,observer_ff=len(registers),
  payload_cdc='EFX_RAM10 native write/read clock ports; bank publication after sixty writes',
  duplicate_snapshot_and_shifter_removed=True,ram_initialized_zero=True,
  local_synchronous_clear_pins=synchronous_clears,no_asynchronous_functional_reset=True,auditor_sha256=sha(__file__),
  physical_timing_pass=False,hardware_access=False)

def audit(root):
 s=structure(root);reports={};rows=[]
 allowed={PREFIX+'request_meta~FF|D',PREFIX+'ack_meta~FF|D'}
 for stem in ('system_to_jtag','jtag_to_system','ddr_to_jtag'):
  p=root/f'outflow/ti60f225_oob.{stem}.rpt'
  if not p.exists():
   log=root/'sta.log';body=log.read_text();block=re.search('BEGIN_'+stem+r'\s+(.*?)END_'+stem,body,re.S)
   assert block and 'No timing path is found' in block[1];reports[str(log)]=sha(log);continue
  body=p.read_text();assert '-npaths 10000 -nworst 1' in body
  reports[str(p)]=sha(p);paths=body.split('++++ Path ')[1:];assert len(paths)<10000
  for path in paths:
   def field(n):return re.search(r'^'+n+r'\s*:\s*(.+)$',path,re.M)[1].strip()
   start,end=field('Path Start'),field('Path End')
   if not end.startswith(PREFIX):continue
   if end in allowed:kind='toggle_first_sync_ff'
   elif re.fullmatch(re.escape(PREFIX)+r'raw_meta\[\d\]~FF\|D',end):kind='raw_status_first_sync_ff'
   else:raise AssertionError(('Unknown new RAM snapshot CDC endpoint',start,end))
   rows.append(dict(start=start,end=end,kind=kind,delay_ns=float(field('Data Path Delay').split()[0])))
 assert {r['end'] for r in rows if r['kind']=='toggle_first_sync_ff'}==allowed
 return dict(status='ROUTED_RAM_SNAPSHOT_CDC_STRUCTURE_PASS',map_sha256=s['map_sha256'],
  structure=s,sync=s['sync'],held_data_endpoints=0,max_held_data_delay_ns=None,
  unknown_new_endpoints=0,reports=reports,endpoints=rows,analogue_metastability_proven=False,
  bank_ownership_proof='Bound four-state shorts and actual-map serial evidence required by release gate',board_release=False)
