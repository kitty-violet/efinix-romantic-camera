"""Audit OFP5's finite held-data crossing on the actual map and routed paths.

No acceptance constraint is changed. Unknown endpoints remain hard failures.
"""
import inspect,itertools,re
from run_units import sha
from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm
from audit_startup_cause_map import extra_map
from fault_probe_clock_cdc import map_pairs

def map_structure(board):
 net=board/'outflow/ti60f225_oob.map.v';text=net.read_text();cc=cells(text)
 params={}
 for n,k,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text,re.M):
  params.setdefault(norm(n),{})[k]=v.strip()
 def number(value):
  m=re.fullmatch(r"\d+'([hb])([0-9a-fA-F]+)",value);assert m,value
  return int(m[2],16 if m[1]=='h' else 2)
 structure=extra_map(board)
 held={n+'|D':p for n,(k,p,a) in cc.items()
       if re.fullmatch(r'cell_diagnostic\[(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\]~FF',n) and k=='EFX_FF'}
 assert 40<=len(held)<=64,len(held)
 ce={p['CE'] for p in held.values()};assert len(ce)==1
 assert all(p['CLK']=='process_clk_150' and p['SR']=='w_arstn' for p in held.values())
 latch=cc['cell_probe/latched~FF'][1]
 assert latch['D']=="1'b1" and latch['CE'] in ce and latch['SR']=='w_arstn'
 drivers=[(n,p) for n,(k,p,a) in cc.items() if k=='EFX_LUT4' and p.get('O') in ce]
 assert len(drivers)==1
 name,p=drivers[0];mask=number(params[name]['LUTMASK'])
 inputs=['cell_probe/latched']+[f'cell_probe/done_sync[{i}]' for i in range(3)]
 assert {p[f'I{i}'] for i in range(4)}==set(inputs)
 for bits in itertools.product((0,1),repeat=4):
  values=dict(zip(inputs,bits));index=sum(values[p[f'I{i}']]<<i for i in range(4))
  assert (mask>>index&1)==int(not bits[0] and all(bits[1:])),bits
 reset_pins=[]
 for chain in structure['reset_release_pairs']:
  for name in (chain['first'],chain['second']):
   p=cc[name][1]
   assert p['SR']=='w_arstn' and p['CE']=="1'b1"
   assert params[name]['SR_SYNC']=="1'b0" and params[name]['SR_POLARITY']=="1'b0"
   if chain['clock']=='sdram_clk':reset_pins.append(name+'|SR')
 assert len(reset_pins)==2
 return dict(status='OFP5_HELD_DATA_GATE_AND_RESET_CHAINS_MAPPED_PASS',
             held_endpoints=sorted(held),reset_endpoints=reset_pins,
             done_endpoint=structure['done_sync_chain'][0]+'|D',
             three_done_stages=structure['done_sync_chain'],ce_truth_checks=16,
             ce_cell=drivers[0][0],map_sha256=sha(net),physical_cdc_pass=False)

def inventory(board):
 import snapshot_clock_cdc as old
 structure=map_structure(board)
 source=inspect.getsource(old.inventory)
 before="if re.fullmatch(r'bringup_ddr_meta\\[\\d+\\]~FF\\|D', end):"
 after="""if end == structure['done_endpoint']:
                assert launch == 'sdram_clk' and capture == 'process_clk_150'
                kind = 'cell_probe_done_first_stage'
            elif end in structure['held_endpoints']:
                assert launch == 'sdram_clk' and capture == 'process_clk_150'
                assert start.startswith('cell_probe/ddr_'), (start,end)
                kind = 'cell_probe_held_result'
            elif end in structure['reset_endpoints']:
                assert launch == 'process_clk_150' and capture == 'sdram_clk'
                assert start == 'startup_reset/released~FF|CLK', (start,end)
                kind = 'cell_probe_async_reset_release_pair'
            elif re.fullmatch(r'clock_probe/meta\\[[0-5]\\]~FF\\|D', end):
                assert capture == 'process_clk_150'
                kind = 'clock_probe_first_stage'
            el"""+before
 assert source.count(before)==1;source=source.replace(before,after)
 before="('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required')"
 after="('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required','clock_probe_first_stage','cell_probe_done_first_stage','cell_probe_held_result','cell_probe_async_reset_release_pair')"
 assert source.count(before)==1;source=source.replace(before,after)
 scope=dict(old.__dict__,structure=structure);exec(compile(source,old.__file__+':OFP5','exec'),scope)
 result=scope['inventory'](board);counts=result['classes']
 assert counts.get('cell_probe_done_first_stage')==1
 assert counts.get('cell_probe_async_reset_release_pair')==2
 assert counts.get('clock_probe_first_stage')==6
 # All nonconstant DDR payload endpoints must be present in the restored-path
 # report. Constants may have held FFs but cannot have a launched CDC path.
 cc=cells((board/'outflow/ti60f225_oob.map.v').read_text())
 expected={end for end in structure['held_endpoints'] if cc[end[:-2]][1]['D'] not in ("1'b0","1'b1")}
 actual={r['end'] for r in result['endpoints'] if r['kind']=='cell_probe_held_result'}
 assert actual==expected,(actual^expected)
 delays=[]
 for record in result['reports']:
  from pathlib import Path
  for path in Path(record['path']).read_text().split('++++ Path ')[1:]:
   m=re.search(r'^Path End\s*:\s*(.+)$',path,re.M);assert m
   if m[1].strip() not in actual:continue
   delay=float(re.search(r'^Data Path Delay\s*:\s*([\d.]+)',path,re.M)[1])
   # Earliest latch occurs three receiving periods after the source's final
   # update. Demand < two periods, retaining one period for margin/skew.
   assert delay<13.332,(m[1],delay)
   delays.append(delay)
 assert len(delays)==len(actual) and delays
 result.update(cell_probe_structure=structure,cell_probe_bundled_settling=dict(
  endpoints=len(actual),max_delay_ns=max(delays),conservative_limit_ns=13.332,
  stable_until_reset=True,earliest_capture_receiving_periods=3),
  clock_probe_sync=map_pairs((board/'outflow/ti60f225_oob.map.v').read_text()),
  observer_auditor_sha256=sha(__file__))
 return result
