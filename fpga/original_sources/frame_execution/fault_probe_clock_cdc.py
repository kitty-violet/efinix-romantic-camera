"""Classify only the new preserved observation synchronizers, retaining all
existing vendor Gray/reset classification and strict unknown-endpoint rejection.
"""
import inspect
from run_units import sha
from audit_snapshot_cells import cells
def map_pairs(text):
 c=cells(text);pairs=[]
 for i in range(6):
  a=f'clock_probe/meta[{i}]~FF';b=f'ddr_clock_diagnostic[{120+i}]~FF'
  for n in (a,b):
   k,p,attrs=c[n]
   assert k=='EFX_FF' and p['CLK']=='process_clk_150' and 'async_reg=' in attrs,(n,k,p)
   assert p['CE']=="1'b1" and p['SR']=="1'b0"
  assert c[b][1]['D']==c[a][1]['Q']
  pairs.append(dict(first=a,second=b))
 return dict(status='FAULT_PROBE_SIX_REAL_SYNC_PAIRS_PASS',pairs=pairs)

def inventory(board):
 import snapshot_clock_cdc as old
 source=inspect.getsource(old.inventory)
 before="if re.fullmatch(r'bringup_ddr_meta\\[\\d+\\]~FF\\|D', end):"
 after="if re.fullmatch(r'clock_probe/meta\\[[0-5]\\]~FF\\|D', end):\n                assert capture == 'process_clk_150'\n                kind = 'clock_probe_first_stage'\n            el"+before
 assert source.count(before)==1;source=source.replace(before,after)
 before="('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required')"
 assert source.count(before)==1
 source=source.replace(before,"('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required','clock_probe_first_stage')")
 scope=dict(old.__dict__);exec(compile(source,old.__file__+':OFP4','exec'),scope)
 result=scope['inventory'](board)
 result['clock_probe_sync']=map_pairs((board/'outflow/ti60f225_oob.map.v').read_text())
 assert result['classes'].get('clock_probe_first_stage',0)==6,result['classes']
 result['observer_auditor_sha256']=sha(__file__)
 return result
