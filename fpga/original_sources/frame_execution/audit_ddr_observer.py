"""Check new observer synchronizers and classify their routed crossings."""
import re,json
from pathlib import Path
from run_units import HERE,sha
from audit_snapshot_cells import cells
def mapped(net):
 text=Path(net).read_text();c=cells(text);checked=[]
 for i in range(20):
  names=[f'ddr_observer/obs_{s}[{i}]~FF' for s in ('meta','sync')]
  for name in names:
   kind,p,attrs=c[name]
   assert kind=='EFX_FF' and p['CLK']=='process_clk_150' and 'async_reg="true"' in attrs,(name,p)
  assert c[names[1]][1]['D']==c[names[0]][1]['Q'],names
  checked+=names
 # No observation wire can drive any original sequential input; the only
 # original destination is the explicitly audited APB read data mux.
 for name,(kind,ports,attrs) in c.items():
  if kind=='EFX_FF' and not name.startswith('ddr_observer/'):
   assert not any('ddr_observer/' in v for k,v in ports.items() if k!='Q'),name
 return dict(status='DDR_OBSERVER_40_SYNC_FF_PASS',map_sha256=sha(net),synchronizer_ffs=checked,
             clocks_divided_by=256,sticky_bits_monotonic=True,analogue_metastability_proven=False)

def inventory(board):
 # Keep the original endpoint inventory intact; add one tightly scoped class.
 source=(HERE/'snapshot_clock_cdc.py').read_text()
 old="            if re.fullmatch(r'bringup_ddr_meta\\[\\d+\\]~FF\\|D', end):"
 new="""            if end.startswith('ddr_observer/obs_meta['):
                assert re.fullmatch(r'ddr_observer/obs_meta\\[(?:[0-9]|1[0-9])\\]~FF\\|D',end)
                assert capture=='process_clk_150'
                kind='ddr_event_observation_first_stage'
            elif re.fullmatch(r'bringup_ddr_meta\\[\\d+\\]~FF\\|D', end):"""
 assert source.count(old)==1;source=source.replace(old,new)
 old="('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required')"
 assert source.count(old)==1;source=source.replace(old,"('read_only_diagnostic_first_stage','new_global_async_reset_pin_review_required','ddr_event_observation_first_stage')")
 ns={'__name__':'_ddr_observer_inventory'};exec(compile(source,str(HERE/'snapshot_clock_cdc.py'),'exec'),ns)
 result=ns['inventory'](board)
 count=result['classes'].get('ddr_event_observation_first_stage',0);assert 4<=count<=20,count
 result['ddr_observer_sync']=mapped(board/'outflow/ti60f225_oob.map.v')
 result['ddr_observer_crossing_policy']='Four independently divided single-bit clocks, independent monotonic sticky bits and slow diagnostic levels; no multi-bit binary CDC counter, no functional feedback.'
 return result
