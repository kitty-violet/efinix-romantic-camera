"""Route a copy of the best legal placement against the unchanged real SDC.

The deferred extra 0.300 ns optimization target is removed, never any original
clock uncertainty, relationship, exception, or acceptance requirement.
"""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True)
 a,unused=p.parse_known_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 original=HERE/'reroute_feedback_shift_guard_board.py';source=original.read_text()
 def swap(old,new):
  nonlocal source
  assert source.count(old)==1,old
  source=source.replace(old,new)
 swap("choices=('CONGESTION_1','CONGESTION_3')","choices=('CONGESTION_2',)")
 anchor=" for row in meta['inputs']:\n  if a.recipe"
 change=""" margin=meta['setup_margin'];assert margin['extra_setup_ns']==0.3 and margin['extra_hold_ns']==0
 canonical=Path(margin['canonical_sdc'])
 assert sha(canonical)==margin['canonical_sdc_sha256']==previous['identity']['canonical_sdc']
 extra='\\n# Reserve 0.300ns during optimization; final acceptance is unchanged.\\nset_clock_uncertainty -setup -to [get_clocks {*}] 0.300\\n'
 assert (old/'ti60f225_oob.sdc').read_text()==canonical.read_text()+extra
 shutil.copy2(canonical,physical/'ti60f225_oob.sdc')
 meta['setup_margin']=None
 margin_change=dict(before_extra_ns=0.3,after_extra_ns=0,acceptance_sdc=str(canonical),acceptance_sdc_sha256=sha(canonical),old_optimization_sdc_sha256=sha(old/'ti60f225_oob.sdc'),canonical_acceptance_unchanged=True)
 for row in meta['inputs']:
  if Path(row['snapshot']).name=='ti60f225_oob.sdc':row['sha256']=sha(canonical)
  if a.recipe"""
 swap(anchor,change)
 swap("if not (a.recipe and n=='ti60f225_oob.xml'):","if n not in ('ti60f225_oob.xml','ti60f225_oob.sdc'):")
 swap("rtl_changed=False,map_changed=False,constraints_changed=False,hardware_access=False)","rtl_changed=False,map_changed=False,constraints_changed=True,acceptance_constraints_changed=False,optimization_margin_change=margin_change,hardware_access=False)")
 swap("for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc','efx_pnr_settings.ini','fixed_dsp.placeloc'):","for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','efx_pnr_settings.ini','fixed_dsp.placeloc'):")
 swap("  route_text=(physical/'outflow/ti60f225_oob.route.out').read_text()","  assert sha(physical/'ti60f225_oob.sdc')==margin_change['acceptance_sdc_sha256']\n  route_text=(physical/'outflow/ti60f225_oob.route.out').read_text()")
 compile(source,'canonical_target_driver','exec')
 driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
 driver.write_text(source)
 driver.with_suffix('.json').write_text(json.dumps(dict(original=str(original),original_sha256=sha(original),generator=str(Path(__file__)),generator_sha256=sha(__file__),generated_sha256=sha(driver),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
