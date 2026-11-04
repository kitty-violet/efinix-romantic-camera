"""Try the documented existing-route input on the best frozen implementation.

No clock/SDC/netlist/placement changes. This is an optimization experiment;
the original constraint signoff remains mandatory.
"""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,EF,sha

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--root',type=Path,required=True)
 a,_=p.parse_known_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 original=HERE/'reroute_feedback_shift_guard_board.py';source=original.read_text()
 def swap(old,new):
  nonlocal source
  assert source.count(old)==1,old
  source=source.replace(old,new)
 swap("choices=('CONGESTION_1','CONGESTION_3')","choices=('CONGESTION_2',)")
 swap("('.place','.gplace','.out.placeloc','.vdb','.map.v','.interface.csv')",
      "('.place','.gplace','.out.placeloc','.vdb','.map.v','.interface.csv','.troutingtraces')")
 swap("'--route']","'--load_route','--route']")
 swap("rtl_changed=False,map_changed=False,constraints_changed=False,hardware_access=False)",
      "rtl_changed=False,map_changed=False,constraints_changed=False,existing_route_requested=True,existing_route_sha256=sha(old/'outflow/ti60f225_oob.troutingtraces'),official_option_doc_sha256=sha(EF/'doc/topics/cli-pnr-opts.html'),hardware_access=False)")
 compile(source,'loaded_route_driver','exec')
 driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
 driver.write_text(source)
 driver.with_suffix('.json').write_text(json.dumps(dict(original=str(original),original_sha256=sha(original),generator=str(Path(__file__)),generator_sha256=sha(__file__),generated_sha256=sha(driver),hardware_access=False),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
