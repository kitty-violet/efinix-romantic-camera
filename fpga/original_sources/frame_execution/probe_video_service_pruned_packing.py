"""Capacity-only stop for a native copy with proved unused cells removed."""
import argparse,json,runpy,subprocess,sys
from pathlib import Path
from run_units import HERE,ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native'])
 graft=json.loads((native/'graft.json').read_text());proof=graft['unused_pruning']
 assert sha(proof['proof'])==proof['proof_sha256']
 graph=json.loads((board/'retained-map-audit.json').read_text())
 assert graph['discarded_unloaded_diagnostic_cells']==[]
 subprocess.run([sys.executable,str(HERE/'place_video_service_serial.py'),'--board',str(board),'--root',str(root),'--prepare-only'],check=True,timeout=120)
 driver=root.parent/(root.name+'-wrapper.py');f=driver.with_suffix('.json');m=json.loads(f.read_text())
 assert sha(driver)==m['driver_sha256'];m['argv'].append('--packing-only')
 m.update(probe='pruned_capacity_only_safe_stop_before_placement',capacity_wrapper_sha256=sha(__file__),unused_proof_sha256=proof['proof_sha256'])
 f.write_text(json.dumps(m,indent=2));sys.argv=m['argv'];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
