"""Existing safe capacity stop, before committing time to placement/routing."""
import argparse,json,runpy,subprocess,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 subprocess.run([sys.executable,str(HERE/'place_video_service_epoch.py'),'--board',str(a.board.resolve()),'--root',str(root),'--prepare-only'],check=True,timeout=120)
 driver=root.parent/(root.name+'-wrapper.py');f=driver.with_suffix('.json');meta=json.loads(f.read_text())
 assert sha(driver)==meta['driver_sha256'];meta['argv'].append('--packing-only')
 meta.update(probe='capacity_only_safe_stop_before_placement',capacity_wrapper_sha256=sha(__file__))
 f.write_text(json.dumps(meta,indent=2));sys.argv=meta['argv'];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
