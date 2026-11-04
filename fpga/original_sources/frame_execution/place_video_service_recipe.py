"""Compare official congestion recipes on one frozen map, stopping after place.

No seed sweep, RTL edit, route, or hardware access. The underlying input
hashes and physical gates are unchanged; the actual argv override is saved.
"""
import argparse,json,runpy,subprocess,sys
from pathlib import Path
from run_units import HERE,ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
 p.add_argument('--root',type=Path,required=True)
 p.add_argument('--optimization',choices=('CONGESTION_1','CONGESTION_3'),required=True)
 a=p.parse_args();root=a.root.resolve();board=a.board.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 builder=HERE/'place_video_service_epoch.py'
 subprocess.run([sys.executable,str(builder),'--board',str(board),'--root',str(root),'--prepare-only'],check=True,timeout=90)
 driver=root.parent/(root.name+'-wrapper.py');receipt=driver.with_suffix('.json')
 meta=json.loads(receipt.read_text());original=list(meta['argv']);argv=list(original)
 index=argv.index('--optimization')+1;assert argv[index]=='CONGESTION_2'
 argv[index]=a.optimization
 assert argv[argv.index('--seed')+1]=='3'
 assert meta['ordinary_anchors']==0 and meta['hard_anchors']==401 and meta['host_threads']==4
 meta.update(argv=argv,prepared_argv=original,recipe_adapter_sha256=sha(__file__),
             change='Official congestion recipe only; same map, seed, pins, clocks, hard anchors; stop before route.')
 receipt.write_text(json.dumps(meta,indent=2)+'\n')
 sys.argv=argv;runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
