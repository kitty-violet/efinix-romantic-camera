"""Resource map for fixed plane/port window prefetch, not the rejected crossbar."""
import argparse,json
from pathlib import Path
from run_units import HERE,ROOT,sha
import map_dw_components as base
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 base.HERE=HERE/'variants/window_paired';r=base.component(root,'dw_window_prefetch');r['runner_sha256']=sha(__file__)
 (root/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);assert r['status']!='FAIL'
if __name__=='__main__':main()
