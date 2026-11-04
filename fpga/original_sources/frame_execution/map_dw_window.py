"""Map specified window components, keeping rejected candidates immutable."""
import argparse,concurrent.futures,json
from pathlib import Path
from run_units import ROOT,sha
from map_dw_components import component
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tops',nargs='+',default=['dw_window_prefetch']);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda t:component(root,t),a.tops))
 result=dict(status='CANDIDATE_RESOURCE_SCREEN' if all(r['status']!='FAIL' for r in results) else 'HAS_FAILED_COMPONENT',components=results,runner_sha256=sha(__file__))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);assert result['status']=='CANDIDATE_RESOURCE_SCREEN'
if __name__=='__main__':main()
