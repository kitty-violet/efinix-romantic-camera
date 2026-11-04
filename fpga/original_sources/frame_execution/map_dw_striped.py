"""Map fixed-word mux/striped candidates concurrently; resource evidence only."""
import argparse,concurrent.futures,json
from run_units import HERE,ROOT,sha
import map_dw_components as base
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 base.HERE=HERE/'variants/striped_words'
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda t:base.component(root,t),['dw_block_plan32','dw_bank_read32','dw_coefficient_registers']))
 result=dict(status='CANDIDATE_RESOURCE_SCREEN' if all(r['status']!='FAIL' for r in results) else 'HAS_FAILED_COMPONENT',components=results,runner_sha256=sha(__file__))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);assert result['status']=='CANDIDATE_RESOURCE_SCREEN'
if __name__=='__main__':main()
