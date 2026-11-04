"""Map the actual numeric-tested cluster with production external-header mode."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
from map_area_screen import one
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
proof=json.loads((a.proof/'result.json').read_text());manifest=json.loads((a.proof/'inputs.json').read_text())
assert proof['status']=='ROW_PAIR_ACTUAL_MAC_RQ_CLUSTER_PASS' and sha(a.proof/'inputs.json')==proof['inputs_sha256']
sources=[]
for rec in manifest['inputs']:
 f=Path(rec['snapshot']);assert sha(f)==rec['sha256']
 if not f.name.startswith(('efx_','tb_')):sources.append(f)
core=next(f for f in sources if f.name=='row_pair_compute_cluster.sv').read_text()
ports=re.search(r'module row_pair_compute_cluster #\(parameter integer EXTERNAL_HEADER=0\)\((.*?)\n\);',core,re.S)[1]
wrapper='module row_pair_cluster_map('+ports.replace('output reg','output wire')+'\n);\nrow_pair_compute_cluster #(.EXTERNAL_HEADER(1)) core(.*);\nendmodule\n'
generated=r/'row_pair_cluster_map.sv';generated.write_text(wrapper+'\n'.join(f.read_text() for f in sources),encoding='utf-8')
meta=dict(inputs=[dict(snapshot=str(f),sha256=sha(f)) for f in sources],generated_sha256=sha(generated),numeric_proof_sha256=sha(a.proof/'result.json'),runner_sha256=sha(__file__),external_arena=True,external_coefficient_store=True)
(r/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
result=one(r/'map',generated,'speed',budget=900)
for rec in meta['inputs']:assert sha(rec['snapshot'])==rec['sha256']
result.update(status='ACTUAL_ROW_PAIR_CLUSTER_MAP_NOT_BOARD',inputs_sha256=sha(r/'inputs.json'),whole_board=False,external_arena=True,external_coefficient_store=True,physical_xlr=None)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
