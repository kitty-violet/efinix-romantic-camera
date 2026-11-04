"""Freeze a proven CPU adapter payload change into a measured whole board."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared-base',action='store_true');a=p.parse_args()
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
gate=ROOT/'tmp/frame-execution/cpu-payload-gate1';proof=json.loads((gate/'result.json').read_text())
assert proof['status']=='BOARD_CPU_WINDOW_FOUR_SEED_PASS' and proof['full_protocol'] and len(proof['results'])==12
assert sha(gate/'inputs.json')==proof['inputs_sha256']
for x in json.loads((gate/'inputs.json').read_text())['inputs']:assert sha(x['snapshot'])==x['sha256']
for test in proof['results']:
 name=f"seed{test['seed']}-full-protocol.log" if test.get('case')=='full_protocol' else f"seed{test['seed']}-bad{test['bad_mapping']}.log"
 assert sha(gate/name)==test['log_sha256']
candidate=HERE/'variants/fixed_cpu_payload/sapphire_axi32_to_axi128.sv'
assert sha(candidate)==sha(gate/candidate.name)
original=HERE.parent/'ddr_fabric'/candidate.name
meta=json.loads((base/'inputs.json').read_text())
if a.prepared_base:
 assert not (base/'result.json').exists() and meta.get('board_release') is False
 assert meta.get('production_binding') and meta.get('fixed_cluster_proof_sha256')
 binding=meta['production_binding'];assert sha(Path(binding['path'])/'result.json')==binding['result_sha256']
else:
 prior=json.loads((base/'result.json').read_text());assert sha(base/'inputs.json')==prior['inputs_sha256']
root.mkdir();records=[];changed=0
for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
for row in meta['inputs']:
 f=Path(row['snapshot']);assert sha(f)==row['sha256'];dst=root/f.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True)
 source=f
 if f.name==candidate.name:
  assert sha(f)==sha(original),'Base CPU adapter differs from checked reference'
  source=candidate;changed+=1
 shutil.copy2(source,dst);records.append(dict(source=str(source),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
assert changed==1 and sum(x['sha256']!=x['base_sha256'] for x in records)==1
warning_base=meta.get('warning_reference_base',meta['base']) if a.prepared_base else str(base)
assert (Path(warning_base)/'outflow/ti60f225_oob.warn.log').is_file()
meta.update(inputs=records,base=str(base),warning_reference_base=warning_base,base_inputs_sha256=sha(base/'inputs.json'),source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],cpu_payload=dict(proof=str(gate),result_sha256=sha(gate/'result.json'),source_sha256=sha(candidate)),runner_sha256=sha(__file__),board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n');print('CPU_PAYLOAD_BOARD_FROZEN',len(records),flush=True)
