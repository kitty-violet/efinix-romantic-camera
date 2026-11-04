"""Bind the measured complete frame, its real RTL map, and compositional gates."""
import json
from pathlib import Path
from run_units import ROOT,sha
from audit_joint_throughput import bound_run
runroot=ROOT/'tmp/frame-execution/production-multiissue-frame1'
r,m,vm=bound_run(runroot);names={Path(x['snapshot']).name:x['sha256'] for x in m['inputs']}
evidence=[]
def artifact(folder,filename='result.json'):
 p=ROOT/'tmp/frame-execution'/folder/filename
 evidence.append(dict(path=str(p),sha256=sha(p)));return json.loads(p.read_text())
for folder,top in [('production-multiissue-map1','compute_column_cluster'),('production-dma-map2','frame_arena_row_dma')]:
 result=artifact(folder);manifest=artifact(folder,'inputs.json');base=ROOT/'tmp/frame-execution'/folder
 assert result['inputs_sha256']==sha(base/'inputs.json')
 for ext,key in [('.map.v','map_sha256'),('.map.rpt','report_sha256'),('.warn.log','warning_sha256')]:
  assert sha(base/'outflow'/(top+ext))==result[key]
 for item in manifest['inputs']:assert sha(item['snapshot'])==item['sha256']==names[Path(item['snapshot']).name]
 if top=='compute_column_cluster':core=result['resources']
native=artifact('production-multiissue-local1');nm=artifact('production-multiissue-local1','native_inputs.json');ns=Path(nm['snapshot'])
assert native['inputs_sha256']==sha(ROOT/'tmp/frame-execution/production-multiissue-local1/native_inputs.json')
assert native['log_sha256']==sha(ROOT/'tmp/frame-execution/production-multiissue-local1/simulation.log')
assert sha(ns/'inputs.json')==nm['source_manifest_sha256'] and sha(nm['binary'])==nm['binary_sha256']
for item in json.loads((ns/'inputs.json').read_text())['inputs']:
 assert sha(item['snapshot'])==item['sha256']
 name=Path(item['snapshot']).name
 if name in names and not name.startswith('tb_'):assert item['sha256']==names[name]
reader=artifact('production-multiissue-fourstate2');rm=artifact('production-multiissue-fourstate2','inputs.json')
assert reader['inputs_sha256']==sha(ROOT/'tmp/frame-execution/production-multiissue-fourstate2/inputs.json')
assert reader['log_sha256']==sha(ROOT/'tmp/frame-execution/production-multiissue-fourstate2/simulation.log')
for item in rm['inputs']:
 assert sha(item['snapshot'])==item['snapshot_sha256']
 if Path(item['snapshot']).name=='unified_column_reader.sv':assert item['snapshot_sha256']==names['unified_column_reader.sv']
dma=artifact('production-dma-source3');dm=artifact('production-dma-source3','inputs.json')
assert dma['inputs_sha256']==sha(ROOT/'tmp/frame-execution/production-dma-source3/inputs.json')
for item in dm['inputs']:
 assert sha(item['snapshot'])==item['sha256']
 name=Path(item['snapshot']).name
 if name in names and not name.startswith('tb_'):assert item['sha256']==names[name]
for folder in ['production-dispatch-factored1','production-factored-source1','production-shallow-fifo1']:
 artifact(folder)
capacity=ROOT/'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5/outflow/ti60f225_oob.place.rpt'
assert '/ 60800 ' in capacity.read_text()
assert r['frame_30fps_measured'] and r['cycles']==4899367 and sum(x['cycles'] for x in r['reasons'])==r['cycles']
result=dict(status='BOUND_PRODUCTION_FRAME_AUDIT_PASS',cycles=r['cycles'],ms_at_target_150mhz=r['measured_frame_ms'],margin_cycles=5000000-r['cycles'],margin_ms=(5000000-r['cycles'])/150000,
 intermediate_bytes=int(r['line'][3]),final_bytes=int(r['line'][4]),nodes=29,exclusive_cycles=r['reasons'],
 core_resources=core,ideal_frame_cycle_budget_pass=True,full_resource_fit=False,final_sta=False,board_test=False,
 scope='Continuous actual CNN RTL data chain, finite arena and resident, diagnostic weights, precompiled TB scheduler, ideal single-client AXI behavioral DDR. Not complete baseline or sustained real-DDR FPS.',
 local_native=native['line'],reader_four_state=reader['line'],
 four_state_boundary='New reader packet-equivalence plus existing dispatcher/FIFO four-state gates; old factored full-cluster four-state gate has serial reader. New complete cluster/frame is native, not complete four-state.',
 source_evidence=[dict(path=str(runroot/x),sha256=sha(runroot/x)) for x in ('result.json','inputs.json','simulation.log')],evidence=evidence)
out=ROOT/'output/toolchain/production_frame_first30fps_verified_2026-10-01.json'
assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','cycles','ms_at_target_150mhz','margin_cycles','full_resource_fit')}))
