"""Audit actual CNN-subsystem maps; no sums of isolated-module estimates."""
import argparse,collections,json,re
from pathlib import Path
from run_units import ROOT,sha

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
base=ROOT/'tmp/frame-execution'
names=('datapath-map1','production-barrel-diagonal-map1','byte-rotated-map1',
       'onehot-writer-map1','onehot-writer-area-map1')
rows=[]
for name in names:
 folder=base/name;r=json.loads((folder/'result.json').read_text())
 manifest=json.loads((folder/'inputs.json').read_text())
 assert r['kind']=='datapath' and r['arena_external'] is False
 assert sha(folder/'inputs.json')==r['inputs_sha256']
 for f in manifest['inputs']:assert sha(f['snapshot'])==f['sha256']
 net=folder/'outflow/cnn_frame_datapath.map.v'
 report=folder/'outflow/cnn_frame_datapath.map.rpt'
 assert sha(net)==r['map_sha256'] and sha(report)==r['report_sha256']
 counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
 assert counts==r['resources']
 for k,v in counts.items():
  assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',report.read_text(),re.M)==[str(v)]
 assert counts['DSP24']==256 and counts['DSP48']==32
 rows.append(dict(name=name,mode=r['mode'],resources=counts,physical_dsp=160,
  total_ram=counts.get('RAM10',0)+counts.get('DPRAM10',0),lut_capacity_excess=counts['LUT4']-60800,
  result_sha256=sha(folder/'result.json'),netlist_sha256=sha(net)))
best=min(rows,key=lambda r:r['resources']['LUT4'])
assert best['lut_capacity_excess']>0
gates=[]
for name in ('byte-rotated-unit1','compact-dma-abort1','compact-delays-sparse-unit1',
             'resource-combined-unit1','onehot-writer-local1','datapath-active-guard3',
             'datapath-active-fourstate1'):
 folder=base/name;r=json.loads((folder/'result.json').read_text())
 mf=folder/('native_inputs.json' if name=='onehot-writer-local1' else 'inputs.json')
 assert 'PASS' in r['status'] and sha(mf)==r['inputs_sha256']
 manifest=json.loads(mf.read_text())
 if name=='onehot-writer-local1':
  assert sha(manifest['binary'])==manifest['binary_sha256']
  mf=Path(manifest['snapshot'])/'inputs.json'
  assert sha(mf)==manifest['source_manifest_sha256']
  manifest=json.loads(mf.read_text())
 for f in manifest['inputs']:
  if 'snapshot' in f:assert sha(f['snapshot'])==f['sha256']
 gates.append(dict(name=name,status=r['status'],result_sha256=sha(folder/'result.json')))
descriptors=base/'production-descriptors3/result.json'
d=json.loads(descriptors.read_text())
assert d['total_bytes']==8789*64+10440*16==729536
assert d['program_axi_128bit_beats']==45596 and d['average_program_bytes_per_second_at_30fps']==21886080
contract=ROOT/'network_ab/model_b_contract.json'
assert sha(contract)==d['network_contract_sha256']=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
incomplete=[]
for name in ('compact-delays-map1','compact-delays-sparse-map1','resource-combined-map1'):
 folder=base/name;t=json.loads((folder/'termination.json').read_text())
 assert not (folder/'result.json').exists()
 incomplete.append(dict(name=name,status=t['status'],record_sha256=sha(folder/'termination.json')))
result=dict(status='DATAPATH_RESOURCE_ROUND_EVIDENCE_AUDIT_PASS',maps=rows,best_map=best['name'],
 resource_fit=False,minimum_lut_excess=best['lut_capacity_excess'],local_gates=gates,
 incomplete_maps=incomplete,descriptor_result_sha256=sha(descriptors),network_contract_sha256=sha(contract),
 full_frame_reference='datapath-frame1, source-bound to datapath-map1; 4898508 cycles',
 candidate_full_frame=False,production_scheduler=False,final_pnr_sta=False,board_test=False,
 limitations=['Capacity comparison uses LUT count alone as a necessary bound; no XLR packing claim.',
 'Whole baseline peripherals and production command fetch/ownership are not yet in these maps.',
 'Best map and active cancellation test use different documented variants; no mixed evidence claim.'])
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],best_map=result['best_map'],minimum_lut_excess=result['minimum_lut_excess'],resource_fit=False)))
