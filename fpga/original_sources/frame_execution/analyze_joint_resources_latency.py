"""Read-only attribution of frozen full-frame evidence; no proposed RTL gains."""
import argparse,collections,json,sys
from pathlib import Path
from run_units import ROOT,sha
from audit_compute_resource_attribution import parse,comb
sys.path.insert(0,str(ROOT/'network_ab'))
from model_contract import load_contract,compile_contract

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 run=ROOT/'tmp/frame-execution/joint-reorder-native1';mapped=ROOT/'tmp/frame-execution/joint-fourway-map3'
 r=json.loads((run/'result.json').read_text());m=json.loads((run/'inputs.json').read_text())
 assert sha(run/'inputs.json')==r['inputs_sha256'] and sha(run/'simulation.log')==r['log_sha256']
 for s in m['inputs']:assert sha(s['snapshot'])==s['sha256']
 mr=json.loads((mapped/'result.json').read_text());assert sha(mapped/'outflow/compute_column_cluster.map.v')==mr['map_sha256']
 rows=parse(mapped/'outflow/compute_column_cluster.res.csv');top=rows[0];children=[x for x in rows if x['depth']==1]
 for k in top['total']:assert top['own'][k]+sum(c['total'][k] for c in children)==top['total'][k]
 for k in ('LUT4','ADD','FF'):assert top['total'][k]==mr['resources'][k]
 contract=ROOT/'network_ab/model_b_contract.json';assert sha(contract)=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 plan=compile_contract(load_contract());byid={n['id']:n for n in plan['nodes']};program=json.loads((Path(m['vectors'])/'program.json').read_text())
 words={x['id']:x['words'] for x in r['nodes']};node_rows=[];executed=0;nominal=plan['macs_per_frame']
 for st,n in zip(r['stalls'],plan['nodes']):
  assert st['node']==n['index']
  actual_node=byid[n['inputs'][0]] if n['kind']=='residual_add' else n
  lanes=6 if n['id']=='rgb_head' else 8
  macs=st['owner_accepts']*lanes;executed+=macs
  expected=words[n['index']]*actual_node['terms'] if n['kind']!='nearest_x2' else 0
  if n['id']=='rgb_head':expected//=2
  assert expected==st['owner_accepts'],(n['id'],expected,st['owner_accepts'])
  measured=sum(st[k] for k in ('input','weight','pool','dispatch','setup_drain'))
  base=actual_node['macs'] if macs else 0
  node_rows.append(dict(node=n['id'],measured_compute_ms=measured/150000,dispatch_ms=st['dispatch']/150000,
    nominal_useful_macs=base,executed_valid_lane_macs=macs,ideal_nominal_mac_ms=base/256/150000,
    active_mac_cycles_lower_bound=macs/256,dispatch_cycles=st['dispatch'],
    input_wait_ms=st['input']/150000,setup_drain_ms=st['setup_drain']/150000,
    compute_window_mac_occupancy=macs/(measured*256) if measured else None,
    dispatch_plus_pool_mac_occupancy_estimate=macs/((st['dispatch']+st['pool'])*256) if st['dispatch'] else None))
 assert sum(x['nominal_useful_macs'] for x in node_rows)==nominal
 # Preserve the measured mutually exclusive ledger. Owner accepts can occur
 # during a partial phase stall; do not subtract all MACs from dispatch-only
 # cycles and label the remainder as a measured, recoverable packing loss.
 partitions=dict(dispatch_cycles=sum(x['dispatch'] for x in r['stalls']),
  input_wait_cycles=sum(x['input'] for x in r['stalls']),weight_wait_cycles=sum(x['weight'] for x in r['stalls']),
  pool_wait_cycles=sum(x['pool'] for x in r['stalls']),setup_drain_cycles=sum(x['setup_drain'] for x in r['stalls']),
  header_cycles=next(x['cycles'] for x in r['reasons'] if x['id']==6),
  unhidden_dma_cycles=next(x['cycles'] for x in r['reasons'] if x['id']==5),
  final_control_cycles=next(x['cycles'] for x in r['reasons'] if x['id']==0))
 assert sum(partitions.values())==r['cycles']
 transfers=collections.defaultdict(lambda:dict(load_words=0,store_words=0,load_payload=0,store_payload=0))
 compute_shapes=collections.Counter()
 for q in program:
  if q['kind']=='compute':compute_shapes[(q['node'],q['rows'])]+=1;continue
  group=transfers[q['group']];d=q['row'];group[q['kind']+'_words']+=d['width']*d['groups'];group[q['kind']+'_payload']+=d['width']*q['channels']
 external_ram=dict(ddr=39,csi=11,raw_bridge=30,video_after_measured_reduction=35,sapphire=15)
 resource_rows=[dict(name=x['name'],**x['total'],comb_estimate=comb(x['total'])) for x in children]
 resource_rows.append(dict(name='top-owned attribution',**top['own'],comb_estimate=comb(top['own'])))
 result=dict(status='FROZEN_RESOURCE_LATENCY_ATTRIBUTION_PASS_NOT_NEW_RTL',frame_ms=r['measured_frame_ms'],cycles=r['cycles'],
  partitions=[dict(name=k,cycles=v,ms=v/150000) for k,v in partitions.items()],
  nominal_macs=nominal,executed_valid_lane_macs=executed,macs_per_clock=256,
  nominal_mac_only_lower_bound_ms=nominal/256/150000,
  repeated_halo_mac_equivalent_ms=(executed-nominal)/256/150000,
  executed_mac_equivalent_ms=executed/256/150000,
  compute_window_minus_executed_mac_equivalent_ms=sum(x['measured_compute_ms'] for x in node_rows)-executed/256/150000,
  mac_equivalent_differences_are_not_guaranteed_recoverable_time=True,
  phase_occupancy_estimate_includes_partial_pool_accepts=True,
  mac_only_bound_is_not_achievable_full_frame_claim=True,
  input_wait_counter_is_symptom_not_bank_cause=True,
  node_rows=node_rows,resource_attribution=sorted(resource_rows,key=lambda x:-x['LUT4']),
  hierarchy_cannot_be_used_as_guaranteed_savings=True,
  conditional_ram=dict(cnn=136,external=external_ram,sum=136+sum(external_ram.values()),
    if_unused_standalone_residual_8_removed=128+sum(external_ram.values()),final_netlist=False),
  transfer_words=dict(transfers),compute_shapes=[dict(node=k[0],rows=k[1],commands=v) for k,v in sorted(compute_shapes.items())],
  evidence={str(f):sha(f) for f in (run/'result.json',run/'inputs.json',mapped/'outflow/compute_column_cluster.res.csv',contract,Path(__file__))})
 a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps(dict(status=result['status'],partitions=result['partitions'],resources=result['resource_attribution'],transfers=result['transfer_words']),ensure_ascii=False))
if __name__=='__main__':main()
