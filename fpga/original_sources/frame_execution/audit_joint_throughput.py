"""Independent binding of one complete measured frame and matching local gates."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha

def bound_run(folder):
 r=json.loads((folder/'result.json').read_text());m=json.loads((folder/'inputs.json').read_text())
 assert r['status']=='CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS'
 assert r['inputs_sha256']==sha(folder/'inputs.json') and r['log_sha256']==sha(folder/'simulation.log')
 assert sha(m['binary'])==m['binary_sha256']
 for s in m['inputs']:assert sha(s['snapshot'])==s['sha256']
 v=Path(m['vectors']);vm=json.loads((v/'frame_manifest.json').read_text());assert sha(v/'frame_manifest.json')==m['vector_manifest_sha256']
 for f,h in vm['files'].items():assert sha(v/f)==h
 for s in vm.get('generation_sources',[]):assert sha(v/s['snapshot'])==s['sha256']
 assert sum(p['cycles'] for p in r['reasons'])==r['cycles']
 assert sum(sum(s[k] for k in ('input','weight','pool','dispatch','setup_drain')) for s in r['stalls'])==sum(p['cycles'] for p in r['reasons'] if p['id'] in (1,2,3))
 assert int(r['line'][3])==vm['expected_compute_words']*8 and int(r['line'][4])==2764800 and int(r['line'][8],16)==(1<<29)-1
 assert r['commands']==vm['commands'] and r['measured_frame_ms']==r['cycles']/150000
 reference=json.loads((ROOT/'tmp/frame-execution/fourway-full-vectors2/frame_manifest.json').read_text())
 assert vm['nodes']==reference['nodes'] and sha(ROOT/'network_ab/model_b_contract.json')=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 return r,m,vm

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--compute-gate',type=Path,required=True);p.add_argument('--dma-gate',type=Path,required=True)
 p.add_argument('--compute-map',type=Path,required=True);p.add_argument('--dma-map',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--four-state-prefix',type=Path);p.add_argument('--native-prefix',type=Path);a=p.parse_args();assert not a.output.exists()
 r,m,vm=bound_run(a.run);byname={Path(s['snapshot']).name:s['sha256'] for s in m['inputs']};evidence=[]
 for label,folder in [('compute_source',a.compute_gate),('dma_source',a.dma_gate),('compute_map',a.compute_map),('dma_map',a.dma_map)]:
  gate=json.loads((folder/'result.json').read_text());manifest=json.loads((folder/'inputs.json').read_text())
  allowed={'compute_source':'LOCAL_FOURWAY_CLUSTER_PASS','dma_source':'ROW_DMA_OFFICIAL_RAM_SOURCE_PASS',
           'compute_map':'ISOLATED_PARALLEL_CANDIDATE_MAP_REQUIRES_INTEGRATION','dma_map':'ISOLATED_PARALLEL_CANDIDATE_MAP_REQUIRES_INTEGRATION'}
  assert gate['status']==allowed[label],(label,gate['status'])
  assert gate['inputs_sha256']==sha(folder/'inputs.json')
  if label.endswith('_map'):
   top='compute_column_cluster' if label=='compute_map' else 'frame_arena_row_dma'
   for suffix,key in (('.map.v','map_sha256'),('.map.rpt','report_sha256'),('.warn.log','warning_sha256')):
    assert sha(folder/'outflow'/(top+suffix))==gate[key],(label,key)
  elif label=='compute_source':
   for item in gate['results']:
    assert sha(folder/f"seed{item['seed']}.log")==item['log_sha256']
    assert sha(folder/f"readback-{item['seed']}.hex")==item['readback_sha256']
   for f,h in manifest['vectors'].items():assert sha(folder/'vectors'/f)==h
  else:assert sha(folder/'simulation.log')==gate['simulation_sha256']
  for s in manifest['inputs']:
   name=Path(s['snapshot']).name;assert sha(s['snapshot'])==s['sha256']
   if name in byname and not name.startswith('tb_'):assert byname[name]==s['sha256'],(label,name)
  evidence.append(dict(label=label,path=str(folder/'result.json'),sha256=sha(folder/'result.json'),resources=gate.get('resources')))
 path=Path(m['vectors'])/'dependency_audit.json';dep=json.loads(path.read_text())
 assert dep['status']=='FINITE_DDR_DEPENDENCY_COVERAGE_PASS' and dep['vector_manifest_sha256']==m['vector_manifest_sha256']
 prefix=None
 assert bool(a.four_state_prefix)==bool(a.native_prefix)
 if a.four_state_prefix:
  sr=json.loads((a.four_state_prefix/'result.json').read_text());sm=json.loads((a.four_state_prefix/'inputs.json').read_text())
  nr=json.loads((a.native_prefix/'result.json').read_text());nm=json.loads((a.native_prefix/'inputs.json').read_text())
  assert sr['status']=='CONTINUOUS_FRAME_FOUR_STATE_PREFIX_PASS' and nr['status']=='CONTINUOUS_FRAME_SMOKE_ONLY'
  assert sm['original_inputs_sha256']==r['inputs_sha256']
  assert nm['binary_sha256']==m['binary_sha256'] and sm['vector_manifest_sha256']==nm['vector_manifest_sha256']==m['vector_manifest_sha256']
  for folder,result,manifest in ((a.four_state_prefix,sr,sm),(a.native_prefix,nr,nm)):
   assert result['inputs_sha256']==sha(folder/'inputs.json') and result['log_sha256']==sha(folder/'simulation.log')
   for item in manifest['inputs']:assert byname[Path(item['snapshot']).name]==sha(item['snapshot'])==item['sha256']
  matches=re.findall(r'^SMOKE FULL_CHAIN commands=(\d+) cycles=(\d+) checks=(\d+)$',(a.native_prefix/'simulation.log').read_text(),re.M)
  assert matches==[(str(sr['commands']),str(sr['cycles']),str(sr['byte_checks']))]
  assert sr['profile']==nm['profile']==1 and sm['limit']==nm['limit']==sr['commands']
  prefix=dict(commands=sr['commands'],cycles=sr['cycles'],byte_checks=sr['byte_checks'],profile=1,full_frame=False)
  evidence.extend(dict(path=str(folder/'result.json'),sha256=sha(folder/'result.json')) for folder in (a.four_state_prefix,a.native_prefix))
 core=json.loads((a.compute_map/'result.json').read_text())['resources']
 capacity=ROOT/'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5/outflow/ti60f225_oob.place.rpt'
 assert re.search(r'^XLRs:\s*\d+ / 60800 ',capacity.read_text(),re.M)
 parts={0:'idle/commit',1:'PW',2:'DW/stem',3:'PW/residual',5:'unhidden DMA',6:'header/config'}
 result=dict(status='BOUND_JOINT_CONTINUOUS_MEASUREMENT_AUDIT_PASS',cycles=r['cycles'],frame_ms_at_150mhz=r['measured_frame_ms'],
  latency_budget_pass=r['cycles']<=5000000,margin_cycles=5000000-r['cycles'],pixel_checks=921600,intermediate_byte_checks=int(r['line'][3]),
  exclusive=[dict(name=parts.get(x['id'],str(x['id'])),cycles=x['cycles'],ms=x['cycles']/150000) for x in r['reasons']],
  per_node=r['nodes'],stalls=r['stalls'],overlap=r['dma_overlap'],golden_29_nodes_unchanged=True,
  scope='One complete 720p Model-B CNN frame, diagnostic weights, finite actual arena and resident ports, AXI behavioral DDR, precompiled commands driven by TB.',
  same_source_four_state_native_prefix=prefix,arena_read_engines=2,physical_arena_banks=12,physical_ports_per_bank=2,
  isolated_core_lut4=core['LUT4'],device_xlr_capacity=60800,isolated_core_lut_capacity_violation=core['LUT4']>60800,
  resource_boundary='Isolated core/DMA mappings are not additive final XLR utilization; no complete netlist fit or placement has passed.',
  full_system_resource_pass=False,final_sta_pass=False,physical_ddr_pass=False,steady_frame_interval_measured=False,full_baseline_pass=False,
  evidence=[dict(path=str(a.run/f),sha256=sha(a.run/f)) for f in ('inputs.json','result.json','simulation.log')]+evidence+[dict(path=str(path),sha256=sha(path)),dict(path=str(capacity),sha256=sha(capacity))])
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ('status','cycles','frame_ms_at_150mhz','latency_budget_pass','margin_cycles')}))
if __name__=='__main__':main()
