"""Audit selected residual real-port evidence and charge every fused row.
This is an isolated-scheduler cost substitution, never a measured whole frame.
"""
import argparse,collections,json,re
from pathlib import Path
from run_units import HERE,ROOT,sha
def evidence(root,status):
 r=json.loads((root/'result.json').read_text());assert r['status']==status
 manifest=root/'inputs.json';assert sha(manifest)==r['inputs_sha256']
 m=json.loads(manifest.read_text())
 for rec in m['inputs']:assert sha(rec['snapshot'])==rec['sha256']
 for v in (r.get('results',[]) if isinstance(r.get('results'),list) else []):
  body=(root/('seed'+str(v['seed'])+'.log')).read_text()
  assert sha(root/('seed'+str(v['seed'])+'.log'))==v['log_sha256'] and v['line'] in body
  assert not re.search(r'^# (?:FAIL|\*\* (?:Error|Fatal))',body,re.M)
 return r,m
def parse_rows(result):
 return [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',s)} for s in result['results'][0]['rows']]
def main():
 p=argparse.ArgumentParser()
 for name in ('source','profile','map','mapped','native-source','native-map','native-audit'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 src,sm=evidence(a.source,'LOCAL_RESIDUAL_REAL_ARENA_SHARED_POOL_ROW_PASS')
 profile,pm=evidence(a.profile,'LOCAL_RESIDUAL_REAL_ARENA_SHARED_POOL_ROW_PASS')
 mapped,mm=evidence(a.mapped,'LOCAL_RESIDUAL_ROW_OFFICIAL_MAP_NUMERIC_PASS')
 mr,mim=evidence(a.map,'LOCAL_RESIDUAL_ROW_ADAPTER_MAP_PASS')
 source_hash=next(x['sha256'] for x in sm['inputs'] if Path(x['path']).name=='residual_row_shared.sv')
 for m in (mim,):assert next(x['sha256'] for x in m['inputs'] if Path(x['path']).name=='residual_row_shared.sv')==source_hash
 assert mapped['map_sha256']==mr['map_sha256']
 assert 'EFX-0201' not in (a.map/'outflow/residual_row_shared.warn.log').read_text()
 assert mr['resources']['FF']>=2048 and mr['resources'].get('RAM10',0)==0
 ns,nsm=evidence(a.native_source,'LOCAL_RESIDUAL_ROW_NATIVE_SOURCE_PASS')
 nm,nmm=evidence(a.native_map,'LOCAL_RESIDUAL_ROW_NATIVE_MAP_PAIRED_PASS')
 assert next(x['sha256'] for x in nsm['inputs'] if Path(x['path']).name=='residual_row_shared.sv')==source_hash
 assert nmm['map_sha256']==mr['map_sha256']
 wa=json.loads(a.native_audit.read_text());assert wa['status']=='LOCAL_NATIVE_RESIDUAL_WARNING_AUDIT_PASS'
 for f,h in wa['evidence'].items():assert sha(f)==h
 for family in ('full','profile'):
  assert ns['results'][family]['trace']==nm['results'][family]['trace']
  assert nsm['families'][family]['hashes']==nmm['families'][family]['hashes']
  for root,r in ((a.native_source,ns),(a.native_map,nm)):
   assert sha(root/('simulation_'+family+'.log'))==r['results'][family]['log_sha256']
 rows=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',s)} for s in ns['results']['profile']['rows']];measured={}
 for r in rows:
  if r['random']==0 and r['groups']==4:
   assert r['reads']==r['width']*4*3 and r['writes']==r['width']*4*2
   if r['width'] in measured:assert measured[r['width']]==r['cycles']
   measured[r['width']]=r['cycles']
 # The4-state gate is deliberately bounded; full native source/map traces
 # above cover45 shapes plus8 fusion-width cases and all fault/recovery cases.
 assert any(r['width']==80 and r['cycles']==213 for r in parse_rows(mapped))
 assert nsm['families']['profile']['hashes']==pm['vectors']
 trace=ROOT/'tmp/frame-execution/residual-layout-ports1/width80-rows4.json'
 assert sha(trace)==profile['oracle']['trace_sha256']
 counts=collections.Counter()
 for strip in json.loads(trace.read_text()):
  for t in strip['residual_tasks']:
   lo,hi=strip['ranges'][t['node']];counts[hi-lo+1]+=t['hi']-t['lo']+1
 assert sum(counts.values())==2160 and set(counts)<=set(measured)
 charged=sum(n*measured[w] for w,n in counts.items())
 lower_path=ROOT/'output/toolchain/frame_arithmetic_lower_bound_2026-09-30.json'
 lower=json.loads(lower_path.read_text());old_residual=lower['by_kind_cycles']['residual_add']
 replaced=lower['ideal_main_network_and_residual_cycles']-old_residual+charged
 result=dict(status='AUDITED_LOCAL_RESIDUAL_PORT_NUMERIC_MAP_AND_COST_PASS',source_sha256=source_hash,map_sha256=mr['map_sha256'],
  row_cycles=measured,row_counts=dict(counts),isolated_residual_rows=2160,isolated_residual_cycles=charged,isolated_residual_ms_at_150mhz=charged/150000,
  previous_residual_ideal_cycles=old_residual,extra_cycles_over_arithmetic_only=charged-old_residual,
  other_nodes_still_ideal_plus_measured_isolated_residual_ms=replaced/150000,
  remaining_cycles_from_5000000=5000000-replaced,
  measured_frame_ms=None,full_frame_30fps=False,final_resource_fit=False,pnr=False,
  row_adapter_resources=mr['resources'],excluded_not_zero=['DW/stem actual read conflicts and returns','PW integration with same arena and shared pool',
   'all non-residual tile halos and launch/drain','parameter reloads','DDR DMA and competing camera/display/CPU','gesture inference','final DDR B and frame publish'],
  four_state_source=True,four_state_bounded_map=True,full_paired_native=True,native_warning_unknown=0,
  evidence={str(f):sha(f) for f in [trace,lower_path,Path(__file__),a.native_audit,*[x/'result.json' for x in (a.source,a.profile,a.map,a.mapped,a.native_source,a.native_map)]]})
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

