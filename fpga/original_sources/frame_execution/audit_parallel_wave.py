"""Bind the parallel optimization wave to frozen source, simulation and map evidence."""
import json,re,hashlib
from pathlib import Path
from run_units import ROOT,HERE,sha
ROOTS=ROOT/'tmp/frame-execution'
PAIRS=[
 ('fifo80-source1','fifo80-map1'),
 ('stride24-lockstep2','parallel-stridelock-map1'),
 ('header-prefetch5','parallel-header-map2'),
 ('header-prefetch5','parallel-transport-map1'),
 ('mixed-fifo80-source1','parallel-mixedcompact-map1')]
def load(name):
 p=ROOTS/name;r=json.loads((p/'result.json').read_text());m=json.loads((p/'inputs.json').read_text())
 assert sha(p/'inputs.json')==r['inputs_sha256']
 for f in m['inputs']:assert sha(f['snapshot'])==f['sha256'],f
 for n,h in m.get('vectors',{}).items():assert sha(p/'vectors'/n)==h
 return p,r,m
def main():
 rows=[]
 for sim_name,map_name in PAIRS:
  sp,s,sm=load(sim_name);mp,m,mm=load(map_name)
  source={Path(r['snapshot']).name:r['sha256'] for r in sm['inputs']}
  for r in mm['inputs']:assert source[Path(r['snapshot']).name]==r['sha256'],r
  for r in s.get('results',[]):
   log=sp/('seed'+str(r['seed'])+'.log');assert sha(log)==r['log_sha256']
  if 'log_sha256' in s:assert sha(sp/'simulation.log')==s['log_sha256']
  nets=list((mp/'outflow').glob('*.map.v'));assert len(nets)==1 and sha(nets[0])==m['map_sha256']
  warns=list((mp/'outflow').glob('*.warn.log'));assert len(warns)==1 and sha(warns[0])==m['warning_sha256']
  body=warns[0].read_text();codes=set(re.findall(r'\[(EFX-\d+) (?:WARNING|VERI-WARNING)\]',body))
  allowed={'EFX-0034','EFX-0200','EFX-0657','EFX-0666','EFX-0011'}
  assert not(codes-allowed),codes
  if 'EFX-0011' in codes:
   if map_name=='parallel-stridelock-map1':assert 'pool_ready[31] is unconnected' in body and 'Found 8 warnings' in body
   elif map_name=='parallel-transport-map1':assert 'Net gnd is dangling' in body and 'Net vcc is dangling' in body and 'Found 2 warnings' in body
   else:raise AssertionError((map_name,body))
  if 'EFX-0666' in codes:assert "logic multiplier 'mult_34'" in body and 'requantize_i32_mul18_q17_pipeline.sv:68' in body
  if 'EFX-0657' in codes:assert "'q_column' (8 bits)" in body and "'q_index' (52 bits)" in body
  resources=m['resources'];rows.append(dict(sim=sim_name,map=map_name,sim_manifest_sha256=sha(sp/'inputs.json'),map_manifest_sha256=sha(mp/'inputs.json'),
   map_sha256=m['map_sha256'],resources=resources,comb_statistics=sum(resources.get(k,0) for k in ('LUT4','ADD','SRL8')),warnings=sorted(codes)))
 faultp,fault,fm=load('mixed-fifo80-fault1')
 mixedp,mixed,mixedm=load('mixed-fifo80-source1')
 mixed_sources={Path(f['snapshot']).name:f['sha256'] for f in mixedm['inputs']}
 for f in fm['inputs']:
  n=Path(f['snapshot']).name
  if n in mixed_sources:assert mixed_sources[n]==f['sha256']
 assert sha(faultp/'simulation.log')==fault['log_sha256']
 cluster=rows[0];assert cluster['resources']['RAM10']==24
 mixed_res=rows[4]['resources'];assert mixed_res['RAM10']==16 and mixed_res['DSP24']==256 and mixed_res['DSP48']==32
 assert sha(ROOT/'network_ab/model_b_contract.json')=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 # One old runner inherited a stale RAM count. Never use it as mapped resource evidence.
 assert json.loads((ROOTS/'fifo80-source1/result.json').read_text())['official_ram_count']==140
 r=dict(status='PARALLEL_WAVE_LOCAL_SOURCE_MAP_BINDING_PASS',pairs=rows,
  fifo80_corrected_local_ram=24+84+20,old_peripherals_conditional_ram=24+84+20+39+11+30+44+15,
  stale_metadata_ignored='fifo80-source1 official_ram_count=140 is an inherited constant; exact common RTL/map proves local128.',
  compact_mixed_fifo_preserves_eight_ram_saving=True,
  stride_consumer_only_weighted_cycles=480*208+240*2*113,
  stride_consumer_only_weighted_ms=(480*208+240*2*113)/150000,
  forbidden_inference='Consumer timing excludes source preparation/scatter, coefficient prefetch and output writer; do not replace whole-frame cost with this sum.',
  measured_frame_ms=None,full_frame_30fps=False,final_fit=False,pnr=False,board_download=False,
  inputs_sha256=sha(__file__))
 out=ROOT/'output/toolchain/parallel_optimization_wave_2026-09-30_v2.json'
 assert not out.exists();out.write_text(json.dumps(r,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in r.items() if k!='pairs'}))
if __name__=='__main__':main()

