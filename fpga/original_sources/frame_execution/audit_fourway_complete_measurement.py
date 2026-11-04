"""Bind continuous RTL result to exact sources, local gates and vector evidence."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists();run=a.run.resolve()
 r=json.loads((run/'result.json').read_text());m=json.loads((run/'inputs.json').read_text())
 assert r['status']=='CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS'
 assert r['inputs_sha256']==sha(run/'inputs.json') and r['log_sha256']==sha(run/'simulation.log')
 assert sha(m['binary'])==m['binary_sha256']
 tb_compatibility=[]
 for source in m['inputs']:
  assert sha(source['snapshot'])==source['sha256'],source
  if sha(source['path'])!=source['sha256']:
   assert Path(source['path']).name=='tb_fourway_full.sv'
   declaration=' integer final_gold_offset=0,nread=0,sum_reason=0;\n'
   old=Path(source['snapshot']).read_text();new=Path(source['path']).read_text()
   assert old.count(declaration)==new.count(declaration)==1
   assert old.replace(declaration,'')==new.replace(declaration,'')
   tb_compatibility.append('Only testbench variable declaration moved before task for ModelSim15; identical RTL and executable statements.')
 vec=Path(m['vectors']);vm=json.loads((vec/'frame_manifest.json').read_text())
 assert sha(vec/'frame_manifest.json')==m['vector_manifest_sha256']
 for n,h in vm['files'].items():assert sha(vec/n)==h
 for p,h in vm['evidence'].items():assert sha(p)==h
 assert sha(ROOT/'network_ab/model_b_contract.json')=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 gates=[]
 for sub in ('fourway-replay1','fourway-source9','fourway-rowdma2','fourway-full-source2'):
  path=ROOT/'tmp/frame-execution'/sub/'result.json';g=json.loads(path.read_text());assert 'PASS' in g['status'];gates.append(dict(path=str(path),sha256=sha(path),status=g['status']))
 byname={Path(s['snapshot']).name:s['sha256'] for s in m['inputs']}
 for sub in ('fourway-source9','fourway-rowdma2'):
  local=json.loads((ROOT/'tmp/frame-execution'/sub/'inputs.json').read_text())
  for source in local['inputs']:
   name=Path(source['snapshot']).name
   if name in byname and not name.startswith('tb_'):assert source['sha256']==byname[name],(sub,name)
 warn=run/'warnings_review.json';wr=json.loads(warn.read_text());assert wr['inputs_sha256']==sha(run/'inputs.json') and wr['no_latch_loop_range_implicit']
 dep=vec/'dependency_audit.json';dr=json.loads(dep.read_text());assert dr['status']=='FINITE_DDR_DEPENDENCY_COVERAGE_PASS'
 assert r['commands']==21556 and int(r['line'][3])==65851200 and int(r['line'][4])==2764800 and int(r['line'][8],16)==(1<<29)-1
 assert sum(x['cycles'] for x in r['reasons'])==r['cycles']
 nodes={x['id']:x for x in r['nodes']};names={n['id']:n for n in vm['nodes']}
 categories={
  'RGB首层': [0],
  '两层下采样DW':[1,3],
  '残差DW':[5,7,10,12,15,17],
  '残差普通PW':[6,11,16],
  '残差末PW与加法融合':[9,14,19],
  '其他PW':[2,4,21,25],
  '解码DW':[20,23,24,27],
  'RGB双像素输出':[28]
 }
 parts=[dict(name=name,cycles=sum(nodes[i]['cycles_with_header_control'] for i in indices)) for name,indices in categories.items()]
 used={i for ids in categories.values() for i in ids}
 assert all(n['cycles_with_header_control']==0 for i,n in nodes.items() if i not in used)
 remainder=r['cycles']-sum(x['cycles'] for x in parts)
 assert remainder>=0;parts.append(dict(name='DDR行搬运与帧收尾',cycles=remainder))
 for p in parts:p['ms_at_150mhz']=p['cycles']/150000
 result=dict(status='BOUND_CONTINUOUS_FRAME_MEASUREMENT_AUDIT_PASS',cycles=r['cycles'],frame_ms_at_150mhz=r['measured_frame_ms'],
  budget_cycles=5000000,budget_ms=1000/30,excess_ms=r['measured_frame_ms']-1000/30,latency_budget_pass=r['cycles']<=5000000,
  parts=parts,exclusive_reasons=r['reasons'],scalar_checks=int(r['line'][3]),final_bytes=int(r['line'][4]),frame_pixels=921600,
  readbeats=int(r['line'][5]),writebeats=int(r['line'][6]),write_responses=int(r['line'][7]),nodes=29,
  scope='One continuous Model-B 720p RTL execution, resident diagnostic weights, input-ready to final AXI B + meter publication.',
  assumptions=['150MHz target used for cycle conversion; final STA not passed','Ideal single-client AXI128 behavior profile; no PHY/refresh/arbitration',
   'Precompiled finite descriptor driver; production scheduler not synthesized','Diagnostic weights/scales, not trained model','One frame; steady commit interval unmeasured',
   'Camera acquisition, gesture CNN, evil pixel effects, display integration not inside this CNN measurement'],
  evidence=[dict(path=str(p),sha256=sha(p)) for p in (run/'result.json',run/'inputs.json',run/'simulation.log',warn,vec/'frame_manifest.json',dep)]+gates,
  testbench_compatibility=tb_compatibility,final_mapped_function=False,pnr=False,board=False,full_baseline_30fps=False)
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({k:result[k] for k in ('status','cycles','frame_ms_at_150mhz','excess_ms','parts')},ensure_ascii=False))
if __name__=='__main__':main()

