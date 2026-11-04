"""Audit area changes against source snapshots, actual logs and fixed netlists."""
import argparse,ast,json,re
from pathlib import Path
from run_units import HERE,ROOT,sha
from audit_joint import freeze
RUNS=('feed32-shared-header-source1','feed32-shared-header-fault1','feed32-shared-header-map1',
 'banked-pool-source2','banked-pool-source3','banked-pool-map1',
 'feed32-banked-source1','feed32-banked-identity1','feed32-banked-fault1','feed32-banked-map1',
 'feed32-aligned-source1','feed32-aligned-fault1','feed32-aligned-map1')
def extra(name):
 root=ROOT/'tmp/frame-execution'/name;path=root/'result.json';r=json.loads(path.read_text())
 assert sha(root/'inputs.json')==r['inputs_sha256'];m=json.loads((root/'inputs.json').read_text())
 for item in m['inputs']:
  assert sha(item['snapshot'])==item['sha256']
  if Path(item['path']).suffix in ('.sv','.v'):assert sha(item['path'])==item['sha256']
 for row in r['results']:
  log=Path(row.get('log',str(root/f'seed{row["seed"]}.log')));body=log.read_text()
  assert sha(log)==row['log_sha256'] and '# '+row['line'] in body
  assert 'Errors: 0, Warnings: 0' in body
 return dict(path=str(path),sha256=sha(path),status=r['status'],results=r['results'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 runs=[freeze(n) for n in RUNS];transpose=extra('transpose-aligned1')
 old=json.loads((ROOT/'tmp/frame-execution/feed32-local-map1/result.json').read_text())
 new=json.loads((ROOT/'tmp/frame-execution/feed32-aligned-map1/result.json').read_text())
 old_count=sum(old['resources'].get(k,0) for k in ('LUT4','ADD','SRL8'));new_count=sum(new['resources'].get(k,0) for k in ('LUT4','ADD','SRL8'))
 assert (old_count,new_count)==(53243,41915)
 for name in ('feed32-banked-source1','feed32-banked-identity1','feed32-aligned-source1'):
  r=json.loads((ROOT/'tmp/frame-execution'/name/'result.json').read_text())
  for row in r['results']:
   intervals=[]
   for profile in range(6):
    points=[tuple(map(int,m.groups())) for line in row['lines'] if (m:=re.match(
      rf'MEASURE ABC width=256 profile={profile} batch=(\d+) accepted=(\d+) started=(\d+) completed=(\d+)',line))]
    assert len(points)>=8
    gaps={b[2]-aa[2] for aa,b in zip(points,points[1:])};assert len(gaps)==1;intervals.append(gaps.pop())
   assert intervals==[128,16,64,64,16,8]
 for f in HERE.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
 contract=ROOT/'network_ab/model_b_contract.json';assert sha(contract)=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 lower=ROOT/'output/toolchain/frame_arithmetic_lower_bound_2026-09-30.json';q=json.loads(lower.read_text())
 assert q['ideal_ms']==27.432 and q['measured_frame_ms'] is None and not q['full_frame_30fps']
 result=dict(status='AUDITED_RAM_QUEUE_AND_ALIGNED_TRANSPOSE_LOCAL_PASS',runs=runs,transpose=transpose,
  combination_lower_bound_before=old_count,combination_lower_bound_after=new_count,reduction=old_count-new_count,
  ff_before=old['resources']['FF'],ff_after=new['resources']['FF'],ram_before=old['resources']['RAM10'],ram_after=new['resources']['RAM10'],
  source_intervals_unchanged=[128,16,64,64,16,8],contract_sha256=sha(contract),
  arithmetic_lower_bound=dict(path=str(lower),sha256=sha(lower)),runner_sha256=sha(Path(__file__)),
  measured_frame_ms=None,full_frame_30fps=False,production_promoted=False,final_pnr=False,physical_resource_fit=False)
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(status=result['status'],runs=len(runs),reduction=old_count-new_count,
  transpose_scalar_checks=sum(int(re.search(r'scalar=(\d+)',r['line'])[1]) for r in transpose['results']))))
if __name__=='__main__':main()
