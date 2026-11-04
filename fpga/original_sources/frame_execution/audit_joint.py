"""Freeze the joint fixes without promoting partial checks to whole-frame PASS."""
import argparse,ast,collections,json,re
from pathlib import Path
from run_units import HERE,ROOT,sha
RUNS=('pool32-source3','compact-source1','compact-identity1','compact-fault1','compact-map1',
 'feed32-source2','feed32-identity2','feed32-fault1','feed32-map1',
 'feed32-local-source1','feed32-local-fault1','feed32-local-fault8','feed32-local-map1',
 'residual-source1','residual-map1','residual-pool2','residual-dispatch-map1','dw32-plan1')
def freeze(name):
 root=ROOT/'tmp/frame-execution'/name;p=root/'result.json';r=json.loads(p.read_text());m=root/'inputs.json'
 assert sha(m)==r['inputs_sha256'];manifest=json.loads(m.read_text());snapshots=0
 for item in manifest.get('inputs',manifest.get('sources',[])):
  path=Path(item.get('snapshot',root/item.get('snapshot_name','')))
  assert path.is_file() and sha(path)==item['sha256'],path;snapshots+=1
  if Path(item['path']).suffix.lower() in ('.sv','.v','.hex'):
   assert sha(item['path'])==item['sha256'],item['path']
 logs=[]
 for run in r.get('results',[]):
  candidates=[root/f'seed{run["seed"]}.log',root/f'simulation-seed{run["seed"]}.log']
  log=next(p for p in candidates if p.exists());assert sha(log)==run['log_sha256']
  body=log.read_text();assert 'Errors: 0, Warnings: 0' in body and not re.search(r'^# \*\* (?:Error|Fatal)',body,re.M)
  for line in run.get('lines',[run.get('line','')]):assert line and '# '+line in body
  logs.append(dict(path=str(log),sha256=sha(log)))
 if 'map_sha256' in r:
  nets=list((root/'outflow').glob('*.map.v'));reports=list((root/'outflow').glob('*.map.rpt'))
  assert len(nets)==len(reports)==1;net,rpt=nets[0],reports[0]
  assert sha(net)==r['map_sha256'] and sha(rpt)==r['report_sha256']
  resources=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  assert resources==r['resources']
  for k,v in resources.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
  console=next(p for p in (root/'map.log',root/'map-console.log') if p.is_file())
  assert re.search(r'map\s*:\s*PASS',console.read_text())
 for item in ('override.json',):
  if (root/item).exists():
   over=json.loads((root/item).read_text())
   for f,h in over['sources'].items():assert sha(f)==h
 return dict(name=name,path=str(p),sha256=sha(p),status=r['status'],snapshots=snapshots,logs=logs,
  resources=r.get('resources'),lines=[line for row in r.get('results',[]) for line in row.get('lines',[row.get('line')]) if line])
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 rows=[freeze(n) for n in RUNS]
 contract=ROOT/'network_ab/model_b_contract.json'
 assert sha(contract)=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 extra=[]
 for name in ('residual-tables1','residual-fusion1','dw32-screen1','feed32-local-lint1'):
  f=ROOT/'tmp/frame-execution'/name/'result.json';r=json.loads(f.read_text());extra.append(dict(path=str(f),sha256=sha(f),status=r['status']))
  if 'source_hashes' in r:
   for f,h in r['source_hashes'].items():assert sha(f)==h
  for case in r.get('cases',[]):
   if 'trace_path' in case:assert sha(case['trace_path'])==case['trace_sha256']
 rq=ROOT/'output/toolchain/rq_single_dsp_audit_2026-09-30.json';assert rq.is_file()
 scripts=list(HERE.glob('*.py'))
 for f in scripts:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
 result=dict(status='AUDITED_JOINT_LOCAL_FIXES_NOT_FULL_FRAME_PASS',runs=rows,other_evidence=extra,
  exact_RQ_audit=dict(path=str(rq),sha256=sha(rq)),contract_sha256=sha(contract),python_ast_files=len(scripts),
  measured_frame_ms=None,full_frame_30fps=False,production_promoted=False,final_pnr=False,physical_resource_fit=False,
  remaining=['actual DW bank read/data return and write competition','unique PW/DW/residual mode integration',
   'row/frame controller and DDR overlap/ownership/final B','full functional frame measurement','packed resources and four-seed STA'],
  runner_sha256=sha(Path(__file__)))
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(status=result['status'],runs=len(rows),snapshots=sum(r['snapshots'] for r in rows),python_ast_files=len(scripts),output=str(a.output))))
if __name__=='__main__':main()
