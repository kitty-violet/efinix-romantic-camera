"""One bounded C3 routing-policy comparison on the already reviewed placement.

Stop if the first three complete iterations exactly replay failed C2. A
different prefix permits the already bounded run, never claims final success.
"""
import argparse,json,re,subprocess,sys,time
from pathlib import Path
from run_units import HERE,ROOT,sha
from windows_process_tree_safe import stop_launched_tree

def iterations(path):
 if not path.exists():return []
 return [tuple(map(int,r)) for r in re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)\s+heapops=(\d+)\s+\([^)]*\)\s+cpd=(\d+)',path.read_text(errors='replace'))]

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',type=Path,required=True)
 p.add_argument('--reference',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--route-root',type=Path,required=True)
 a=p.parse_args();root=a.root.resolve();job=a.route_root.resolve();physical=a.physical.resolve();reference=a.reference.resolve()
 assert root.is_relative_to(ROOT/'tmp') and job.is_relative_to(ROOT/'tmp') and not root.exists() and not job.exists()
 expected=iterations(reference);assert len(expected)>=3
 assert json.loads((physical/'physical_inputs.json').read_text())['optimization_change']['value']=='CONGESTION_3'
 root.mkdir();record=dict(status='C3_ROUTER_POLICY_PREFIX_COMPARISON_RUNNING',reference=str(reference),reference_sha256=sha(reference),
    first_three_reference=expected[:3],physical=str(physical),route_root=str(job),runner_sha256=sha(__file__),budget_s=1200,hardware_access=False)
 save=lambda:(root/'result.json').write_text(json.dumps(record,indent=2)+'\n')
 save();start=time.monotonic()
 cmd=[sys.executable,'-u',str(HERE/'route_video_service.py'),'--physical',str(physical),'--review',str(a.review.resolve()),'--root',str(job),'--budget','1200']
 with (root/'launcher.log').open('x') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);record['pid']=proc.pid;save();compared=False
  while proc.poll() is None:
   actual=iterations(physical/'outflow/ti60f225_oob.route.out')
   if not compared and len(actual)>=3:
    compared=True;record['first_three_actual']=actual[:3]
    record['identical_prefix']=actual[:3]==expected[:3];save()
    if record['identical_prefix']:
     record.update(status='IDENTICAL_C3_ROUTE_PREFIX_STOPPED_NOT_ROUTE_FAILURE',cleanup=stop_launched_tree(proc),
        limitation='Only the initial three iterations matched; later policy divergence was not tested. No legal route or timing claim.')
     break
    record['status']='DISTINCT_C3_PREFIX_BOUNDED_ROUTE_CONTINUES';save()
   if time.monotonic()-start>1300:
    record.update(status='C3_COMPARISON_HOST_TIMEOUT_NOT_PASS',cleanup=stop_launched_tree(proc));break
   time.sleep(1)
  else:
   record.update(exit_code=proc.returncode,status='C3_POLICY_ROUTE_RUN_FINISHED' if proc.returncode==0 else 'C3_POLICY_ROUTE_RUN_FAILED')
 record['seconds']=time.monotonic()-start;save();print(json.dumps(record))

if __name__=='__main__':main()
