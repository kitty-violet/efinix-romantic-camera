"""Measure stem and both stride2 tensor partitions on a frozen shared cluster."""
import argparse,json,os,re,subprocess,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,sha
from cluster_stride_vectors import vectors
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import run,BIN,VROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  ref=json.loads((a.native/'result.json').read_text());m=json.loads((a.native/'inputs.json').read_text())
  assert ref['status']=='LOCAL_SHARED_STRIDE_COLUMN_NATIVE_PASS'
  assert sha(a.native/'inputs.json')==ref['inputs_sha256']
  binary=a.native/'obj/Vtb_cluster_native.exe';assert sha(binary)==ref['binary_sha256']
  for r in m['inputs']:assert sha(r['snapshot'])==r['sha256']
  shapes=[];counts=[];names=[]
  for name,width,height,tile,rows,sw,sh in [('stem',1280,720,320,4,1280,720),('down1.dw',640,360,160,2,1280,720),('down2.dw',320,180,160,2,640,360)]:
   for x in range(0,width,tile):
    for y,count in [(0,1),(rows,height//rows-2),(height-rows,1)]:
     shapes.append((name,tile,rows,x,y,0,sw,sh));counts.append(count);names.append(name)
  shapes += [('down1.pw',9,2,79,6,0),('up2_high_dw',9,2,79,6,0),('down2.pw',9,4,78,9,0),('up1_high_dw',9,2,0,0,1),('down1.pw',80,2,160,3,0)]
  counts += [0]*5;names += ['coverage']*5
  oracle=vectors(root/'vectors',task_shapes=shapes)
  counts += [0]*(oracle['cases']-len(counts));names += ['coverage']*(oracle['cases']-len(names))
  assert oracle['output_words']<=524288
  checker=HERE.parent/'coefficient_store/validate_readback.py'
  deps=[Path(__file__),HERE/'cluster_stride_vectors.py',HERE/'validate_dw_block.py',HERE.parent/'pw_array_feed/feed_reference.py',ROOT/'network_ab/model_b_contract.json',ROOT/'network_ab/contract_reference.py',checker]
  evidence={str(f):sha(f) for f in deps}
  manifest=dict(binary=str(binary),binary_sha256=sha(binary),source_manifest_sha256=sha(a.native/'inputs.json'),evidence=evidence,oracle=oracle,
   counts=counts,names=names,vectors={f.name:sha(f) for f in (root/'vectors').iterdir() if f.is_file()})
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  env=os.environ.copy();env.update(VERILATOR_ROOT=str(VROOT),PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'])
  dump=root/'readback.hex';command=subprocess.list2cmdline([str(Path(sys.executable)),'-B',str(checker),'--image-dir',str(root/'vectors'),'--readback',str(dump)])
  body,seconds=run([binary.resolve(),'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+command],root,root/'simulation.log',env)
  assert not re.search(r'\bFAIL\b|%Error',body) and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body
  passed=re.findall(r'^(PASS COMPUTE_CLUSTER.*)$',body,re.M);results=re.findall(r'^BLOCK (.*)$',body,re.M)
  assert len(passed)==1 and len(results)==oracle['cases']*2
  totals={n:0 for n in ('stem','down1.dw','down2.dw')};detail=[]
  for line in results:
   d={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
   if not d['random']:
    n=d['case'];weighted=d['cycles']*counts[n]
    if counts[n]:totals[names[n]]+=weighted
    detail.append(dict(case=n,operator=names[n],count=counts[n],cycles=d['cycles'],weighted_cycles=weighted))
  for f,h in evidence.items():assert sha(f)==h
  result=dict(status='LOCAL_FRONT_PARTITION_SCREENING_PASS',line=passed[0],rows=results,detail=detail,seconds=seconds,
   weighted_cycles=totals,weighted_ms={k:v/150000 for k,v in totals.items()},inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
   measured_frame_ms=None,full_frame_30fps=False,header_external=True,allocator_lifetime_external=True,
   notes=['Real resident/arena reads and write acknowledgements; synthetic original-tensor golden.',
   'Stride2 uses even-output filtering of the normal column stream; numeric diagnostic, not throughput-approved.',
   'Partition covers each tensor but is not a live cross-layer frame schedule.'])
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('rows','detail')}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
