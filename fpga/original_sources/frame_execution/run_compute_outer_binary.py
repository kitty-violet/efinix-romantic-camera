"""All actual fused DW/PW representative costs through one frozen real cluster."""
import argparse,itertools,json,os,re,subprocess,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,sha
from cluster_nn_vectors import vectors

sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import run,BIN,VROOT
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  ref=json.loads((a.native/'result.json').read_text());m=json.loads((a.native/'inputs.json').read_text())
  assert ref['status']=='LOCAL_COMPUTE_NN_NATIVE_SOURCE_PASS' and sha(a.native/'inputs.json')==ref['inputs_sha256']
  binary=a.native/'obj/Vtb_cluster_native.exe';assert sha(binary)==ref['binary_sha256']
  for r in m['inputs']:assert sha(r['snapshot'])==r['sha256']
  floor=json.loads((ROOT/'output/toolchain/frame_arithmetic_lower_bound_2026-09-30.json').read_text())
  nodes=[n for n in floor['nodes'] if not n['node'].startswith(('res1_','res2_','res3_')) and n['node'] not in ('stem','down1.dw','down2.dw') and n['kind']!='nearest_x2']
  shapes=[];counts=[];kinds=[]
  for node in nodes:
   name=node['node'];height,width,_=node['shape']
   tile=80 if width==320 else 160 if width==640 else 320
   nn=int(name in ('up1_high_dw','up2_high_dw'))
   for x in range(0,width,tile):
    for y,count in [(0,1),(4,height//4-2),(height-4,1)]:
     shapes.append((name,tile,4,x,y,nn,width,height));counts.append(count);kinds.append(name)
  shapes.append(('down1.pw',80,2,160,3,0,320,180));counts.append(0);kinds.append('stall_coverage')
  assert len(shapes)<=256
  oracle=vectors(root/'vectors',task_shapes=shapes);oracle.update(counts=counts,kinds=kinds)
  assert oracle['output_words']<=524288
  checker=HERE.parent/'coefficient_store/validate_readback.py'
  deps=[Path(__file__),HERE/'cluster_nn_vectors.py',ROOT/'output/toolchain/frame_arithmetic_lower_bound_2026-09-30.json',
   HERE/'validate_dw_block.py',HERE.parent/'pw_array_feed/feed_reference.py',ROOT/'network_ab/model_b_contract.json',ROOT/'network_ab/contract_reference.py',checker]
  evidence={str(f):sha(f) for f in deps}
  manifest=dict(binary=str(binary),binary_sha256=sha(binary),source_manifest_sha256=sha(a.native/'inputs.json'),evidence=evidence,oracle=oracle,
   block_partition_commands=sum(counts),partition_nodes=nodes,vectors={f.name:sha(f) for f in (root/'vectors').iterdir() if f.is_file()})
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  env=os.environ.copy();env.update(VERILATOR_ROOT=str(VROOT),PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'])
  dump=root/'readback.hex';command=subprocess.list2cmdline([str(Path(sys.executable)),'-B',str(checker),'--image-dir',str(root/'vectors'),'--readback',str(dump)])
  body,seconds=run([binary.resolve(),'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+command],root,root/'simulation.log',env)
  assert not re.search(r'\bFAIL\b|%Error',body) and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body
  passed=re.findall(r'^(PASS COMPUTE_CLUSTER.*)$',body,re.M);rows=re.findall(r'^BLOCK (.*)$',body,re.M)
  assert len(passed)==1 and len(rows)==len(shapes)*2
  detail=[];totals={n['node']:0 for n in nodes}
  for line in rows:
   d={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
   if not d['random']:
    n=d['case'];s=shapes[n];assert (d['width'],d['rows'])==s[1:3]
    weighted=d['cycles']*counts[n]
    if counts[n]:totals[kinds[n]]+=weighted
    detail.append(dict(shape=s,count=counts[n],kind=kinds[n],cycles=d['cycles'],weighted_cycles=weighted))
  for f,h in evidence.items():assert sha(f)==h
  assert sha(binary)==manifest['binary_sha256']
  result=dict(status='LOCAL_OUTER_BLOCK_PARTITION_SCREENING_PASS',line=passed[0],rows=rows,detail=detail,seconds=seconds,
   weighted_cycles=totals,weighted_ms={k:v/150000 for k,v in totals.items()},inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
   measured_frame_ms=None,full_frame_30fps=False,notes=['Real frozen cluster including resident-weight reads and actual write ACK.',
    'Costs include test-driver command handshake/completion spacing; header prefetch and live allocator placement are not included.',
    'Complete tensor coverage by 80/160/320 x4 blocks; independent original-tensor fixtures. This is a screening partition, not a validated live full-graph schedule; cross-layer halo/rematerialization overhead is still uncharged.'])
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('rows','detail')}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
