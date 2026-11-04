"""Run new frozen golden geometry through an already verified unchanged binary."""
import argparse,json,os,re,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
from pw_fusion_shapes import vectors
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import run,BIN,VROOT
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 ref=json.loads((a.native/'result.json').read_text());m=json.loads((a.native/'inputs.json').read_text())
 assert ref['status']=='LOCAL_PW_COLUMN_NATIVE_SOURCE_PASS' and sha(a.native/'inputs.json')==ref['inputs_sha256']
 binary=a.native/'obj/Vtb_dw_block.exe';assert sha(binary)==ref['binary_sha256']
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
 oracle=vectors(root/'vectors')
 evidence={str(f):sha(f) for f in (Path(__file__),HERE/'pw_fusion_shapes.py',HERE/'residual_fusion.py',HERE/'validate_pw_column.py',HERE/'validate_dw_block.py',ROOT/'network_ab/model_b_contract.json',ROOT/'network_ab/contract_reference.py',ROOT/'network_ab/model_contract.py')}
 manifest=dict(binary=str(binary),binary_sha256=sha(binary),source_inputs_sha256=sha(a.native/'inputs.json'),evidence=evidence,oracle=oracle,vectors={f.name:sha(f) for f in (root/'vectors').iterdir()})
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 env=os.environ.copy();env.update(VERILATOR_ROOT=str(VROOT),PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'])
 body,seconds=run([binary.resolve(),'+VECTORS='+(root/'vectors').as_posix()],root,root/'simulation.log',env)
 assert not re.search(r'\bFAIL\b|%Error',body)
 passed=re.findall(r'^(PASS PW_COLUMN.*)$',body,re.M);rows=re.findall(r'^BLOCK (.*)$',body,re.M)
 assert len(passed)==1 and len(rows)==oracle['cases']*2
 detail=[]
 for line in rows:
  d={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
  if not d['random']:
   n=d['case'];shape=oracle['shapes'][n];assert (d['width'],d['rows'],d['groups'])==(shape[0],shape[1],shape[3])
   detail.append(dict(shape=shape,count=oracle['counts'][n],cycles=d['cycles'],weighted_cycles=d['cycles']*oracle['counts'][n]))
 assert len(detail)==oracle['cases']
 total=sum(x['weighted_cycles'] for x in detail)
 for f,h in evidence.items():assert sha(f)==h
 assert sha(binary)==manifest['binary_sha256']
 result=dict(status='LOCAL_PW_FUSED_COMMAND_SHAPES_PASS',line=passed[0],rows=rows,detail=detail,seconds=seconds,oracle=oracle,
 weighted_isolated_cycles=total,weighted_isolated_ms=total/150000,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
 measured_frame_ms=None,full_frame_30fps=False,coefficient_reload_not_included=True,real_allocator_bases_not_used=True)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('oracle','rows','detail')}),flush=True)
if __name__=='__main__':main()
