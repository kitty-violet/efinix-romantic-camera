"""Bounded gate partitions using exact frozen binaries; full shapes stay at RTL."""
import argparse,concurrent.futures,json,os,re,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
from validate_dw_block import vectors
from audit_dw_mapped_compile import audit
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import run,BIN,VROOT
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--mapped',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 source=a.source.resolve();mapped=a.mapped.resolve();review=audit(mapped)
 binaries={k:(d/'obj/Vtb_dw_block.exe') for k,d in (('source',source),('mapped',mapped))}
 sr=json.loads((source/'result.json').read_text());assert sha(binaries['source'])==sr['binary_sha256']
 recovery=json.loads((mapped/'warning-recovery.json').read_text())
 for name,h in recovery['codegen'].items():assert sha(mapped/'obj'/name)==h
 binary_hashes={k:sha(f) for k,f in binaries.items()}
 shapes=[[(9,2,1,79,6),(33,3,2,287,177)],[(1,1,1,0,0),(32,4,1,0,0)],[(80,4,4,0,0),(84,2,2,236,178)]]
 parts=[]
 for i,ss in enumerate(shapes):
  folder=root/str(i);folder.mkdir();oracle=vectors(folder/'vectors',shapes=ss)
  parts.append(dict(folder=folder,oracle=oracle))
 manifest=dict(binaries={k:str(f) for k,f in binaries.items()},binary_hashes=binary_hashes,review=review,
  source_manifest_sha256=sha(source/'inputs.json'),map_manifest_sha256=sha(mapped/'inputs.json'),
  vectors={str(p['folder'].name):{f.name:sha(f) for f in (p['folder']/'vectors').iterdir()} for p in parts},
  evidence={str(f):sha(f) for f in (Path(__file__),HERE/'validate_dw_block.py',ROOT/'network_ab/contract_reference.py')})
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],VERILATOR_ROOT=str(VROOT))
 def one(part):
  folder=part['folder'];traces={}
  for k,binary in binaries.items():
   log=folder/(k+'.log');body,secs=run([binary,'+VECTORS='+(folder/'vectors').as_posix()],folder,log,env)
   assert not re.search(r'\bFAIL\b|%Error',body)
   passed=re.findall(r'^PASS DW_BLOCK.*$',body,re.M);rows=re.findall(r'^BLOCK .*$',body,re.M)
   assert len(passed)==1 and len(rows)==part['oracle']['cases']*2
   traces[k]=dict(line=passed[0],rows=rows,seconds=secs,log_sha256=sha(log))
  assert traces['source']['line']==traces['mapped']['line'] and traces['source']['rows']==traces['mapped']['rows']
  result=dict(oracle=part['oracle'],traces=traces,status='SOURCE_MAP_NUMERIC_PROTOCOL_AND_CYCLES_EQUAL')
  (folder/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(part=folder.name,status=result['status'])),flush=True)
  return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,parts))
 for k,f in binaries.items():assert sha(f)==binary_hashes[k]
 for f,h in manifest['evidence'].items():assert sha(f)==h
 result=dict(status='LOCAL_DW_JOINT_MAP_REPRESENTATIVE_PAIRED_PASS',results=results,warning_review=review,inputs_sha256=sha(root/'inputs.json'),
 full_shape_map_suite=False,full_shape_rtl_suite=True,bounded_reader=False,full_frame=False,pnr=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
