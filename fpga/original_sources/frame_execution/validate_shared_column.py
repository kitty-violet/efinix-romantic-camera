"""Verify both existing real-port suites through the single shared reader."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--kind',choices=['dw','pw'],required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 records=[];paths=[]
 try:
  sr=json.loads((a.source/'result.json').read_text());m=json.loads((a.source/'inputs.json').read_text());assert sha(a.source/'inputs.json')==sr['inputs_sha256']
  for r in m['inputs']:
   old=Path(r['snapshot']);assert sha(old)==r['sha256']
   if old.suffix not in ('.sv','.v'):continue
   f=HERE/'variants/shared_columns'/old.name if old.name==a.kind+'_column_reader.sv' else old
   paths.append(f)
  paths.append(HERE/'compute_column_reader.sv')
  for f in paths+[Path(__file__)]:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  shutil.copytree(a.source/'vectors',root/'vectors')
  for n,h in m['vectors'].items():assert sha(root/'vectors'/n)==h
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,vectors=m['vectors'],oracle=m['oracle'],shared_reader=True),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[root/f.name for f in paths]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_dw_block','-gSEED=71','+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'seed71.log')
  prefix='PW_COLUMN' if a.kind=='pw' else 'DW_BLOCK'
  passed=re.findall(r'^# (PASS '+prefix+r'.*)$',body,re.M);rows=re.findall(r'^# BLOCK (.*)$',body,re.M)
  assert len(passed)==1 and len(rows)==m['oracle']['cases']*2 and 'Errors: 0, Warnings: 0' in body
  assert rows==sr['results'][0]['rows'] and passed[0]==sr['results'][0]['line'],'separate/shared reader trace mismatch'
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_SHARED_COLUMN_'+a.kind.upper()+'_SOURCE_PASS',results=[dict(seed=71,line=passed[0],rows=rows,seconds=secs)],oracle=m['oracle'],
   inputs_sha256=sha(root/'inputs.json'),separate_reader_exact_trace=True,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
