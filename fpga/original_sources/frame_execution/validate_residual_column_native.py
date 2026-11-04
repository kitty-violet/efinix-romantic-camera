"""Frozen shared cluster native regression, official models and live readback verification."""
import argparse,collections,json,os,re,shutil,subprocess,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,sha
from validate_residual_column import vectors
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--full',action='store_true');p.add_argument('--nn',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False);src=root/'source';src.mkdir()
 try:
  m=json.loads((a.source/'inputs.json').read_text());records=[];compiled=[]
  for r in m['inputs']:
   f=Path(r['snapshot']);assert sha(f)==r['sha256'];dst=src/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
   if f.suffix in ('.v','.sv'):compiled.append(dst)
  if a.full:
   producer=vectors
   oracle=producer(root/'vectors',True);vh={f.name:sha(f) for f in (root/'vectors').iterdir() if f.is_file()}
  else:
   shutil.copytree(a.source/'vectors',root/'vectors');oracle=m['oracle'];vh=m['vectors']
   for n,h in vh.items():assert sha(root/'vectors'/n)==h
  deps={str(f):sha(f) for f in (Path(__file__),HERE/'validate_residual_column.py',HERE/'residual_q17_align.py',HERE/'residual_tables.py',HERE/'validate_dw_block.py',ROOT/'network_ab/contract_reference.py',ROOT/'network_ab/model_b_contract.json')}
  env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),
    MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
  obj=root/'obj';top='tb_residual_column'
  args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,
   '--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),*[f.as_posix() for f in compiled]]
  manifest=dict(inputs=records,dependencies=deps,oracle=oracle,vectors=vh,source_manifest_sha256=sha(a.source/'inputs.json'),
   tools={str(f):sha(f) for f in (BIN/'verilator_bin.exe',BIN/'g++.exe',BIN/'mingw32-make.exe',Path(sys.executable))})
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2));(root/'compile-arguments.json').write_text(json.dumps(list(map(str,args)),indent=2))
  body,compile_seconds=run(args,root,root/'compile.log',env)
  warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M));assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warnings
  opts='-O1 -munroll-only-small-loops -mvzeroupper'
  make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}',
   'PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',
   f'OPT_FAST={opts}',f'OPT_GLOBAL={opts}','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
  (root/'make-arguments.json').write_text(json.dumps(list(map(str,make)),indent=2));_,build_seconds=run(make,obj,root/'build.log',env)
  binary=obj/('V'+top+'.exe')
  body,sim_seconds=run([binary,'+VECTORS='+(root/'vectors').as_posix()],root,root/'simulation.log',env)
  assert not re.search(r'\bFAIL\b|%Error',body)
  passed=re.findall(r'^(PASS RESIDUAL_COLUMN.*)$',body,re.M);rows=re.findall(r'^RESCOL (.*)$',body,re.M)
  assert len(passed)==1 and len(rows)==oracle['cases']*2+1
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  for f,h in deps.items():assert sha(f)==h
  result=dict(status='LOCAL_RESIDUAL_COLUMN_NATIVE_SOURCE_PASS',line=passed[0],rows=rows,oracle=oracle,warning_codes=dict(warnings),
   compile_seconds=compile_seconds,build_seconds=build_seconds,simulation_seconds=sim_seconds,inputs_sha256=sha(root/'inputs.json'),
   binary_sha256=sha(binary),two_state=True,warning_review_open=True,coefficient_reads_for_residual=False,
   header_external=True,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
