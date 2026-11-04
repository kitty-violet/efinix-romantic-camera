"""Fast whole local DW suite, exact frozen RTL/official primitives; two-state."""
import argparse,collections,json,os,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--full-vectors',action='store_true');p.add_argument('--fusion-vectors',action='store_true');p.add_argument('--phase-steps',nargs='+',type=int,default=[8]);p.add_argument('--map',type=Path);p.add_argument('--paired',type=Path);p.add_argument('--pw',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False);src=root/'source';src.mkdir()
 try:
  m=json.loads((a.source/'inputs.json').read_text());records=[];compiled=[]
  replaced=set();net=None
  if a.map:
   assert a.paired and not a.full_vectors and not a.fusion_vectors
   mr=json.loads((a.map/'result.json').read_text());mi=json.loads((a.map/'inputs.json').read_text())
   assert mr['status']=='JOINT_DW_CANDIDATE_MAP_REQUIRES_GATE' and sha(a.map/'inputs.json')==mr['inputs_sha256']
   byname={Path(r['snapshot']).name:r for r in m['inputs']}
   for r in mi['inputs']:
    name=Path(r['snapshot']).name;assert r['sha256']==byname[name]['sha256'];replaced.add(name)
   net=a.map/'outflow/dw_column_cluster.map.v';assert sha(net)==mr['map_sha256']
  for r in m['inputs']:
   if Path(r['snapshot']).name in replaced:continue
   f=Path(r['snapshot']);assert sha(f)==r['sha256'];dst=src/f.name;shutil.copy2(f,dst)
   records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
   if f.suffix in ('.sv','.v'):compiled.append(dst.as_posix())
  if net:
   kinds=set(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M))
   for f in [net]+[EF/'sim_models/verilog'/('efx_'+k.lower()+'.v') for k in sorted(kinds)]:
    dst=src/f.name
    if dst.exists():assert sha(dst)==sha(f);continue
    shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)));compiled.append(dst.as_posix())
  if a.full_vectors or a.fusion_vectors:
   from validate_dw_block import vectors
   if a.pw:
    from validate_pw_column import vectors
   if a.fusion_vectors:
    from dw_fusion_shapes import vectors
   m['oracle']=vectors(root/'vectors');m['vectors']={f.name:sha(f) for f in (root/'vectors').iterdir()}
   if a.fusion_vectors:
    for name in ('dw_fusion_shapes.py','residual_fusion.py'):
     f=HERE/name;dst=src/name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
   f=HERE/('validate_pw_column.py' if a.pw else 'validate_dw_block.py');dst=src/f.name
   # Record the new vector producer separately from the frozen RTL runner copy.
   dst=src/'full_vector_producer.py';shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  else:
   shutil.copytree(a.source/'vectors',root/'vectors')
   for n,h in m['vectors'].items():assert sha(root/'vectors'/n)==h
  env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),
   MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
  top='tb_dw_block';obj=root/'obj'
  args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,
   '--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),'-GSEED=71',*(['+define+DW_STRIPED'] if m.get('striped') else []),*compiled]
  manifest=dict(inputs=records,vectors=m['vectors'],oracle=m['oracle'],source_manifest_sha256=sha(a.source/'inputs.json'),
   runner_sha256=sha(__file__),tools={str(f):sha(f) for f in (BIN/'verilator_bin.exe',BIN/'g++.exe',BIN/'mingw32-make.exe',Path(sys.executable))})
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2));(root/'compile-arguments.json').write_text(json.dumps(list(map(str,args)),indent=2))
  body,compile_seconds=run(args,root,root/'compile.log',env)
  codes=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M));assert not (set(codes)&{'UNOPTFLAT','LATCH','SELRANGE'}),codes
  if net:
   from audit_dw_mapped_compile import audit
   (root/'compile-warning-audit.json').write_text(json.dumps(audit(root),indent=2))
  else:assert 'IMPLICIT' not in codes
  make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',
   f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}',
   'SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',
   'OPT_FAST=-O1 -munroll-only-small-loops -mvzeroupper','OPT_GLOBAL=-O1 -munroll-only-small-loops -mvzeroupper',
   'CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
  (root/'make-arguments.json').write_text(json.dumps(list(map(str,make)),indent=2));_,build_seconds=run(make,obj,root/'build.log',env)
  binary=obj/('V'+top+'.exe');assert binary.exists()
  runs=[]
  for step in a.phase_steps:
   log=root/('simulation_step'+str(step)+'.log')
   body,seconds=run([binary,'+VECTORS='+(root/'vectors').as_posix(),'+PHASE_STEP='+str(step)],root,log,env)
   assert not re.search(r'\bFAIL\b|%Error',body)
   passed=re.findall(r'^(PASS '+('PW_COLUMN' if a.pw else 'DW_BLOCK')+r'.*)$',body,re.M);rows=re.findall(r'^BLOCK (.*)$',body,re.M)
   assert len(passed)==1 and len(rows)==m['oracle']['cases']*2
   runs.append(dict(phase_step=step,line=passed[0],rows=rows,seconds=seconds,log_sha256=sha(log)))
   print(json.dumps(dict(phase_step=step,line=passed[0],seconds=seconds)),flush=True)
  if a.paired:
   paired=json.loads((a.paired/'result.json').read_text())
   assert paired['status']=='LOCAL_DW_BLOCK_NATIVE_SOURCE_PASS'
   for got,expected in zip(runs,paired['runs'],strict=True):
    assert got['rows']==expected['rows'] and got['line']==expected['line'],'source/map trace mismatch'
  result=dict(status='LOCAL_DW_BLOCK_NATIVE_MAP_PAIRED_PASS' if net else ('LOCAL_PW_COLUMN_NATIVE_SOURCE_PASS' if a.pw else 'LOCAL_DW_BLOCK_NATIVE_SOURCE_PASS'),runs=runs,warning_codes=dict(codes),mapped=bool(net),
   compile_seconds=compile_seconds,build_seconds=build_seconds,inputs_sha256=sha(root/'inputs.json'),
   binary_sha256=sha(binary),two_state=True,warning_review_open=True,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
