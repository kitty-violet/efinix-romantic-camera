"""Native replay of the exact local numeric/held-token/fault testbench snapshot."""
import argparse,collections,json,os,re,shutil,subprocess,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reuse',type=Path);p.add_argument('--trace-case',type=int);p.add_argument('--multi-issue',action='store_true');p.add_argument('--rotating-dispatch',action='store_true');p.add_argument('--overrides-dir',type=Path);p.add_argument('--frozen-base',type=Path);p.add_argument('--specialized',action='store_true');p.add_argument('--dsp-model',action='store_true');p.add_argument('--selected-cases',type=int,nargs='+',default=[5,6,7,8,12,9,14,15,10,13,16,21]);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp');r.mkdir(parents=True,exist_ok=False)
try:
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(r),TMP=str(r),TMPDIR=str(r),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
 if a.reuse:
  prior=json.loads((a.reuse/'native_inputs.json').read_text());snapshot=Path(prior['snapshot']);binary=Path(prior['binary']);assert sha(binary)==prior['binary_sha256']
  assert prior.get('multi_issue',False)==a.multi_issue
  assert prior.get('rotating_dispatch',False)==a.rotating_dispatch
  assert prior.get('specialized',False)==a.specialized
  assert prior.get('dsp_model',False)==a.dsp_model
  assert prior.get('selected_cases',[5,6,7,8,12,9,14,15,10,13,16,21])==a.selected_cases
  assert prior.get('overrides_dir')==(str(a.overrides_dir.resolve()) if a.overrides_dir else None)
  assert sha(snapshot/'inputs.json')==prior['source_manifest_sha256']
 else:
  snapshot=r/'frozen'
  subprocess.run([sys.executable,'-B',str(HERE/'validate_fourway.py'),'--root',str(snapshot),'--production','--factored-dispatch','--fast-skip','--snapshot-only','--selected-cases',*map(str,a.selected_cases)]+(['--specialized'] if a.specialized else []),check=True,cwd=ROOT)
  source=json.loads((snapshot/'inputs.json').read_text())
  if a.frozen_base:
   from frozen_sources import frozen_paths
   resolved=frozen_paths([Path(record['path']) for record in source['inputs']],a.frozen_base)
   for record,f in zip(source['inputs'],resolved):
    shutil.copy2(f,record['snapshot']);record.update(path=str(f),sha256=sha(f))
   (snapshot/'inputs.json').write_text(json.dumps(source,indent=2))
  if a.multi_issue:
   f=HERE/'variants/multi_issue_reader/unified_column_reader.sv'
   for record in source['inputs']:
    if Path(record['path']).name==f.name:
     shutil.copy2(f,record['snapshot']);record.update(path=str(f),sha256=sha(f))
   (snapshot/'inputs.json').write_text(json.dumps(source,indent=2))
  if a.rotating_dispatch:
   f=HERE/'variants/rotating_dispatch/unified_column_dispatch32.sv'
   for record in source['inputs']:
    if Path(record['path']).name==f.name:
     shutil.copy2(f,record['snapshot']);record.update(path=str(f),sha256=sha(f))
   (snapshot/'inputs.json').write_text(json.dumps(source,indent=2))
  if a.overrides_dir:
   override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
   for record in source['inputs']:
    f=override/Path(record['path']).name
    if f.is_file():shutil.copy2(f,record['snapshot']);record.update(path=str(f),sha256=sha(f))
   (snapshot/'inputs.json').write_text(json.dumps(source,indent=2))
  if a.dsp_model:
   f=EF/'sim_models/verilog/efx_dsp48.v';dst=snapshot/f.name;shutil.copy2(f,dst)
   source['inputs'].insert(0,dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
   (snapshot/'inputs.json').write_text(json.dumps(source,indent=2))
  # A board snapshot may have a newer private RAM/header ABI than the
  # generic historical TB. Reject that mismatch before paying compile cost.
  tb_path=next(Path(x['snapshot']) for x in source['inputs'] if Path(x['snapshot']).name=='tb_compute_column_cluster.sv')
  tb_body=tb_path.read_text()
  arena_path=next(Path(x['snapshot']) for x in source['inputs'] if Path(x['snapshot']).name=='tensor_arena12.sv')
  if re.search(r'assign\s+rsp_tag\s*=\s*(?:16|32)\s*\x27d0',arena_path.read_text()):
   assert "st[p*16+:16]!=16'hbeef" not in tb_body,'TB expects retired SRAM tag echo; select the matched zero-tag testbench'
  if 'residual_abort_checks!=5' in tb_body:
   assert 9 in a.selected_cases,'Active residual abort coverage requires case 9'
  obj=r/'obj';top='tb_compute_column_cluster'
  body,_=run([BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,'--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),*[Path(x['snapshot']).as_posix() for x in source['inputs']]],r,r/'compile.log',env)
  warning_codes=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M));assert not(set(warning_codes)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warning_codes
  run([BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}','OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch'],obj,r/'build.log',env)
  binary=obj/('V'+top+'.exe');prior=dict(snapshot=str(snapshot),binary=str(binary),binary_sha256=sha(binary),warning_codes=dict(warning_codes),multi_issue=a.multi_issue,rotating_dispatch=a.rotating_dispatch,specialized=a.specialized,overrides_dir=str(a.overrides_dir.resolve()) if a.overrides_dir else None)
 source=json.loads((snapshot/'inputs.json').read_text())
 for x in source['inputs']:assert sha(x['snapshot'])==x['sha256']
 prior['dsp_model']=a.dsp_model
 prior['selected_cases']=a.selected_cases
 prior['source_manifest_sha256']=sha(snapshot/'inputs.json');(r/'native_inputs.json').write_text(json.dumps(prior,indent=2))
 command=subprocess.list2cmdline([sys.executable,'-B',str(HERE.parent/'coefficient_store/validate_readback.py'),'--image-dir',str(snapshot/'vectors'),'--readback',str(r/'readback.hex')])
 args=[binary,'+VECTORS='+(snapshot/'vectors').as_posix(),'+DUMP='+(r/'readback.hex').as_posix(),'+VERIFY_COMMAND='+command]
 if a.trace_case is not None:args+=['+TRACE_CASE='+str(a.trace_case)]
 body,seconds=run(args,r,r/'simulation.log',env,timeout=600)
 lines=re.findall(r'^PASS COMPUTE_CLUSTER.*$',body,re.M);assert len(lines)==1 and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and not re.search(r'\bFAIL\b|%Error',body)
 result=dict(status='NATIVE_LOCAL_CLUSTER_NUMERIC_PROTOCOL_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(r/'native_inputs.json'),log_sha256=sha(r/'simulation.log'),four_state=False,full_frame=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
