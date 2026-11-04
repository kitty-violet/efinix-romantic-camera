"""Resume frozen generated C++ after exact unused-output warning proof.
Retains original failure.json and logs; never reuses codegen after DUT changes.
"""
import argparse,json,os,re,sys
from pathlib import Path
from run_units import HERE,ROOT,sha
from audit_dw_mapped_compile import audit
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paired',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not (root/'result.json').exists()
 review=audit(root);m=json.loads((root/'inputs.json').read_text());paired=json.loads((a.paired/'result.json').read_text())
 assert paired['status']=='LOCAL_DW_BLOCK_NATIVE_SOURCE_PASS'
 for name,h in m['vectors'].items():assert sha(root/'vectors'/name)==h
 for f,h in m['tools'].items():assert sha(f)==h
 obj=root/'obj';top='tb_dw_block'
 codegen={f.name:sha(f) for f in obj.iterdir() if f.suffix in ('.cpp','.h','.mk')}
 manifest=dict(audit=review,compile_sha256=sha(root/'compile.log'),codegen=codegen,paired_sha256=sha(a.paired/'result.json'),runner_sha256=sha(__file__))
 (root/'warning-recovery.json').write_text(json.dumps(manifest,indent=2))
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
 make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}',
 'PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',
 'OPT_FAST=-O1 -munroll-only-small-loops -mvzeroupper','OPT_GLOBAL=-O1 -munroll-only-small-loops -mvzeroupper','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
 (root/'make-arguments.json').write_text(json.dumps(list(map(str,make)),indent=2))
 body,build_seconds=run(make,obj,root/'build.log',env)
 warnings=[line for line in body.splitlines() if 'warning:' in line];assert len(warnings)==2
 assert all(re.fullmatch(r".*include/verilated\.cpp:(78|79):10: warning: '(STDOUT|STDERR)_FILENO' redefined",line) for line in warnings)
 binary=obj/('V'+top+'.exe');runs=[]
 for reference in paired['runs']:
  step=reference['phase_step'];log=root/('simulation_step'+str(step)+'.log')
  body,seconds=run([binary,'+VECTORS='+(root/'vectors').as_posix(),'+PHASE_STEP='+str(step)],root,log,env)
  assert not re.search(r'\bFAIL\b|%Error',body)
  passed=re.findall(r'^(PASS DW_BLOCK.*)$',body,re.M);rows=re.findall(r'^BLOCK (.*)$',body,re.M)
  assert len(passed)==1 and rows==reference['rows'] and passed[0]==reference['line'],'source/map mismatch'
  runs.append(dict(phase_step=step,line=passed[0],rows=rows,seconds=seconds,log_sha256=sha(log)))
 for name,h in codegen.items():assert sha(obj/name)==h
 for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
 result=dict(status='LOCAL_DW_BLOCK_NATIVE_MAP_PAIRED_PASS',mapped=True,runs=runs,warning_review=review,warning_codes=review['codes'],
  inputs_sha256=sha(root/'inputs.json'),recovery_sha256=sha(root/'warning-recovery.json'),binary_sha256=sha(binary),build_seconds=build_seconds,
  two_state=True,warning_review_open=False,full_frame=False,pnr=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)
if __name__=='__main__':main()
