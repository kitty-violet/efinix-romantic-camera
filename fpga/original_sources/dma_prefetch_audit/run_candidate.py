from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time
R=Path(__file__).resolve().parent;P=Path('E:/fpga')
sys.path.insert(0,str(P/'hw/efinity/prototypes/frame_execution'))
from owned_simulator_process import spawn_owned,stop_owned_tree,close_owned
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 d=R/('candidate-trace'+sys.argv[1]);d.mkdir(exist_ok=False)
 limit=int(sys.argv[2]);assert 1<=limit<=6546
 prim=Path('E:/efinity-2026.1.132-windows-x64/efinity/2026.1/sim_models/verilog')
 src=list((R/'rtl').glob('*.sv'))+[prim/n for n in ['efx_add.v','efx_lut4.v','efx_srl8.v','efx_dsp48.v','efx_dsp24.v','efx_ram10.v','efx_dpram10.v']]+[R/'tb_candidate.sv']
 inputs={str(p):sha(p)for p in src};(d/'inputs.json').write_text(json.dumps(inputs,indent=2),'utf8')
 b=P/'tools/verilator/ucrt64/bin';vr=b.parent/'share/verilator';obj=d/'obj';top='tb_s4r1_frame'
 env=dict(os.environ,PATH=str(b)+';E:/GIT/Git/usr/bin;'+os.environ['PATH'],TEMP=str(d),TMP=str(d),TMPDIR=str(d),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(vr),MAKE=str(b/'mingw32-make.exe'))
 cmds=[('compile',[str(b/'verilator_bin.exe'),'--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,'--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),*[p.as_posix()for p in src]],d,90),
 ('build',[str(b/'mingw32-make.exe'),'-f','V'+top+'.mk','-j4',f'CXX={(b/"g++.exe").as_posix()}',f'LINK={(b/"g++.exe").as_posix()}',f'AR={(b/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3="{Path(sys.executable).as_posix()}"','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={vr.as_posix()}','OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch'],obj,240),
 ('test',[str(obj/('V'+top+'.exe')),'+VECTORS='+(R/'candidate-vectors').as_posix(),'+MAX_COMMANDS='+str(limit if limit<6546 else 0)],d,90 if limit<6546 else 900)]
 stages=[]
 for name,cmd,cwd,budget in cmds:
  st=time.monotonic()
  with(d/(name+'.log')).open('w',encoding='utf8')as log:
   p=spawn_owned(cmd,cwd=cwd,env=env,stdout=log,stderr=-2)
   try:p.wait(timeout=budget)
   except subprocess.TimeoutExpired:stop_owned_tree(p);p.wait(timeout=10);raise
   finally:close_owned(p)
  body=(d/(name+'.log')).read_text('utf8',errors='replace');stages.append(dict(stage=name,exit=p.returncode,seconds=time.monotonic()-st))
  (d/'stages.json').write_text(json.dumps(stages,indent=2),'utf8')
  if p.returncode:print(body[-4000:]);raise RuntimeError(name)
  if name=='compile':assert not re.search(r'%Warning-(IMPLICIT|LATCH|UNOPTFLAT|SELRANGE|PINMISSING)',body),body[-4000:]
  print(name,'PASS',round(time.monotonic()-st,2),flush=True)
 assert ('PASS BOUNDED_FRAME_PREFIX'if limit<6546 else'PASS S4R1_FULL_COMPUTE_FRAME') in body and all(sha(Path(n))==h for n,h in inputs.items())
 (d/'result.json').write_text(json.dumps(dict(status='COMPUTE_FRAME_PREFIX_PASS'if limit<6546 else'FULL_COMPUTE_FRAME_PASS_NOT_BOARD_RELEASE',inputs=inputs,stages=stages,coverage=body.splitlines()[-7:],hardware_access=False,scope='Actual production fetch/issuer/header/store/six-bank RAM/compute/DMA. Stalled transactional DDR model, no physical PHY or video concurrency. Not board FPS or P&R.'),indent=2),'utf8')
 print(body[-1200:])
if __name__=='__main__':main()
