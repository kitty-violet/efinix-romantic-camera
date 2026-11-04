"""Paired two-state source/map verification, unchanged official primitive models.
Compile once per netlist and run both full45-shape and actual fusion-width suites.
Four-state source and bounded mapped evidence remain separate mandatory gates.
"""
import argparse,collections,json,os,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
def main():
 p=argparse.ArgumentParser()
 for key in ('root','source','profile'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--map',type=Path);p.add_argument('--paired',type=Path);p.add_argument('--no-inline-front',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False);src=root/'source';src.mkdir()
 records=[];compiled=[]
 try:
  sr=json.loads((a.source/'result.json').read_text());sm=json.loads((a.source/'inputs.json').read_text())
  assert sr['status']=='LOCAL_RESIDUAL_REAL_ARENA_SHARED_POOL_ROW_PASS' and sha(a.source/'inputs.json')==sr['inputs_sha256']
  byname={Path(r['path']).name:r for r in sm['inputs']};net=None
  if a.map:
   assert a.paired
   mr=json.loads((a.map/'result.json').read_text());mi=json.loads((a.map/'inputs.json').read_text())
   assert mr['status']=='LOCAL_RESIDUAL_ROW_ADAPTER_MAP_PASS' and sha(a.map/'inputs.json')==mr['inputs_sha256']
   for r in mi['inputs']:assert r['sha256']==byname[Path(r['path']).name]['sha256']
   net=a.map/'outflow/residual_row_shared.map.v';assert sha(net)==mr['map_sha256']
  def copy(path,expected=None,compile=True):
   f=Path(path);h=sha(f)
   if expected:assert h==expected
   dst=src/f.name;assert not dst.exists();shutil.copy2(f,dst)
   records.append(dict(path=str(f),snapshot=str(dst),sha256=h))
   if compile:compiled.append(dst.as_posix())
  kinds={'DSP24','RAM10','DPRAM10'}
  if net:kinds|=set(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M))
  for k in sorted(kinds):copy(EF/'sim_models/verilog'/('efx_'+k.lower()+'.v'))
  for r in sm['inputs']:
   name=Path(r['path']).name
   if not name.endswith('.sv') or name.startswith('tb_'):continue
   if net and name in ('residual_packet_dispatch32.sv','residual_row_shared.sv'):continue
   copy(r['snapshot'],r['sha256'])
  if net:copy(net,mr['map_sha256'])
  copy(HERE/'tb_residual_row_mapped.sv')
  if a.no_inline_front:copy(HERE/'residual_no_inline_front.vlt')
  copy(Path(__file__),compile=False)
  families={}
  for family,origin in (('full',a.source),('profile',a.profile)):
   r=json.loads((origin/'result.json').read_text());m=json.loads((origin/'inputs.json').read_text())
   assert sha(origin/'inputs.json')==r['inputs_sha256']
   dest=root/('vectors_'+family);shutil.copytree(origin/'vectors',dest)
   for name,h in m['vectors'].items():assert sha(dest/name)==h
   families[family]=dict(path=dest.as_posix(),hashes=m['vectors'],reference=str(origin/'result.json'),reference_sha256=sha(origin/'result.json'),oracle=r['oracle'])
  env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),
   MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
  top='tb_residual_row_shared';obj=root/'obj'
  args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,
   '--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal',
   '--Mdir',obj.as_posix(),'-GSEED=71',*compiled]
  tools={str(f):sha(f) for f in (BIN/'verilator_bin.exe',BIN/'g++.exe',BIN/'mingw32-make.exe',BIN/'ar.exe',Path(sys.executable))}
  manifest=dict(inputs=records,tools=tools,mapped=bool(net),families=families,map_sha256=sha(net) if net else None)
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  (root/'compile-arguments.json').write_text(json.dumps(list(map(str,args)),indent=2))
  body,compile_seconds=run(args,root,root/'compile.log',env)
  codes=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M))
  assert not (set(codes)&{'UNOPTFLAT','SELRANGE','LATCH','IMPLICIT'}),codes
  options='-O1 -munroll-only-small-loops -mvzeroupper'
  make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',
   f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}',
   'SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',f'OPT_FAST={options}',f'OPT_GLOBAL={options}',
   'CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
  (root/'make-arguments.json').write_text(json.dumps(list(map(str,make)),indent=2))
  _,build_seconds=run(make,obj,root/'build.log',env)
  binary=obj/('V'+top+'.exe')
  if not binary.exists():binary=obj/('V'+top)
  results={}
  for family,data in families.items():
   body,seconds=run([binary,'+VECTORS='+data['path']],root,root/('simulation_'+family+'.log'),env)
   assert not re.search(r'\bFAIL\b|%Error',body)
   lines=re.findall(r'^(PASS RESIDUAL_ROW.*)$',body,re.M);assert len(lines)==1
   rows=re.findall(r'^ROW (.*)$',body,re.M);assert len(rows)==data['oracle']['cases']+1
   trace=re.findall(r'^(?:ROW .*|PASS RESIDUAL_ROW.*)$',body,re.M)
   results[family]=dict(line=lines[0],rows=rows,trace=trace,seconds=seconds,log_sha256=sha(root/('simulation_'+family+'.log')))
  if a.paired:
   paired=json.loads((a.paired/'result.json').read_text())
   assert paired['status']=='LOCAL_RESIDUAL_ROW_NATIVE_SOURCE_PASS' and not paired['mapped']
   for family in results:assert results[family]['trace']==paired['results'][family]['trace'],family+' source/map mismatch'
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  for f,h in tools.items():assert sha(f)==h
  result=dict(status='LOCAL_RESIDUAL_ROW_NATIVE_MAP_PAIRED_PASS' if net else 'LOCAL_RESIDUAL_ROW_NATIVE_SOURCE_PASS',mapped=bool(net),
   results=results,inputs_sha256=sha(root/'inputs.json'),binary_sha256=sha(binary),warning_codes=dict(codes),
   compile_seconds=compile_seconds,build_seconds=build_seconds,two_state=True,warning_review_open=True,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','warning_codes','compile_seconds','build_seconds')}))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

