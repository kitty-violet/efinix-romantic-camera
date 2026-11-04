"""Isolated C8/C16 residual support; never edits the shared hardware tree."""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .validate_region_native import NATIVE,BIN

VROOT=BIN.parent/'share/verilator'

def run(args,cwd,log,timeout=240):
    env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(cwd),TMP=str(cwd),
        MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
    start=time.perf_counter()
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    assert r.returncode==0,(str(log),r.returncode)
    return time.perf_counter()-start

def build(root,files,top):
    obj=root/'obj'
    args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,
        '--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),*[f.as_posix() for f in files]]
    write_json(root/'compile-arguments.json',list(map(str,args)));seconds=run(args,root,root/'compile.log')
    body=(root/'compile.log').read_text();warnings=re.findall(r'%Warning-([A-Z0-9_]+)',body)
    assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT','MULTIDRIVEN'}),warnings
    opts='-O1 -munroll-only-small-loops -mvzeroupper'
    make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}',
        'PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',
        f'OPT_FAST={opts}',f'OPT_GLOBAL={opts}','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
    seconds+=run(make,obj,root/'build.log',600);binary=obj/('V'+top+'.exe')
    write_json(root/'binary.json',dict(binary=str(binary),binary_sha256=file_hash(binary),build_seconds=seconds,warnings=warnings,fit=False,timing=False))
    print('REGION_RTL_BUILT',binary,seconds,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='rtl-variable-groups');a=p.parse_args()
    root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);src=root/'frozen';src.mkdir();records=[]
    for f in (NATIVE/'frozen').iterdir():
        if f.suffix not in ('.v','.sv'):continue
        dst=src/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),source_sha256=file_hash(f),snapshot=str(dst)))
    core=src/'compute_column_cluster.sv';s=core.read_text();old='cfg_in_groups!=4 || cfg_out_groups!=4'
    assert s.count(old)==1;s=s.replace(old,'(cfg_in_groups!=1 && cfg_in_groups!=2 && cfg_in_groups!=4) || cfg_in_groups!=cfg_out_groups');core.write_text(s)
    fuse=src/'pw_skip_fuse4.sv';s=fuse.read_text();assert s.count('groups!=4 ||')==1
    s=s.replace('groups!=4 ||','(groups!=1 && groups!=2 && groups!=4) ||')
    old="{1'b0,sx[9:2],2'd0}";assert s.count(old)==1
    s=s.replace(old,"(groups==1 ? {3'd0,sx[9:2]} : groups==2 ? {2'd0,sx[9:2],1'b0} : {1'b0,sx[9:2],2'd0})")
    s=s.replace('tag[15:13]!=0 || y!=',"tag[15:13]!=0 || {1'b0,g}>=groups || y!=")
    fuse.write_text(s)
    for r in records:r['sha256']=file_hash(r['snapshot'])
    write_json(src/'inputs.json',dict(inputs=records,changed=['compute_column_cluster.sv','pw_skip_fuse4.sv'],production_modified=False))
    build(root,[Path(r['snapshot']) for r in records],'tb_compute_column_cluster')

if __name__=='__main__':main()
