"""Bind the complete current CNN source set and replay immutable accepted A.

Behavioral AXI memory; these checks are not whole-board or PHY evidence.
"""
import argparse,collections,json,os,re,shutil,subprocess,sys,time,traceback,zlib
from pathlib import Path
from run_units import ROOT,HERE,sha
from registered_ownership_board import audit_delta
from accepted_a_assets import accepted_package,PACKAGE_SHA
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cluster',type=Path);p.add_argument('--reuse',type=Path);p.add_argument('--full',action='store_true')
    a=p.parse_args();board=a.board.resolve();root=a.root.resolve();audit_delta(board);accepted_package()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir();started=time.monotonic()
    integration=ROOT/'tmp/frame-execution/accepted-a-integration1'
    env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),
        MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
    try:
        if a.cluster:
            cp=a.cluster.resolve();cr=json.loads((cp/'result.json').read_text());ci=json.loads((cp/'inputs.json').read_text())
            assert cr['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS' and cr['inputs_sha256']==sha(cp/'inputs.json')
            assert ci['frozen_board']['inputs_sha256']==sha(board/'inputs.json')
            for rec in ci['inputs']:assert sha(rec['snapshot'])==rec['sha256']
            binary=cp/'obj/Vtb_compute_column_cluster.exe';vector=integration/'cluster-vectors';files=ci['inputs']
        else:
            vector=integration/'val_new_hallway_120';vm=json.loads((vector/'frame_manifest.json').read_text())
            assert vm['package_sha256']==PACKAGE_SHA and vm['accepted_image_error']==0
            for name,digest in vm['files'].items():assert sha(vector/name)==digest
            if a.reuse:
                previous=json.loads((a.reuse/'inputs.json').read_text());prior=json.loads((a.reuse/'result.json').read_text())
                assert prior['status']=='REGISTERED_ACCEPTED_A_SMOKE_PASS' and previous['board_inputs_sha256']==sha(board/'inputs.json')
                assert sha(a.reuse/'inputs.json')==prior['inputs_sha256']
                binary=Path(previous['binary']);files=previous['inputs'];assert sha(binary)==previous['binary_sha256']
                for rec in files:assert sha(rec['snapshot'])==rec['sha256']
            else:
                parent=ROOT/'tmp/frame-execution/skip-credit-frame1';meta=json.loads((parent/'inputs.json').read_text())
                old=json.loads((parent/'result.json').read_text());assert sha(parent/'inputs.json')==old['inputs_sha256']
                source=root/'source';source.mkdir();files=[]
                for rec in meta['inputs']:
                    origin=Path(rec['snapshot']);assert sha(origin)==rec['sha256']
                    candidate=board/'assembly'/origin.name
                    if candidate.is_file():origin=candidate
                    destination=source/origin.name;assert not destination.exists();shutil.copy2(origin,destination)
                    files.append(dict(source=str(origin),snapshot=str(destination),sha256=sha(destination)))
                obj=root/'obj';top='tb_fourway_full';conc=vm['concurrency']
                args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,
                    '-GOPTIMIZED=1','-GCOMMANDS='+str(vm['commands']),'-GEXPECTED_WORDS='+str(vm['expected_compute_words']),
                    '-GCONCURRENT=1','-GCOMPUTE_COMMANDS='+str(conc['compute_commands']),'-GDMA_COMMANDS='+str(conc['dma_commands']),
                    '--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal',
                    '--Mdir',obj.as_posix(),*[Path(rec['snapshot']).as_posix() for rec in files]]
                body,_=run(args,root,root/'compile.log',env,timeout=90)
                warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body))
                assert not set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'},warnings
                make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',
                    f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}',
                    'SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}','OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
                run(make,obj,root/'build.log',env,timeout=420);binary=obj/('V'+top+'.exe')
        hashes={str(f):sha(f) for f in vector.iterdir() if f.is_file()}
        dump=root/'readback.hex';checker=HERE/'accepted_a_assets.py'
        command=subprocess.list2cmdline([sys.executable,'-B',str(checker),'--image-dir',str(vector),'--readback',str(dump)])
        args=[binary,'+VECTORS='+vector.as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+command]
        if not a.cluster:
            descriptors=integration/'descriptors';dm=json.loads((descriptors/'result.json').read_text())
            assert sha(descriptors/'result.json')==vm['descriptor_manifest_sha256']
            for name in ('compute','dma'):
                f=descriptors/(name+'.bin');assert sha(f)==dm['files'][f.name]
                args.append('+'+('CCRC' if name=='compute' else 'DCRC')+'='+format(zlib.crc32(f.read_bytes()),'x'))
            args+=['+DESCRIPTORS='+descriptors.as_posix(),'+FINAL_GOLD_OFFSET='+str(next(n['offset'] for n in vm['nodes'] if n['id']=='rgb_head')),
                   '+PROFILE=1','+LIMIT='+str(19229 if a.full else 1550)]
            hashes.update({str(f):sha(f) for f in descriptors.iterdir() if f.is_file()})
        manifest=dict(board_inputs_sha256=sha(board/'inputs.json'),board=str(board),inputs=files,binary=str(binary),binary_sha256=sha(binary),
                      vector_hashes=hashes,package_sha256=PACKAGE_SHA,runner_sha256=sha(__file__),whole_board=False)
        (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
        body,seconds=run(args,root,root/'simulation.log',env,timeout=180 if a.cluster else 1200 if a.full else 360)
        assert 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and not re.search(r'\bFAIL\b|%Error|\*\* (Error|Fatal)',body)
        if a.cluster:
            assert 'PASS COMPUTE_CLUSTER' in body and 'PASS RESIDUAL_ABORT_PROBE' in body;kind='CLUSTER'
        elif a.full:
            rows=re.findall(r'^PASS CONTINUOUS_720P profile=(\d+) commands=(\d+) cycles=(\d+) scalar_checks=(\d+) final_bytes=(\d+) readbeats=(\d+) writebeats=(\d+) write_responses=(\d+) nodes=([0-9a-f]+)$',body,re.M)
            assert len(rows)==1 and rows[0][0]=='1' and rows[0][1]=='19229' and rows[0][3]=='65952000' and rows[0][4]=='2764800' and rows[0][-1]=='1fffffff'
            kind='FULL'
        else:assert len(re.findall(r'^SMOKE FULL_CHAIN .*$',body,re.M))==1;kind='SMOKE'
        for rec in files:assert sha(rec['snapshot'])==rec['sha256']
        for name,digest in hashes.items():assert sha(name)==digest
        result=dict(status='REGISTERED_ACCEPTED_A_'+kind+'_PASS',simulation_seconds=seconds,total_seconds=time.monotonic()-started,
             inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),package_sha256=PACKAGE_SHA,
             whole_board=False,ddr_phy=False,hardware_access=False,lines=re.findall(r'^(?:PASS|SMOKE) .*$',body,re.M))
        if a.full:result.update(cycles=int(rows[0][2]),cnn_ms=int(rows[0][2])/150000)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
