"""Bounded accepted-A replay using previously built, hash-bound RTL executables."""
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path
from accepted_a_assets import ROOT, HERE, PACKAGE_SHA, read_json, write_json, sha, accepted_package


def execute(root, integration, case, kind, profile=0):
    assert not root.exists()
    root.mkdir(parents=True)
    start = time.monotonic()
    try:
        accepted_package()
        cluster = kind == 'cluster'
        frozen = ROOT / ('tmp/frame-execution/skip-credit-cluster1' if cluster else 'tmp/frame-execution/skip-credit-frame1')
        old = read_json(frozen/'inputs.json')
        proof = read_json(frozen/'result.json')
        assert sha(frozen/'inputs.json') == proof['inputs_sha256']
        assert sha(frozen/'simulation.log') == proof['log_sha256']
        for rec in old['inputs']:
            assert sha(rec['snapshot']) == rec['sha256'], rec['snapshot']
        binary = frozen/'obj/Vtb_compute_column_cluster.exe' if cluster else Path(old['binary'])
        if not cluster:
            assert sha(binary) == old['binary_sha256']
        vector = integration / ('cluster-vectors' if cluster else case)
        if not cluster:
            vm = read_json(vector/'frame_manifest.json')
            assert vm['package_sha256'] == PACKAGE_SHA and vm['accepted_image_error'] == 0
            for name, digest in vm['files'].items():
                assert sha(vector/name) == digest, name
        # Freeze all input files. Nothing in the accepted or prior evidence directories is written.
        hashes = {str(p):sha(p) for p in vector.iterdir() if p.is_file()}
        checker = HERE/'accepted_a_assets.py'
        dump = root/'readback.hex'
        command = subprocess.list2cmdline([sys.executable,'-B',str(checker),'--image-dir',str(vector),'--readback',str(dump)])
        args = [str(binary), '+VECTORS='+vector.as_posix(), '+DUMP='+dump.as_posix(), '+VERIFY_COMMAND='+command]
        if not cluster:
            d = integration/'descriptors'
            dm = read_json(d/'result.json')
            assert sha(d/'result.json') == vm['descriptor_manifest_sha256']
            for name in ('compute','dma'):
                assert sha(d/(name+'.bin')) == dm['files'][name+'.bin']
                args.append('+'+('CCRC' if name=='compute' else 'DCRC')+'='+format(zlib.crc32((d/(name+'.bin')).read_bytes()),'x'))
            args += ['+DESCRIPTORS='+d.as_posix(), '+FINAL_GOLD_OFFSET='+str(next(n['offset'] for n in vm['nodes'] if n['id']=='rgb_head')),
                     '+PROFILE='+str(profile), '+LIMIT='+str(1550 if kind=='smoke' else 19229)]
            hashes.update({str(p):sha(p) for p in d.iterdir() if p.is_file()})
        manifest = dict(kind=kind, case=case, profile=profile, binary=str(binary),binary_sha256=sha(binary),
                        prior_inputs=str(frozen/'inputs.json'),prior_inputs_sha256=sha(frozen/'inputs.json'),
                        files=hashes,checker_sha256=sha(checker),runner_sha256=sha(__file__),package_sha256=PACKAGE_SHA)
        write_json(root/'inputs.json',manifest)
        env = os.environ.copy()
        env['PATH'] = str(ROOT/'tools/verilator/ucrt64/bin')+';E:\\GIT\\Git\\usr\\bin;'+env['PATH']
        scratch = root/'temp'; scratch.mkdir(); env.update(TEMP=str(scratch),TMP=str(scratch))
        timeout = 180 if cluster else (360 if kind=='smoke' else 1800)
        with (root/'simulation.log').open('w',encoding='utf-8') as log:
            proc = subprocess.run(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        body = (root/'simulation.log').read_text(encoding='utf-8',errors='replace')
        assert proc.returncode == 0, proc.returncode
        assert not re.search(r'\bFAIL\b|%Error|\*\* (Error|Fatal)',body)
        assert 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body
        result = dict(status='ACCEPTED_A_'+kind.upper()+'_RTL_PASS',case=case,profile=profile,
                      package_sha256=PACKAGE_SHA,seconds=time.monotonic()-start,numerical_errors=0,
                      inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
                      actual_board=False,whole_board=False,ddr_phy=False)
        if cluster:
            assert 'PASS COMPUTE_CLUSTER' in body and 'PASS RESIDUAL_ABORT_PROBE' in body
            result['lines']=re.findall(r'^PASS .*$',body,re.M)
        elif kind=='smoke':
            rows=re.findall(r'^SMOKE FULL_CHAIN .*$',body,re.M);assert len(rows)==1
            result['line']=rows[0]
        else:
            rows=re.findall(r'^PASS CONTINUOUS_720P profile=(\d+) commands=(\d+) cycles=(\d+) scalar_checks=(\d+) final_bytes=(\d+) readbeats=(\d+) writebeats=(\d+) write_responses=(\d+) nodes=([0-9a-f]+)$',body,re.M)
            assert len(rows)==1
            p,commands,cycles,checks,final,*rest=rows[0]
            assert int(p)==profile and int(commands)==19229 and int(checks)==65952000 and int(final)==2764800 and rest[-1]=='1fffffff'
            result.update(cycles=int(cycles),cnn_frame_ms=int(cycles)/150000,scalar_checks=int(checks),final_rgb_bytes=int(final),
                          complete_29_node_graph=True,steady_frame_interval_measured=False,
                          nodes=[dict(id=int(i),cycles_with_header_control=int(c),words=int(w)) for i,c,w in re.findall(r'^NODE id=(\d+) cycles_with_header_control=(\d+) words=(\d+)$',body,re.M)])
        assert sha(binary)==manifest['binary_sha256'] and sha(checker)==manifest['checker_sha256']
        for path,digest in hashes.items():assert sha(path)==digest,path
        write_json(root/'result.json',result)
        print(result['status'],case,round(result['seconds'],2),flush=True)
        return result
    except Exception as e:
        write_json(root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--integration',type=Path,default=ROOT/'tmp/frame-execution/accepted-a-integration1')
    p.add_argument('--kind',choices=['short','full'],required=True);p.add_argument('--jobs',type=int,default=3)
    a=p.parse_args();root=a.root.resolve();integration=a.integration.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists() and 1<=a.jobs<=3
    root.mkdir(parents=True)
    if a.kind=='short':
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(execute,root/'cluster',integration,'cluster','cluster'),
                     pool.submit(execute,root/'smoke',integration,'val_new_hallway_120','smoke',1)]
            results=[f.result() for f in futures]
    else:
        short=read_json(integration/'rtl-short1/result.json');assert short['status']=='ACCEPTED_A_SHORT_RTL_PASS'
        cases=read_json(integration/'result.json')['cases']
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
            futures=[pool.submit(execute,root/c['id'],integration,c['id'],'full',i%2) for i,c in enumerate(cases)]
            results=[f.result() for f in futures]
    write_json(root/'result.json',dict(status='ACCEPTED_A_'+a.kind.upper()+'_RTL_PASS',cases=results,actual_board=False))


if __name__=='__main__':main()
