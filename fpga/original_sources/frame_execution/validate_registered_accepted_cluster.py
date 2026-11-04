"""Replay accepted A on the already-built new-board executable.

The earlier synthetic-domain test is deliberately retained as FAILED. Its
tail coefficients violate the existing A-only shift specialization; reuse
its SHA-bound compiled RTL, not its numerical-pass claim.
"""
import json,os,re,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from accepted_a_assets import accepted_package,PACKAGE_SHA
from validate_registered_board_cluster import audit,BOARD

BUILD=ROOT/'tmp/frame-execution/split-registered-cluster1'
RUN=ROOT/'tmp/frame-execution/split-registered-accepted-cluster1'
VECTOR=ROOT/'tmp/frame-execution/accepted-a-integration1/cluster-vectors'

def main():
    audit();raw,_,_,_=accepted_package();assert not RUN.exists();RUN.mkdir()
    try:
        inputs=json.loads((BUILD/'inputs.json').read_text())
        assert inputs['frozen_board']['inputs_sha256']==sha(BOARD/'inputs.json')
        assert 'Errors: 0, Warnings: 0' in (BUILD/'fourstate_compile.log').read_text()
        for row in inputs['inputs']:assert sha(row['snapshot'])==row['sha256']
        assert (VECTOR/'coefficients-padded.bin').read_bytes()==raw
        binary=BUILD/'obj/Vtb_compute_column_cluster.exe';assert binary.is_file()
        dump=RUN/'readback.hex';checker=HERE/'accepted_a_assets.py'
        check=subprocess.list2cmdline([sys.executable,'-B',str(checker),'--image-dir',str(VECTOR),'--readback',str(dump)])
        command=[str(binary),'+VECTORS='+VECTOR.as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+check]
        bad=(BUILD/'vectors/coefficients.bin').read_bytes()[3008+64:3008+72]
        assert list(bad)==[3,0,252,2,252,0,3,5]
        manifest=dict(board_inputs_sha256=sha(BOARD/'inputs.json'),build_inputs_sha256=sha(BUILD/'inputs.json'),
            binary=str(binary),binary_sha256=sha(binary),inputs=inputs['inputs'],
            vectors={str(p):sha(p) for p in VECTOR.iterdir() if p.is_file()},
            checker_sha256=sha(checker),runner_sha256=sha(__file__),package_sha256=PACKAGE_SHA,
            command=command,prior_synthetic_result='FAILED_NOT_RECLASSIFIED',
            prior_tail_bytes=list(bad),tail_domain=[0,4,8],hardware_access=False)
        (RUN/'inputs.json').write_text(json.dumps(manifest,indent=2))
        env=os.environ.copy();env['PATH']=str(ROOT/'tools/verilator/ucrt64/bin')+';E:/GIT/Git/usr/bin;'+env['PATH']
        start=time.monotonic()
        with (RUN/'simulation.log').open('w',encoding='utf8') as log:
            p=subprocess.run(command,cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        body=(RUN/'simulation.log').read_text();assert p.returncode==0,body[-3500:]
        assert not re.search(r'\bFAIL\b|%Error|\*\* (Error|Fatal)',body)
        assert 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED accepted=B1-A-20261008' in body
        assert 'PASS COMPUTE_CLUSTER' in body and 'PASS RESIDUAL_ABORT_PROBE' in body
        for path,digest in manifest['vectors'].items():assert sha(path)==digest
        for row in inputs['inputs']:assert sha(row['snapshot'])==row['sha256']
        assert sha(binary)==manifest['binary_sha256'];audit()
        result=dict(status='REGISTERED_BOUNDARY_ACCEPTED_A_CLUSTER_PASS',seconds=time.monotonic()-start,
            inputs_sha256=sha(RUN/'inputs.json'),log_sha256=sha(RUN/'simulation.log'),package_sha256=PACKAGE_SHA,
            lines=re.findall(r'^(?:PASS|COVERAGE|BLOCK) .*$',body,re.M),whole_board=False,
            physical_pass=False,board_release=False,hardware_access=False)
        (RUN/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (RUN/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
