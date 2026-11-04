"""Run new vectors on an immutable already-built fourway binary; no rebuilt RTL."""
import argparse,json,os,re,subprocess,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from fourway_vectors import vectors
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import run,BIN,VROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
    native=a.native.resolve();binary=native/'obj/Vtb_cluster_native.exe'
    m=json.loads((native/'inputs.json').read_text())
    for record in m['inputs']:assert sha(record['snapshot'])==record['sha256']
    oracle=vectors(root/'vectors',True);checker=HERE.parent/'coefficient_store/validate_readback.py'
    manifest=dict(binary=str(binary),binary_sha256=sha(binary),native_manifest_sha256=sha(native/'inputs.json'),
                  oracle=oracle,vectors={f.name:sha(f) for f in (root/'vectors').iterdir() if f.is_file()},
                  dependencies={str(f):sha(f) for f in (Path(__file__),HERE/'fourway_vectors.py',checker)})
    (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
    env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],VERILATOR_ROOT=str(VROOT),TEMP=str(root),TMP=str(root),TMPDIR=str(root))
    dump=root/'readback.hex';command=subprocess.list2cmdline([str(Path(sys.executable)),'-B',str(checker),'--image-dir',str(root/'vectors'),'--readback',str(dump)])
    try:
        body,seconds=run([binary,'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+command],root,root/'simulation.log',env)
        passed=re.findall(r'^(PASS COMPUTE_CLUSTER.*)$',body,re.M);rows=re.findall(r'^BLOCK (.*)$',body,re.M)
        assert len(passed)==1 and len(rows)==oracle['cases']*2
        assert not re.search(r'\bFAIL\b|%Error',body) and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body
        assert sha(binary)==manifest['binary_sha256']
        for f,h in manifest['dependencies'].items():assert sha(f)==h
        result=dict(status='LOCAL_FOURWAY_NATIVE_NUMERIC_PROTOCOL_PASS',line=passed[0],rows=rows,seconds=seconds,
                    inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulation.log'),
                    full_frame=False,measured_frame_ms=None,two_state=True,physical_timing_pass=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    except Exception as error:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
