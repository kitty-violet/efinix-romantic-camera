"""Snapshot, run actual official primitive tests, and fail on any simulator error."""
import argparse,hashlib,json,re,shutil,subprocess,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
MS=Path('E:/altera/15.0/modelsim_ase/win32aloem')
EF=Path('E:/efinity-2026.1.132-windows-x64/efinity/2026.1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args,build,name,timeout=600):
    t=time.monotonic()
    with (build/name).open('w',encoding='utf-8') as f:r=subprocess.run(list(map(str,args)),cwd=build,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    body=(build/name).read_text()
    if r.returncode or re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body):raise RuntimeError(name+body[-2000:])
    return body,time.monotonic()-t
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['meter','arena'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[71,391,1027]);a=p.parse_args()
    build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
    stem='frame_execution_meter' if a.kind=='meter' else 'tensor_arena12'
    paths=[HERE/(stem+'.sv'),HERE/('tb_'+stem+'.sv')]
    if a.kind=='arena':paths.insert(0,EF/'sim_models/verilog/efx_dpram10.v')
    records=[]
    for src in paths:
        dst=build/src.name;shutil.copy2(src,dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(src)))
    manifest=dict(inputs=records,runner_sha256=sha(__file__),seeds=a.seeds,kind=a.kind)
    (build/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        run([MS/'vlib.exe','work'],build,'vlib.log')
        body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[Path(r['snapshot']) for r in records]],build,'compile.log')
        assert 'Errors: 0, Warnings: 0' in body
        results=[]
        for seed in (a.seeds if a.kind=='arena' else [0]):
            args=[MS/'vsim.exe','-c','work.tb_'+stem]
            if a.kind=='arena':args += ['-gSEED='+str(seed)]
            args+=['-do','onerror {quit -code 1}; run -all; quit -code 0']
            body,seconds=run(args,build,f'seed{seed}.log')
            lines=re.findall(r'^# (PASS (?:TENSOR_ARENA|FRAME_METER).*)$',body,re.M)
            assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
            results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(build/f'seed{seed}.log')))
        for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
        assert sha(__file__)==manifest['runner_sha256']
        result=dict(status='LOCAL_OFFICIAL_PRIMITIVE_PASS' if a.kind=='arena' else 'LOCAL_METER_PASS',
                    results=results,inputs_sha256=sha(build/'inputs.json'),full_frame_rtl=False,pnr=False)
        (build/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    except Exception as e:
        (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

