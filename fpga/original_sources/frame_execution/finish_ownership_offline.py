"""Finish a full-function 150MHz candidate offline under an explicit seed policy."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree
from audit_canonical_multiseed import audit
from ownership_evidence import bind_proofs

def main():
    p=argparse.ArgumentParser()
    for n in ('board','batch','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--selected-seed',type=int,help='Choose a passing seed explicitly; all four must still pass.')
    p.add_argument('--reuse-gray',type=Path,help='Reuse a completed CDC diagnostic on exactly the selected route.')
    p.add_argument('--selected-canonical',type=Path,help='Use the user-approved selected-route policy and preserve all other results.')
    a=p.parse_args();board=a.board.resolve();batch=a.batch.resolve();output=a.output.resolve()
    selected_receipt=None
    if a.selected_canonical:
        from audit_ownership_selection import selection
        selected_receipt=selection(board,batch,a.selected_canonical)
        rows=[selected_receipt['selected']]
    else:
        terminal=json.loads((batch/'result.json').read_text())
        assert terminal['status']=='REGISTERED_OWNERSHIP_FOUR_SEED_PASS_PENDING_CDC'
        assert sha(terminal['evidence'])==terminal['evidence_sha256']
        receipt=json.loads(Path(terminal['evidence']).read_text())
        rows=[audit(Path(r['canonical'])) for r in receipt['rows']]
        assert len(rows)==4 and len({r['seed'] for r in rows})==4
        assert all(r['current_timing_pass'] and r['identity']==receipt['identity'] for r in rows)
    bind_proofs(board,{})
    selected=(next(r for r in rows if r['seed']==a.selected_seed) if a.selected_seed is not None
              else max(rows,key=lambda r:(min(r['wns_ns'],r['whs_ns']),r['wns_ns'])))
    physical=Path(selected['physical']);canonical=Path(selected['canonical'])
    out=batch/'offline-finish';assert not out.exists();out.mkdir()
    (out/'runner.py').write_bytes(Path(__file__).read_bytes())
    (out/'selection.json').write_text(json.dumps(selected,indent=2));stages=[]
    if selected_receipt:
        (out/'selection_audit.json').write_text(json.dumps(selected_receipt,indent=2))
    def run(name,args,budget,cwd=ROOT):
        start=time.monotonic();logfile=out/(name+'.log')
        with logfile.open('x') as log:
            proc=subprocess.Popen([sys.executable,*map(str,args)],cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
        row=dict(stage=name,exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(logfile))
        stages.append(row);(out/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,row
    try:
        run('host_tests',['-m','unittest','test_control_snapshot_host','-v'],30,HERE)
        (batch/'host_tests.log').write_bytes((out/'host_tests.log').read_bytes())
        gray=a.reuse_gray.resolve() if a.reuse_gray else out/'gray';bit=out/'bit'
        if a.reuse_gray:
            gm=json.loads((gray/'inputs.json').read_text());gr=json.loads((gray/'result.json').read_text())
            assert Path(gm['base'])==physical
            assert gr['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
            assert gr['inputs_sha256']==sha(gray/'inputs.json')
            for path,digest in gm['original'].items():assert sha(path)==digest
            (out/'reused_gray.json').write_text(json.dumps(dict(root=str(gray),result_sha256=sha(gray/'result.json')),indent=2))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(run,'bit',[HERE/'prepare_routed_bitstream.py','--canonical',canonical,'--root',bit],300)]
            if not a.reuse_gray:
                jobs.append(pool.submit(run,'gray',[HERE/'audit_snapshot_routed_gray.py','--base',physical,'--root',gray],370))
            for job in jobs:job.result()
        review=out/'cdc_review.json'
        run('review',[HERE/'review_ownership_cdc.py','--board',board,'--batch',batch,'--gray',gray,'--output',review],100)
        bundle_args=[HERE/'prepare_ownership_release.py','--board',board,'--batch',batch,'--gray',gray,
                      '--bitroot',bit,'--review',review,'--output',output]
        if selected_receipt:bundle_args+=['--selection-audit',out/'selection_audit.json']
        run('bundle',bundle_args,100)
        run('validate',[output/'program.py','--report',out/'offline-validation.json'],60)
        result=dict(status='FULL_150MHZ_OFFLINE_READY_AWAIT_SUPERVISION',bundle=str(output),selected=selected,
                    manifest_sha256=sha(output/'manifest.json'),hardware_access=False,stages=stages)
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
