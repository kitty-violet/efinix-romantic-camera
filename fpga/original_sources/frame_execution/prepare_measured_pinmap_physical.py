"""Two concurrent offline tool processes, four seeds; never access hardware."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from run_units import ROOT,HERE,sha
from board_measured_pinmap import audit as observation
from windows_process_tree import stop_launched_tree


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resume-first-pair',type=Path,
                   help='Reuse saved seed1 retry STA and seed3 PASS without rerouting either')
    a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    gate=observation(board)
    assert gate['status']=='MEASURED_GPIO_DELTA_SAME_RTL_MAP_PASS'
    root.mkdir();(root/'runner.py').write_bytes(Path(__file__).read_bytes())
    records=[];lock=threading.Lock();sta_lock=threading.Lock()
    def record(row):
        with lock:
            records.append(row)
            (root/'stages.json').write_text(json.dumps(records,indent=2))
    def run(name,args,budget):
        start=time.monotonic()
        with (root/(name+'.log')).open('w') as log:
            child=subprocess.Popen([sys.executable,str(HERE/args[0]),*map(str,args[1:])],cwd=ROOT,
                                   stdout=log,stderr=subprocess.STDOUT)
            record(dict(stage=name,status='RUNNING',pid=child.pid,budget_s=budget))
            try:
                code=child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                cleanup=stop_launched_tree(child)
                record(dict(stage=name,status='HOST_TIMEOUT',cleanup=cleanup))
                raise
        row=dict(stage=name,status='EXIT_ZERO' if code==0 else 'FAIL',exit_code=code,
                 seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log')))
        record(row)
        assert code==0, row
    def seed_run(seed):
        physical=root/f'pnr-seed{seed}';canonical=root/f'canonical-seed{seed}'
        run(f'pnr{seed}',['probe_measured_pinmap_physical.py','--base',board,'--root',physical,
            '--logic-limit','60800','--optimization','CONGESTION_1','--seed',seed,'--fixed-dsp-layout',
            '--clock-profile-proof',ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json',
            '--setup-reserve-proof',ROOT/'tmp/frame-execution/setup-margin-proof2/result.json'],2220)
        terminal=json.loads((physical/'result.json').read_text())
        assert any(s['flow']=='pnr' and s['exit_code']==0 for s in terminal['stages'])
        # Reporting in separate processes is serialized after the vendor report
        # crash; routed geometry and all timing checks stay identical.
        with sta_lock:
            run(f'sta{seed}',['canonical_routed_sta.py','--base',physical,'--root',canonical],250)
        sta=json.loads((canonical/'result.json').read_text())
        assert sta['status']=='CANONICAL_ROUTED_TIMING_PASS_NOT_RELEASE',sta
        return dict(seed=seed,physical=str(physical),canonical=str(canonical),
                    wns=sta['rows']['setup']['worst_ns'],whs=sta['rows']['hold']['worst_ns'])
    (root/'inputs.json').write_text(json.dumps(dict(board=str(board),observation=gate,
        runner_sha256=sha(__file__),seeds=[1,3,2,4],parallel_processes=2,parallel_sta=1,
        resume_first_pair=str(a.resume_first_pair.resolve()) if a.resume_first_pair else None,
        user_requires_supervision_before_hardware=True,hardware_access=False),indent=2))
    start=time.monotonic()
    try:
        rows=[]
        pairs=((1,3),(2,4))
        if a.resume_first_pair:
            from audit_canonical_multiseed import audit
            previous=a.resume_first_pair.resolve()
            assert previous.is_relative_to(ROOT/'tmp') and (previous/'failure.json').exists()
            for seed,name in ((1,'canonical-seed1-retry1'),(3,'canonical-seed3')):
                canonical=previous/name
                checked=audit(canonical)
                assert checked['seed']==seed and checked['current_timing_pass']
                assert checked['identity']['map']==gate['map_sha256']
                rows.append(dict(seed=seed,physical=checked['physical'],canonical=str(canonical),
                                 wns=checked['wns_ns'],whs=checked['whs_ns']))
                record(dict(stage=f'reuse{seed}',status='VERIFIED_SAVED_PASS',
                            canonical=str(canonical),receipt_sha256=sha(canonical/'result.json')))
            pairs=((2,4),)
        for pair in pairs:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(seed_run,s) for s in pair]
                rows.extend(f.result() for f in futures)
        receipt=ROOT/'output/toolchain/实测引脚修正版四seed原约束_2026-10-09.json'
        run('multiseed',['audit_canonical_multiseed.py','--roots',*[r['canonical'] for r in rows],
                        '--output',receipt],45)
        selected=max(rows,key=lambda r:(r['whs'],r['wns']))
        gray=root/'selected-gray'
        run('gray',['audit_bringup_routed_gray.py','--base',selected['physical'],'--root',gray],370)
        g=json.loads((gray/'result.json').read_text())
        assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
        run('bitstream',['prepare_routed_bitstream.py','--canonical',selected['canonical'],
                         '--root',root/'offline-bitstream'],320)
        result=dict(status='OFFLINE_PHYSICAL_EVIDENCE_COMPLETE_AWAIT_REVIEW',seeds=rows,selected=selected,
            multiseed_receipt=str(receipt),gray=str(gray),bitstream=str(root/'offline-bitstream'),
            elapsed_s=time.monotonic()-start,board_release=False,hardware_access=False,
            next='Review receipts and prepare read-only diagnostic release; STOP before hardware for user supervision')
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        result=dict(status='OFFLINE_BATCH_STOPPED',error=repr(exc),traceback=traceback.format_exc(),
                    elapsed_s=time.monotonic()-start,hardware_access=False)
        (root/'failure.json').write_text(json.dumps(result,indent=2));raise


if __name__=='__main__':main()
