"""Finish the selected69 offline gates sequentially; never access a board.

Waits for the already running Gray gate, then uses existing validated release
tools. Any failure stops the sequence and preserves the prefix. No automatic
programming, retries, source changes or alternate implementations are allowed.
"""
import json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree_safe import stop_launched_tree


def main():
    base=ROOT/'tmp/frame-execution'
    root=base/'video-input69-offline-finish1'
    root.mkdir(exist_ok=False)
    gray=base/'video-input-gray2'; pll=base/'video-input-pll2'
    canonical=base/'video-input-canonical2'; bits=base/'video-input-bit2'
    board=base/'connected-board-video-service-probe60-input-isp-explicit'
    bundle=ROOT/'output/deployment/board-full-150-video-service69-input-isp-20261017'
    stages=[]
    def save(name,value):
        (root/name).write_text(json.dumps(value,indent=2),encoding='utf8')
    def run(name,args,budget):
        assert not any(str(x) in ('--execute','--supervised','--supervised-program','--worker') for x in args)
        command=[sys.executable,*map(str,args)]
        row=dict(name=name,command=command,budget_seconds=budget,status='running')
        stages.append(row);save('stages.json',stages);start=time.monotonic()
        with (root/(name+'.log')).open('x',encoding='utf8') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            try:rc=child.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                row['cleanup']=stop_launched_tree(child);save('stages.json',stages);raise
        row.update(status='PASS' if rc==0 else 'FAIL',exit_code=rc,seconds=time.monotonic()-start,
                   log_sha256=sha(root/(name+'.log')))
        save('stages.json',stages)
        assert rc==0,(name,rc)
    save('inputs.json',dict(runner_sha256=sha(__file__),gray=str(gray),canonical=str(canonical),
                            bundle=str(bundle),hardware_access=False))
    try:
        deadline=time.monotonic()+1200
        save('waiting.json',dict(status='WAITING_FOR_EXISTING_GRAY_ONLY',hardware_access=False))
        while not (gray/'result.json').exists():
            assert not (gray/'failure.json').exists(),'Current Gray gate failed'
            assert time.monotonic()<deadline,'Wait budget exceeded; original process untouched'
            time.sleep(2)
        proof=json.loads((gray/'result.json').read_text())
        assert proof['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
        assert proof['inputs_sha256']==sha(gray/'inputs.json')
        run('pll-survey',[HERE/'video_service_pll_survey.py','--base',canonical,'--root',pll],180)
        run('pll-bound',[HERE/'video_input_pll.py','--board',board,'--root',pll],300)
        run('bitstream',[HERE/'prepare_routed_bitstream.py','--canonical',canonical,'--root',bits],400)
        run('package',[HERE/'package_video_input_isp.py','--board',board,'--canonical',canonical,
            '--gray',gray,'--pll',pll,'--bitstream-root',bits,'--output',bundle],300)
        report=ROOT/'tmp/board-test/video-service-review1/input69-program-offline.json'
        assert not report.exists()
        run('program-offline',[bundle/'program.py','--report',report],240)
        run('service-offline',[HERE/'video_service_board.py','--manifest',bundle/'manifest.json'],240)
        run('input-isp-offline',[HERE/'video_service_input_isp.py','--manifest',bundle/'manifest.json'],240)
        assert 'OFFLINE_VALIDATED' in (root/'program-offline.log').read_text()
        assert 'SERVICE_OFFLINE_BINDING_PASS_NO_USB' in (root/'service-offline.log').read_text()
        assert 'INPUT_ISP_OFFLINE_VALIDATED_NO_USB' in (root/'input-isp-offline.log').read_text()
        save('result.json',dict(status='OFFLINE_GATES_COMPLETE_STOP_BEFORE_SUPERVISED_BOARD',
            bundle=str(bundle),manifest_sha256=sha(bundle/'manifest.json'),
            bitstream_sha256=sha(bundle/'ti60f225_oob.bit'),stages=stages,hardware_access=False))
        print('OFFLINE_GATES_COMPLETE_STOP_BEFORE_SUPERVISED_BOARD',flush=True)
    except Exception as exc:
        save('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False))
        raise


if __name__=='__main__':main()
