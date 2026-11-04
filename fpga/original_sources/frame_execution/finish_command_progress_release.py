"""Finish a passing OFP2 implementation offline; never access hardware."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree import stop_launched_tree

def main():
    p=argparse.ArgumentParser()
    for name in ('board','batch','paired','root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();board=a.board.resolve();batch=a.batch.resolve();paired=a.paired.resolve();root=a.root.resolve()
    terminal=json.loads((batch/'result.json').read_text())
    assert terminal['status']=='COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE'
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir();(root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(dict(board=str(board),batch=str(batch),batch_result_sha256=sha(batch/'result.json'),paired=str(paired),paired_result_sha256=sha(paired/'paired_reply_result.json'),runner_sha256=sha(__file__),hardware_access=False),indent=2))
    stages=[]
    def run(name,script,args,budget):
        started=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(name=name,status='running',pid=proc.pid,budget_s=budget);stages.append(row);(root/'stages.json').write_text(json.dumps(stages,indent=2))
            try:code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                (root/(name+'-timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
        row.update(status='finished',exit_code=code,seconds=time.monotonic()-started);(root/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,code)
    try:
        gray=Path(terminal['gray']);canonical=Path(terminal['selected']['canonical'])
        run('functional_cdc','audit_functional_jtag_paths.py',['--root',gray],45)
        run('pll_arrival','survey_pll_shift_routed.py',['--base',canonical,'--root',root/'pll-paths'],160)
        run('pll_bounds','audit_command_progress_pll.py',['--board',board,'--root',root/'pll-paths'],45)
        run('bitstream','prepare_routed_bitstream.py',['--canonical',canonical,'--root',root/'bitstream'],330)
        run('bundle','prepare_command_progress_release.py',['--board',board,'--batch',batch,'--paired',paired,'--shiftpaths',root/'pll-paths','--bitroot',root/'bitstream','--output',a.output.resolve()],90)
        manifest=a.output.resolve()/'manifest.json'
        (root/'result.json').write_text(json.dumps(dict(status='OFFLINE_READY_AWAIT_USER_SUPERVISION',bundle=str(a.output.resolve()),manifest_sha256=sha(manifest),hardware_access=False,automatic_download_allowed=False),indent=2))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise

if __name__=='__main__':main()
