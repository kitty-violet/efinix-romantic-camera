"""Three bounded affected mapped checks; never runs P&R or hardware."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_local_control_decode_map import audit
from windows_process_tree import stop_launched_tree

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();checked=audit(board)
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),
        map_sha256=checked['map_sha256'],runner_sha256=sha(__file__),hardware_access=False),indent=2))
    stages=[]
    def run(name,args,budget):
        start=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            row=dict(name=name,pid=proc.pid,status='running',budget_s=budget);stages.append(row)
            (root/'stages.json').write_text(json.dumps(stages,indent=2))
            try:code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                (root/(name+'-timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
        row.update(status='finished',exit_code=code,seconds=time.monotonic()-start)
        (root/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,code)
    try:
        run('startup',[HERE/'validate_local_decode_board.py','startup','--board',board,'--root',root/'startup'],180)
        run('observe',[HERE/'validate_local_decode_board.py','observe','--source',root/'startup','--root',root/'observe'],120)
        serial=root.parent/(root.name+'-serial')
        run('serial',[HERE/'validate_local_decode_board.py','serial','--board',board,'--root',serial,'--startup',root/'startup'],300)
        (root/'result.json').write_text(json.dumps(dict(status='LOCAL_DECODE_MAPPED_STARTUP_OBSERVATION_SERIAL_PASS',
            map_sha256=checked['map_sha256'],stages=stages,hardware_access=False,board_release=False),indent=2))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
