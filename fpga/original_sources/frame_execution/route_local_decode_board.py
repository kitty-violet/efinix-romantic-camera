"""Route an explicitly reviewed native placement once, then original STA.

No route is allowed solely from placement legal/capacity or geometric distance.
The review file is produced after inspecting the actual placement evidence.
"""
import argparse,json,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from local_control_decode import audit_delta
from audit_local_control_decode_map import audit as audit_map
from windows_process_tree import stop_launched_tree

def save(p,value):p.write_text(json.dumps(value,indent=2),encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',type=Path,required=True)
    p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--serial-proof',type=Path,help='Same-map serial result, optionally with verified public model inlining')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--budget',type=int,default=1200);a=p.parse_args()
    physical=a.physical.resolve();root=a.root.resolve();evidence=a.evidence_root.resolve();review_path=a.review.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists() and 60<=a.budget<=1500
    meta=json.loads((physical/'physical_inputs.json').read_text());board=Path(meta['base']);audit_map(board)
    old=json.loads((physical/'result.json').read_text());assert old['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    assert all(s['exit_code']==0 for s in old['stages'])
    assert not (physical/'outflow/ti60f225_oob.troutingtraces').exists()
    review=json.loads(review_path.read_text());assert review['decision']=='PROCEED_TO_BOUNDED_ROUTE'
    assert review['physical']==str(physical) and review['rationale'] and review['remaining_risks']
    assert review['placement_review_sha256']==sha(physical/'placement_review_inputs.json')
    required=review['required_evidence'];assert required
    for path,digest in required.items():assert sha(path)==digest,path
    for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
    assert sha(physical/'outflow/ti60f225_oob.map.v')==meta['base_map_sha256']
    serial=evidence.parent/(evidence.name+'-serial')
    serial_result=a.serial_proof.resolve() if a.serial_proof else serial/'new'
    # All three affected mapped tests must belong to this map, not a parent.
    for folder in (evidence/'startup',serial_result,evidence/'observe'):
        result=json.loads((folder/'result.json').read_text());inputs=json.loads((folder/'inputs.json').read_text())
        assert result['inputs_sha256']==sha(folder/'inputs.json')
        assert inputs.get('source_map_sha256',inputs.get('original_map_sha256'))==meta['base_map_sha256']
        for name,h in inputs['files'].items():assert sha(folder/name)==h
        assert sha(folder/'simulation.log')==result.get('log_sha256',result.get('simulation_sha256'))
    counter_dir=serial_result if a.serial_proof else serial
    checks=json.loads((counter_dir/'progress_counter_checks.json').read_text())
    assert checks['status']=='CONTROL_DDR_MAPPED_SERIAL_IDLE_RESET_PASS'
    assert checks['source_result_sha256']==sha(serial_result/'result.json')
    si=json.loads((serial_result/'inputs.json').read_text())
    if 'simulation_only_inline' in si:
        from inline_official_state_luts import inline
        source=Path(si['source_replay']);proof=si['simulation_only_inline']
        assert sha(source/'inputs.json')==si['source_inputs_sha256']
        assert proof['transform_sha256']==sha(HERE/'inline_official_state_luts.py')
        for name,h in proof['original_files'].items():
            assert sha(source/name)==h
            if name!='mapped_control.v':assert sha(serial_result/name)==h
        expected,_=inline((source/'mapped_control.v').read_text(),source/'efx_lut4.v',lambda n:n.startswith('\\cpu/ctrl_decode/'))
        assert (serial_result/'mapped_control.v').read_text()==expected
    root.mkdir();shutil.copy2(__file__,root/'runner.py')
    frozen={str(p):sha(p) for p in (physical/'outflow/ti60f225_oob.map.v',physical/'outflow/ti60f225_oob.place',
        physical/'outflow/ti60f225_oob.out.placeloc',physical/'ti60f225_oob.sdc',physical/'ti60f225_oob.xml')}
    save(root/'inputs.json',dict(physical=str(physical),board=str(board),board_inputs_sha256=sha(board/'inputs.json'),
        review=str(review_path),review_sha256=sha(review_path),required_evidence=required,frozen=frozen,
        runner_sha256=sha(__file__),hardware_access=False,board_release=False))
    stages=[]
    def run(name,args,cwd,budget):
        start=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(list(map(str,args)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
            row=dict(name=name,pid=proc.pid,status='running',budget_s=budget);stages.append(row);save(root/'stages.json',stages)
            try:code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:save(root/(name+'-timeout.json'),stop_launched_tree(proc));raise
        row.update(status='finished',exit_code=code,seconds=time.monotonic()-start);save(root/'stages.json',stages)
        assert code==0,(name,code)
        return row
    try:
        before=physical/'before_route';before.mkdir()
        for n in ('ti60f225_oob.place.out','ti60f225_oob.place.rpt','ti60f225_oob.place.timing.rpt','ti60f225_oob.out.placeloc'):
            if (physical/'outflow'/n).exists():shutil.copy2(physical/'outflow'/n,before/n)
        shutil.copy2(physical/'result.json',before/'result.json')
        cmd=physical/'route_only.cmd';assert not cmd.exists()
        cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts route max_router_iterations=100 troute_options=max_iterations=100 --timeout '+str(a.budget)+'\nexit /b %errorlevel%\n')
        row=run('route',['cmd.exe','/d','/c',cmd],physical,a.budget+45)
        assert re.search(r'pnr\s*:\s*PASS',(root/'route.log').read_text())
        route=physical/'outflow/ti60f225_oob.route.out';body=route.read_text()
        iterations=re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)',body)
        assert iterations and int(iterations[-1][1])==0
        assert (physical/'outflow/ti60f225_oob.troutingtraces').exists()
        for path,h in frozen.items():assert sha(path)==h,path
        ps=old['stages']+[dict(flow='pnr',phase='route_after_review',exit_code=0,seconds=row['seconds'],log_sha256=sha(root/'route.log'))]
        save(physical/'stages.json',ps)
        save(physical/'result.json',dict(status='PHYSICAL_CAPACITY_PROBE_FINISHED',stages=ps,
            strict_sta_pass=False,board_download_allowed=False,source_unchanged=True))
        import audit_control_state_implementation as impl
        impl.audit_delta=audit_delta;save(root/'implementation.json',impl.audit(board,physical))
        canonical=root/'canonical-seed3'
        run('canonical',[sys.executable,HERE/'canonical_snapshot_sta.py','--base',physical,'--root',canonical],ROOT,300)
        from audit_canonical_multiseed import audit
        selected=audit(canonical);save(root/'selected.json',selected)
        assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
        run('gray',[sys.executable,HERE/'audit_command_progress_gray.py','--base',physical,'--root',root/'gray-seed3'],ROOT,400)
        audit_delta(board)
        save(root/'result.json',dict(status='LOCAL_DECODE_ROUTED_ORIGINAL_STA_GRAY_PASS_PENDING_RELEASE_AUDITS',
            selected=selected,iterations=int(iterations[-1][0]),hardware_access=False,board_release=False))
    except Exception as exc:
        save(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False,board_release=False));raise
if __name__=='__main__':main()
