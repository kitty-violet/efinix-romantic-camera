"""Reuse frozen full-board flows with an explicit, audited RQ-only delta."""
import argparse,collections,json,re,runpy,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from rq_cached_head_board import audit_delta,BASE,REL
from windows_process_tree import stop_launched_tree

def save(p,obj):p.write_text(json.dumps(obj,indent=2),encoding='utf8')

def proofs(board,folder):
    result=json.loads((folder/'result.json').read_text());inputs=json.loads((folder/'inputs.json').read_text())
    assert result['status']=='RQ_CACHED_HEAD_QUEUE_NUMERIC_LOCKSTEP_PASS'
    assert result['inputs_sha256']==sha(folder/'inputs.json')
    assert inputs['board_inputs_sha256']==sha(board/'inputs.json') and inputs['new_rq_sha256']==sha(board/REL)
    for row in inputs['snapshots']:assert sha(row['path'])==row['sha256']
    assert len(result['results'])==4 and result['public_cycles_changed'] is False
    for row in result['results']:
        log=folder/row['kind']/f"simulation-{row['shared_feedback']}-{row['seed']}.log"
        assert sha(log)==row['log_sha256']
    return dict(path=str(folder),result_sha256=sha(folder/'result.json'))

def map_board(board,proof):
    audit_delta(board);evidence=proofs(board,proof)
    assert not (board/'outflow').exists()
    cmd=board/'run.cmd'
    cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat ti60f225_oob.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
    start=time.monotonic()
    try:
        with (board/'map.log').open('x') as log:
            proc=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=board,stdout=log,stderr=subprocess.STDOUT)
            save(board/'progress.json',dict(stage='map',pid=proc.pid,budget_s=960))
            try:code=proc.wait(timeout=960)
            except subprocess.TimeoutExpired:
                save(board/'timeout_cleanup.json',stop_launched_tree(proc));raise
        assert code==0 and re.search(r'map\s*:\s*PASS',(board/'map.log').read_text())
        net=board/'outflow/ti60f225_oob.map.v';rpt=board/'outflow/ti60f225_oob.map.rpt';warn=board/'outflow/ti60f225_oob.warn.log'
        counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
        dumped=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
        for key in ('RAM10','DPRAM10','DSP24','DSP48'):assert counts.get(key,0)==dumped.get(key,0)
        warnings=warn.read_text(errors='replace');previous=(BASE/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
        pattern=r'^\[EFX-0201 WARNING\] (.*?)\. \('
        added=sorted(set(re.findall(pattern,warnings,re.M))-set(re.findall(pattern,previous,re.M)))
        assert not added and 'EFX-0654' not in warnings,(added,'EFX-0654' in warnings)
        audit_delta(board)
        result=dict(status='CONNECTED_BOARD_MAP_MEASURED_NOT_RELEASE',resources=counts,dumped_net_primitives=dumped,
            seconds=time.monotonic()-start,new_undriven_signatures=added,comb_loop_warning=False,
            inputs_sha256=sha(board/'inputs.json'),map_sha256=sha(net),report_sha256=sha(rpt),warning_sha256=sha(warn),
            short_proof=evidence,resource_fit_proven=False,physical_timing_pass=False,board_download_allowed=False)
        save(board/'result.json',result);print(json.dumps(result),flush=True)
    except Exception as exc:
        save(board/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise

def bind():
    # Keep the parent transform module unchanged on disk. Its already-imported
    # original audit remains the parent audit used by rq_cached_head_board.
    import local_control_decode
    local_control_decode.audit_delta=audit_delta

def validate(mode,args):
    import validate_control_ddr as vd
    import control_ddr_combined as combined
    combined.audit_delta=audit_delta;vd.audit_delta=audit_delta
    original=vd.driver_run
    def adapted(source,code,root,argv):
        if mode=='serial':
            anchor="  (dst/'inputs.json').write_text(json.dumps(inp,indent=2))"
            assert code.count(anchor)==1
            inject='''  original=dst/'uninlined';original.mkdir()
  for name in inp['files']:shutil.copy2(dst/name,original/name)
  (original/'inputs.json').write_text(json.dumps(inp,indent=2))
  from inline_official_state_luts import inline
  old_files=dict(inp['files'])
  content,proof=inline((dst/'mapped_control.v').read_text(),dst/'efx_lut4.v',lambda n:n.startswith('\\\\cpu/ctrl_decode/'))
  (dst/'mapped_control.v').write_text(content)
  inp['files']['mapped_control.v']=sha(dst/'mapped_control.v')
  inp.update(source_replay=str(original),source_inputs_sha256=sha(original/'inputs.json'),simulation_only_inline=dict(proof,original_files=old_files,transform_sha256=sha(H/'inline_official_state_luts.py')))
'''
            code=code.replace(anchor,inject+anchor)
            code=code.replace("'simulation.log',180)","'simulation.log',240)")
        return original(source,code,root,argv)
    vd.driver_run=adapted
    # Same Starter diagnostic warning handling as the existing local adapter.
    popen=subprocess.Popen
    def launch(cmd,*pos,**kw):
        if Path(str(cmd[0])).name.lower()=='vsim.exe':cmd=[cmd[0],'-warning','14408,16154','-voptargs=-warning 14408',*cmd[1:]]
        return popen(cmd,*pos,**kw)
    subprocess.Popen=launch
    sys.argv=[str(HERE/'validate_control_ddr.py'),*args]
    if mode=='observe':
        sys.argv.pop(1)
        from validate_control_ddr_observation import main
        main()
    else:vd.main()

def main():
    mode=sys.argv.pop(1)
    if mode=='map':
        p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args()
        map_board(a.board.resolve(),a.proof.resolve());return
    bind()
    if mode=='audit':
        from audit_local_control_decode_map import audit
        board=Path(sys.argv[1]).resolve();save(board/'local_decode_map_audit.json',audit(board));print('LOCAL_CONTROL_MAP_AUDIT_PASS');return
    if mode in ('startup','observe','serial','rtl'):
        validate(mode,[mode,*sys.argv[1:]]);return
    if mode=='place':
        from probe_local_decode_place import main as place
        place();return
    if mode=='route':
        source=HERE/'route_local_decode_board.py';code=source.read_text()
        # Preserve the failure and exact stop state if the independent host
        # deadline fires; do not leave a misleading running stage.
        before="            except subprocess.TimeoutExpired:save(root/(name+'-timeout.json'),stop_launched_tree(proc));raise"
        after="            except subprocess.TimeoutExpired:\n                save(root/(name+'-timeout.json'),stop_launched_tree(proc))\n                row.update(status='timed_out_process_tree_stopped',seconds=time.monotonic()-start)\n                save(root/'stages.json',stages);raise"
        assert code.count(before)==1;code=code.replace(before,after)
        root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();driver=root.parent/(root.name+'-driver.py')
        assert not driver.exists();driver.write_text(code,encoding='utf8')
        save(driver.with_suffix('.json'),dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),wrapper_sha256=sha(__file__)))
        sys.argv[0]=str(driver);runpy.run_path(str(driver),run_name='__main__');return
    raise ValueError(mode)
if __name__=='__main__':main()
