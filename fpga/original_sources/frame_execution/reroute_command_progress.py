"""Documented 100-iteration route-only retry of a preserved legal placement."""
import argparse,json,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree import stop_launched_tree
from control_command_repair import audit_delta
from audit_canonical_multiseed import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    source=a.source.resolve();root=a.root.resolve();old=source/'pnr-seed1';physical=root/'pnr-seed1'
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    prior=json.loads((old/'result.json').read_text());meta=json.loads((old/'physical_inputs.json').read_text());board=Path(meta['base'])
    assert any(s['flow']=='pnr' and s['exit_code']==1 for s in prior['stages'])
    failed=(old/'outflow/ti60f225_oob.route.out').read_text()
    assert 'Routing FAILED; could not resolve congestion.' in failed
    iterations=re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)',failed);assert iterations[-1]==('70','2'),iterations[-1]
    assert meta['setup_margin'] is None;audit_delta(board)
    for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
    root.mkdir();(root/'runner.py').write_bytes(Path(__file__).read_bytes());shutil.copytree(old,physical)
    for n in ('result.json','stages.json'):
        (physical/n).rename(physical/('source_'+n))
    for row in meta['inputs']:
        row['snapshot']=str(physical/Path(row['snapshot']).relative_to(old));assert sha(row['snapshot'])==row['sha256']
    (physical/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
    shutil.copytree(source/'startup',root/'startup');shutil.copy2(source/'command_sync.json',root/'command_sync.json')
    serial=source.parent/(source.name+'-serial');assert (serial/'progress_counter_checks.json').exists()
    (root/'inputs.json').write_text(json.dumps(dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),map_sha256=meta['base_map_sha256'],runner_sha256=sha(__file__),source_batch=str(source),source_result_sha256=sha(old/'result.json'),hardware_access=False),indent=2))
    frozen={str(old/n):sha(old/n) for n in ('outflow/ti60f225_oob.place','outflow/ti60f225_oob.out.placeloc','outflow/ti60f225_oob.vdb','outflow/ti60f225_oob.map.v','ti60f225_oob.sdc','ti60f225_oob.xml','fixed_dsp.placeloc','efx_pnr_settings.ini')}
    args=[str(EF/'bin/efx_pnr.exe'),'--circuit','ti60f225_oob','--family','Titanium','--device','Ti60F225','--operating_conditions','I3','--vdb_file','outflow/ti60f225_oob.vdb','--use_vdb_file','on','--prj',str(physical/'ti60f225_oob.xml'),'--output_dir','outflow','--work_dir','work_pnr','--place_file','outflow/ti60f225_oob.place','--route_file','outflow/ti60f225_oob.route','--settings_file','efx_pnr_settings.ini','--sync_file','outflow/ti60f225_oob.interface.csv','--generate_prevpr_netlist','off','--sdc_file=ti60f225_oob.sdc','--max_router_iterations','100','--route']
    cmd=physical/'route_only.cmd';cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\n'+subprocess.list2cmdline(args)+'\nexit /b %errorlevel%\n')
    (root/'route_continuation.json').write_text(json.dumps(dict(source=str(old),frozen=frozen,command=args,command_sha256=sha(cmd),official_doc=str(EF/'doc/topics/cli-pnr-opts.html'),official_doc_sha256=sha(EF/'doc/topics/cli-pnr-opts.html'),rtl_changed=False,map_changed=False,placement_changed=False,constraints_changed=False,hardware_access=False),indent=2))
    stages=[]
    def run(name,args,cwd,budget):
        start=time.monotonic();log=root/(name+'.log')
        with log.open('x') as f:
            proc=subprocess.Popen(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
            row=dict(name=name,pid=proc.pid,status='running',budget_s=budget);stages.append(row);(root/'stages.json').write_text(json.dumps(stages,indent=2))
            try:code=proc.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                (root/(name+'-timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
        row.update(status='finished',exit_code=code,seconds=time.monotonic()-start);(root/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,code)
        return row
    try:
        row=run('route_only',['cmd.exe','/d','/c',str(cmd)],physical,900)
        for path,h in frozen.items():assert sha(path)==sha(physical/Path(path).relative_to(old))==h,path
        route=(physical/'outflow/ti60f225_oob.route.out').read_text();overlaps=re.findall(r'Iter\s+\d+:\s*overlap=(\d+)',route)
        assert overlaps and int(overlaps[-1])==0 and not re.search(r'^\s*(ERROR|FATAL)\s*:',route,re.M)
        (physical/'result.json').write_text(json.dumps(dict(status='PRESERVED_PLACEMENT_ROUTE_PASS_PENDING_STA',stages=[*[s for s in prior['stages'] if s['flow']=='interface'],dict(flow='pnr',exit_code=0,seconds=row['seconds'],route_only=True)],board_release=False),indent=2))
        import audit_control_state_implementation as impl
        impl.audit_delta=audit_delta
        (root/'implementation.json').write_text(json.dumps(impl.audit(board,physical),indent=2))
        run('canonical',[sys.executable,str(HERE/'canonical_snapshot_sta.py'),'--base',str(physical),'--root',str(root/'canonical-seed1')],ROOT,270)
        selected=audit(root/'canonical-seed1');(root/'selected.json').write_text(json.dumps(selected,indent=2));assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
        run('gray',[sys.executable,str(HERE/'audit_command_progress_gray.py'),'--base',str(physical),'--root',str(root/'gray-seed1')],ROOT,380)
        (root/'result.json').write_text(json.dumps(dict(status='COMMAND_PROGRESS_PHYSICAL_PASS_PENDING_FUNCTIONAL_CDC_AND_RELEASE',selected=selected,serial=str(serial),gray=str(root/'gray-seed1'),route_continuation_sha256=sha(root/'route_continuation.json'),hardware_access=False,board_release=False),indent=2))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise

if __name__=='__main__':main()
