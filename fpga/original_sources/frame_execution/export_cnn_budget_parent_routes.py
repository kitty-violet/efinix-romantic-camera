"""Probe documented route loading/template export on a private passing parent.

Use load_route alone: the prior project probe already showed it cannot be
combined with --route. This is not an incremental routing or release claim.
"""
import json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree import stop_launched_tree

BASE=ROOT/'tmp/frame-execution/camera-factor-compact-bound3'
OUT=ROOT/'tmp/frame-execution/cnn-budget-parent-route-export1'

def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')

def main():
    assert not OUT.exists()
    selected=ROOT/'tmp/frame-execution/camera-factor-compact-route3/selected.json'
    prior=json.loads(selected.read_text());assert prior['current_timing_pass'] and prior['wns_ns']==.099
    names=['outflow/ti60f225_oob'+s for s in ('.map.v','.place','.out.placeloc','.vdb','.troutingtraces')]
    names+=['ti60f225_oob.sdc','ti60f225_oob.xml','efx_pnr_settings.ini','fixed_dsp.placeloc']
    frozen={str(BASE/n):sha(BASE/n) for n in names}
    OUT.mkdir()
    for d in ('outflow','work_pnr'):shutil.copytree(BASE/d,OUT/d)
    for f in BASE.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,OUT/f.name)
    ini=OUT/'efx_pnr_settings.ini';assert 'generate_rcf_template' not in ini.read_text()
    ini.write_text(ini.read_text()+'generate_rcf_template = on\n')
    args=[str(EF/'bin/efx_pnr.exe'),'--circuit','ti60f225_oob','--family','Titanium','--device','Ti60F225',
        '--operating_conditions','I3','--vdb_file','outflow/ti60f225_oob.vdb','--use_vdb_file','on',
        '--prj',str(OUT/'ti60f225_oob.xml'),'--output_dir','outflow','--work_dir','work_pnr',
        '--place_file','outflow/ti60f225_oob.place','--route_file','outflow/ti60f225_oob.route',
        '--settings_file','efx_pnr_settings.ini','--sync_file','outflow/ti60f225_oob.interface.csv',
        '--generate_prevpr_netlist','off','--sdc_file=ti60f225_oob.sdc','--load_route']
    cmd=OUT/'export.cmd';cmd.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\n'+subprocess.list2cmdline(args)+'\nexit /b %errorlevel%\n')
    save(OUT/'inputs.json',dict(original=frozen,selected=str(selected),selected_sha256=sha(selected),
        script_sha256=sha(__file__),command=args,command_sha256=sha(cmd),settings_sha256=sha(ini),
        docs={str(EF/'doc/topics'/n):sha(EF/'doc/topics'/n) for n in ('cli-pnr-opts.html','rcf-flow.html','rcf-enable-template.html','rcf-file.html','rcf-enable.html')},
        hardware_access=False,host_budget_s=180))
    started=time.monotonic()
    try:
        with (OUT/'export.log').open('x') as log:
            proc=subprocess.Popen(['cmd.exe','/d','/c',str(cmd)],cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
            save(OUT/'stage.json',dict(status='running',pid=proc.pid,budget_s=180))
            try:code=proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                save(OUT/'timeout.json',stop_launched_tree(proc));raise
        save(OUT/'stage.json',dict(status='finished',exit_code=code,seconds=time.monotonic()-started))
        body=(OUT/'export.log').read_text(errors='replace')
        assert code==0 and not re.search(r'^\s*(?:ERROR|FATAL)\s*:',body,re.M)
        for f,h in frozen.items():assert sha(f)==h,f
        for n in names:
            if n!='efx_pnr_settings.ini':assert sha(OUT/n)==frozen[str(BASE/n)],n
        exported={}
        for suffix in ('.rcf.template','.placer_keepout'):
            p=OUT/'outflow'/('ti60f225_oob'+suffix);assert p.exists() and p.stat().st_size>0,p
            exported[str(p)]=sha(p)
        save(OUT/'result.json',dict(status='PASSING_PARENT_ROUTE_TEMPLATE_EXPORTED_NO_GEOMETRY_CHANGE',
            exported=exported,original_unchanged=True,geometry_unchanged=True,inputs_sha256=sha(OUT/'inputs.json'),
            seconds=time.monotonic()-started,hardware_access=False,board_release=False))
        print('PASSING_PARENT_ROUTE_TEMPLATE_EXPORTED_NO_GEOMETRY_CHANGE',flush=True)
    except Exception as e:
        save(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),hardware_access=False));raise

if __name__=='__main__':main()
