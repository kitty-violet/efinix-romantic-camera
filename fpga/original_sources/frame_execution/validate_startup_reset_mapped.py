"""Check startup initialization in the actual Efinity mapped reset circuit."""
import argparse,json,os,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
MS=Path('E:/quatra/questa_fse/win64')


def review_warnings(text, log):
    # Efinity omits physically unused LUT pins and the last carry output.
    # Accept only those exact omissions after independently proving the mask
    # is insensitive to each missing LUT input. Never blanket-suppress warnings.
    harmless={}
    for m in re.finditer(r'EFX_LUT4 (\S+)\s*\((.*?)\) /\*',text,re.S):
        name,ports=m.groups()
        mask=int(re.search(r'defparam '+re.escape(name)+r"\.LUTMASK = 16'h([0-9a-fA-F]+)",text)[1],16)
        missing={f'I{k}' for k in range(4) if f'.I{k}(' not in ports}
        for pin in missing:
            k=int(pin[1]);assert all(((mask>>v)&1)==((mask>>(v^(1<<k)))&1) for v in range(16))
        if missing:harmless[name]=missing
    for m in re.finditer(r'EFX_ADD (\\\S+\s*)\((.*?)\) /\*',text,re.S):
        name,ports=m.groups()
        if '.CO(' not in ports:harmless[name.rstrip()]={'CO'}
    warnings=re.findall(r'^# \*\* Warning: (.+)$',log,re.M)
    active=None
    for w in warnings:
        assert 'board_startup_reset.map.v(' in w,w
        if '(vopt-2685)' in w:
            active=re.search(r"connections for '([^']+)'",w)[1].rstrip()
            assert active in harmless,w
        else:
            assert '(vopt-2718)' in w and active is not None,w
            assert re.search(r"port '([^']+)'",w)[1] in harmless[active],w
    counts=re.findall(r'Errors: (\d+), Warnings: (\d+)',log)
    assert counts and tuple(map(int,counts[-1]))==(0,len(warnings))
    assert not re.search(r'\*\* (Error|Fatal)',log)
    return dict(count=len(warnings),reason='Only proven mask-independent LUT inputs and unused carry outputs',
                exact_cells={k:sorted(v) for k,v in harmless.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--review-existing',action='store_true');a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
    if a.review_existing:
        assert (root/'failure.json').exists() and not (root/'mapped_review.json').exists()
        meta=json.loads((root/'inputs.json').read_text())
        for key in ('source','testbench'):
            assert sha(meta[key])==sha(root/Path(meta[key]).name)==meta[key+'_sha256']
        net=root/'outflow/board_startup_reset.map.v';log=(root/'simulation.log').read_text()
        assert 'PASS BOARD_STARTUP_RESET' in log
        review=review_warnings(net.read_text(),log)
        result=dict(status='EFINITY_MAPPED_STARTUP_RESET_FOURSTATE_PASS',
            source_sha256=meta['source_sha256'],map_sha256=sha(net),
            inputs_sha256=sha(root/'inputs.json'),simulation_sha256=sha(root/'simulation.log'),
            warnings_review=review,reviewer_sha256=sha(__file__),
            prior_harness_failure_preserved=True,no_simulation_rerun=True,board_release=False)
        (root/'mapped_review.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    assert not root.exists();root.mkdir()
    src=HERE/'variants/board_deterministic_reset/board_startup_reset.sv'
    tb=src.with_name('tb_board_startup_reset.sv');shutil.copy2(src,root/src.name);shutil.copy2(tb,root/tb.name)
    xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text()
    (root/'board_startup_reset.xml').write_text(xml.replace('frame_execution_meter','board_startup_reset'))
    (root/'board_startup_reset.sdc').write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat board_startup_reset.xml --flow map --timeout 60\nexit /b %errorlevel%\n')
    (root/'inputs.json').write_text(json.dumps(dict(source=str(src),source_sha256=sha(src),
          testbench=str(tb),testbench_sha256=sha(tb),runner_sha256=sha(__file__),hardware_access=False),indent=2))
    try:
        body,seconds=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=85)
        assert re.search(r'map\s*:\s*PASS',body)
        net=root/'outflow/board_startup_reset.map.v';text=net.read_text()
        types=set(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+',text,re.M))
        assert types<={'FF','LUT4','ADD'} and 'FF' in types,types
        assert 'EFX_SRL8' not in text
        models=[EF/'sim_models/verilog'/('efx_'+t.lower()+'.v') for t in sorted(types)]
        os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
        run([MS/'vlib.exe','work'],root,'vlib.log',timeout=30)
        log,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*models,net,root/tb.name],root,'compile.log',timeout=40)
        assert 'Errors: 0, Warnings: 0' in log
        log,simseconds=run([MS/'vsim.exe','-c','work.tb_board_startup_reset',
            '-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=45)
        assert 'PASS BOARD_STARTUP_RESET' in log
        warning_review=review_warnings(text,log)
        assert sha(src)==sha(root/src.name) and sha(tb)==sha(root/tb.name)
        result=dict(status='EFINITY_MAPPED_STARTUP_RESET_FOURSTATE_PASS',map_seconds=seconds,
            sim_seconds=simseconds,source_sha256=sha(src),map_sha256=sha(net),
            models={str(f):sha(f) for f in models},inputs_sha256=sha(root/'inputs.json'),warnings_review=warning_review,
            board_release=False,physical_timing_pass=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
