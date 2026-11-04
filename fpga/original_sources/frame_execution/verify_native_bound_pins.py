"""Query each rebound Gray pin in a private placement-only STA view."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,EF,sha,run
from rebind_native_placement import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();r=audit(base);root.mkdir()
    for folder in ('outflow','work_pnr'):shutil.copytree(base/folder,root/folder)
    for f in base.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'):shutil.copy2(f,root/f.name)
    lines=[]
    for index,name in enumerate(sorted(set(r['pins'].values()))):
        lines.append('puts "NATIVE_PIN_'+str(index)+' [llength [get_pins {'+name+'}]]"')
    lines.append('puts {NATIVE_PIN_QUERY_DONE}; flush stdout')
    (root/'pins.tcl').write_text('\n'.join(lines)+'\n')
    (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow sta_tclsh --tcl_script pins.tcl --timeout 90\nexit /b %errorlevel%\n')
    meta=dict(base=str(base),rebind_sha256=sha(base/'endpoint_rebinding.json'),runner_sha256=sha(__file__),
        files={f.name:sha(f) for f in root.iterdir() if f.is_file()},hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    try:
        body,seconds=run(['cmd.exe','/d','/c','run.cmd'],root,'query.log',timeout=110)
        rows=re.findall(r'NATIVE_PIN_(\d+)\s+(\d+)',body)
        assert len(rows)==140 and all(int(n)==1 for _,n in rows),rows[:3]
        assert 'NATIVE_PIN_QUERY_DONE' in body
        combined=body+'\n'+'\n'.join(f.read_text(errors='replace') for f in (root/'outflow').glob('*.sta.out'))
        assert not re.search(r'No (?:pins|ports|clocks) matched|Unable to run|^\s*(?:ERROR|FATAL)\s*:',combined,re.M)
        result=dict(status='ACTUAL_TOOL_140_GRAY_PINS_NONEMPTY_PASS',seconds=seconds,inputs_sha256=sha(root/'inputs.json'),
            log_sha256=sha(root/'query.log'),pin_count=140,hardware_access=False,routed_timing_pass=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),hardware_access=False),indent=2));raise

if __name__=='__main__':main()
