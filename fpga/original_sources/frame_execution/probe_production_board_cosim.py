"""Bounded board startup with the narrow production-CNN partition.

Retains the frozen CSI/DDR models, CPU, video path and measurement monitors.
Always stops at an explicit short duration; cannot declare frame completion.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from probe_board_dpi_cost import BIN, SIM, run
from run_units import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--run-only',action='store_true')
    p.add_argument('--duration-us',type=int,default=100)
    a=p.parse_args();root=a.root.resolve();base=a.base.resolve();native=a.native.resolve()
    assert root.is_relative_to(ROOT/'tmp') and 1<=a.duration_us<=100
    env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    if not a.run_only:
        assert not root.exists();root.mkdir();bm=json.loads((base/'inputs.json').read_text())
        nm=json.loads((native/'inputs.json').read_text())
        assert nm['board_inputs_sha256']==sha(base/'inputs.json')
        assert sha(native/'production.dll')==nm['dll_sha256']
        records=[]
        for row in bm['inputs']:
            old=Path(row['snapshot']);assert sha(old)==row['sha256']
            assert old.is_relative_to(base),old
            target=root/old.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,target)
            records.append(dict(snapshot=str(target),sha256=sha(target),original_sha256=sha(old)))
        # The prepared testbench is covered separately by tb_sha256 in the
        # existing board manifest. Rewrite only its artifact root, preserving
        # every original check and all causal model paths.
        oldtb=base/'tb_board_startup.sv';assert sha(oldtb)==bm['tb_sha256']
        tb=oldtb.read_text().replace(base.as_posix(),root.as_posix()).replace(str(base),str(root))
        (root/'tb_board_startup.sv').write_text(tb)
        for f in base.glob('*.hex'):
            if not (root/f.name).exists():shutil.copy2(f,root/f.name)
        files=[root/Path(f).relative_to(base) for f in bm['source_files'] if Path(f).name!='cnn_production_frame.sv']
        wrapper=root/'cnn_production_frame_dpi.sv';shutil.copy2(native/'cnn_production_frame_dpi.sv',wrapper);files.append(wrapper)
        inc=[root/Path(f).relative_to(base) for f in bm['include_dirs']]
        for f in files:assert f.is_file(),f
        manifest=dict(base=str(base),base_inputs_sha256=sha(base/'inputs.json'),native=str(native),
            native_inputs_sha256=sha(native/'inputs.json'),native_dll_sha256=sha(native/'production.dll'),
            inputs=[dict(snapshot=str(f),sha256=sha(f)) for f in dict.fromkeys([Path(r['snapshot']) for r in records]+files)],
            tb_sha256=sha(root/'tb_board_startup.sv'),retained_measurement_probes=True,
            simulated_us=a.duration_us,timeout_seconds=240,full_frame=False)
        (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
        run([SIM/'vlib.exe','work'],root,'vlib',env,20)
        run([SIM/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16',
             *['+incdir+'+str(x) for x in inc],*files],root,'compile',env,90)
        print('BOARD_PRODUCTION_PARTITION_COMPILE_PASS',flush=True)
    if a.prepare_only:return
    manifest=json.loads((root/'inputs.json').read_text())
    for row in manifest['inputs']:assert sha(row['snapshot'])==row['sha256']
    assert sha(native/'inputs.json')==manifest['native_inputs_sha256']
    assert sha(native/'production.dll')==manifest['native_dll_sha256']
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\n'+f'run {a.duration_us} us\nputs "BOUNDED_BOARD_PARTITION_STOP_US {a.duration_us}"\nquit -code 0\n')
    seconds=run([SIM/'vsim.exe','-c','-warning','14408,16154','-voptargs=-warning 14408 -O5',
                 '-sv_lib',native/'production','work.tb_board_startup','-do','run.do'],root,'simulation',env,240)
    log=(root/'simulation.log').read_text(errors='replace')
    assert f'BOUNDED_BOARD_PARTITION_STOP_US {a.duration_us}' in log
    assert not re.search(r'\*\* (Fatal|Error)| ERROR: ',log)
    result=dict(status='BOARD_PRODUCTION_PARTITION_BOUNDED_STARTUP_PASS',wall_seconds=seconds,simulated_us=a.duration_us,
        full_frame=False,active_cnn_speed=False,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
        progress=re.findall(r'BOARD_(?:PROGRESS|BOOT_PHASE).*',log))
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
