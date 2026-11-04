"""Freeze the complete CSI-to-TMDS bench with the validated narrow partition.

Copies the previous board's exact hardware and checks, preserves its map
identity, and requires same-DLL boundary/active/unknown/real-DDR proofs. Does
not start a simulation or change the three-distinct-screen-frame criterion.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
from probe_board_dpi_cost import BIN,SIM,run
from production_board_binding import validate
from run_units import ROOT,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    for name in ('boundary','active','unknown','vendor'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();base=a.base.resolve();native=a.native.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    bm=json.loads((base/'inputs.json').read_text());nm=json.loads((native/'inputs.json').read_text())
    assert nm['board_inputs_sha256']==sha(base/'inputs.json')
    assert bm['required_distinct_screen_frames']==3 and bm['ddr516'] and not bm['raw_diagnostic']
    assert bm['native'] is None
    proofs={}
    for name in ('boundary','active','unknown','vendor'):
        f=getattr(a,name).resolve()/'result.json';assert f.is_file()
        proofs[name]=dict(path=str(f),sha256=sha(f))
    root.mkdir();meta=copy.deepcopy(bm)
    for row in meta['inputs']:
        source=Path(row['snapshot']);assert source.is_relative_to(base) and sha(source)==row['sha256']
        target=root/source.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        row['snapshot']=str(target)
    # Preserve relative-read ROMs and existing host-only equivalence proofs.
    for f in list(base.glob('*.hex'))+list(base.glob('*.bin'))+list(base.glob('*proof.json')):
        if not (root/f.name).exists():shutil.copy2(f,root/f.name)
    tb=base/'tb_board_startup.sv';assert sha(tb)==bm['tb_sha256']
    content=tb.read_text().replace(base.as_posix(),root.as_posix()).replace(str(base),str(root))
    (root/tb.name).write_text(content)
    meta['tb_sha256']=sha(root/tb.name)
    for row in meta['inputs']:
        if Path(row['snapshot']).name==tb.name:row['sha256']=meta['tb_sha256']
    wrapper=root/'cnn_production_frame_dpi.sv';shutil.copy2(native/wrapper.name,wrapper)
    meta['inputs'].append(dict(snapshot=str(wrapper),sha256=sha(wrapper),source=str(native/wrapper.name)))
    meta['source_files']=[str(root/Path(f).relative_to(base)) for f in bm['source_files'] if Path(f).name!='cnn_production_frame.sv']+[str(wrapper)]
    meta['include_dirs']=[str(root/Path(f).relative_to(base)) for f in bm['include_dirs']]
    meta.update(native=str(native),native_kind='production_cnn',native_library=str(native/'production'),
                native_inputs_sha256=sha(native/'inputs.json'),native_dll_sha256=sha(native/'production.dll'),
                partition_proofs=proofs,simulation_parent=str(base),simulation_parent_inputs_sha256=sha(base/'inputs.json'))
    proof=validate(root,meta)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    proof['inputs_sha256']=sha(root/'inputs.json')
    (root/'production_partition_binding.json').write_text(json.dumps(proof,indent=2))
    env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16',
         *['+incdir+'+x for x in meta['include_dirs']],*meta['source_files']],root,'compile',env,120)
    result=dict(status='BOARD_FRAME_COMPILE_PASS',inputs_sha256=sha(root/'inputs.json'),
                compile_log_sha256=sha(root/'compile.log'),full_frame_run=False,hardware_changed=False)
    (root/'prepared.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
