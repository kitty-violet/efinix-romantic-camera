"""Explicit-path compile/elaboration compatibility gate for a new simulator.

No install, global environment changes, license acquisition or board access.
Elaboration PASS is not a functional simulation, PHY or throughput PASS.
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import traceback
from pathlib import Path
from run_units import ROOT, EF, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--sim-bin',type=Path,required=True)
    p.add_argument('--license-file',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();sim=a.sim_bin.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
    env=os.environ.copy()
    if a.license_file:
        assert a.license_file.is_file()
        # Current Questa Altera reads SALT_LICENSE_SERVER; retain MGLS for older tools.
        env['SALT_LICENSE_SERVER']=str(a.license_file.resolve())
        env['MGLS_LICENSE_FILE']=str(a.license_file.resolve())
    try:
        old=json.loads((ROOT/'tmp/frame-execution/csi-official-bundle1/result.json').read_text())
        for row in old['files']:assert sha(row['path'])==row['sha256']
        primitives=sorted((EF/'sim_models/verilog').glob('*.v'))
        specs=[('common_async','efx_asyncreg',[EF/'ipm/ip/common/source/efx_asyncreg.v'],True),
               ('installed_csi_bundle','efx_csi2_rx_top',[Path(row['path']) for row in old['files']],True),
               ('current_board_csi','csi_rx_controller',
                [ROOT/'tmp/frame-execution/connected-board-sharing1/rtl/csi_rx_controller.sv'],False)]
        jobs=[]
        for name,top,files,macro in specs:
            target=root/name;target.mkdir();records=[]
            for src in primitives+files:
                dst=target/src.name;assert not dst.exists();shutil.copy2(src,dst)
                records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
            args=[str(sim/'vlog.exe'),'-sv','-mfcu','+incdir+.']
            if macro:
                (target/'macro.sv').write_text('`define IP_MODULE_NAME(name) name\n')
                args.append('macro.sv')
            args += [Path(r['snapshot']).name for r in records]
            jobs.append(dict(name=name,top=top,path=str(target),inputs=records,compile=args,
                elaborate=[str(sim/'vsim.exe'),'-c','work.'+top,'-do',
                           'onerror {quit -code 1}; onbreak {quit -code 1}; run 0 ns; quit -code 0']))
        manifest=dict(simulator=str(sim),jobs=jobs,runner_sha256=sha(__file__),prepare_only=a.prepare_only,
                      scope='Original IP compilation and zero-time elaboration only',functional_pass=False)
        (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
        if a.prepare_only:
            print(json.dumps(dict(status='PREPARED_NOT_RUN',jobs=[j['name'] for j in jobs],inputs_sha256=sha(root/'inputs.json'))));return
        for exe in ('vlib.exe','vlog.exe','vsim.exe'):assert (sim/exe).is_file()
        def execute(command,target,log):
            with (target/log).open('wb') as out:
                proc=subprocess.run(command,cwd=target,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=300)
            return proc.returncode
        execute([str(sim/'vsim.exe'),'-version'],root,'version.log')
        # This Starter nodelocked license permits one simulation session at a time.
        # Independent compiles may still overlap.
        simulation_lock=threading.Lock()
        def one(job):
            target=Path(job['path']);codes={}
            for stage,command in (('vlib',[str(sim/'vlib.exe'),'work']),('compile',job['compile']),('elaborate',job['elaborate'])):
                if stage=='elaborate':
                    with simulation_lock:
                        codes[stage]=execute(command,target,stage+'.log')
                else:
                    codes[stage]=execute(command,target,stage+'.log')
                if codes[stage]:break
            return dict(name=job['name'],codes=codes,elaboration_pass=codes.get('elaborate')==0,
                        logs={f.name:sha(f) for f in target.glob('*.log')},functional_pass=False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,jobs))
        for job in jobs:
            for r in job['inputs']:assert sha(r['snapshot'])==r['sha256']
        result=dict(status='COMPATIBILITY_ELAB_PASS' if all(r['elaboration_pass'] for r in results) else 'COMPATIBILITY_INCOMPLETE',
                    results=results,inputs_sha256=sha(root/'inputs.json'),full_board=False,functional_pass=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
