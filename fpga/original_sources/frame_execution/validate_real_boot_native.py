"""Accelerated boot validation using Efinity's actual mapped Sapphire CPU.

This is a two-state mapped test, separately labelled from the four-state vendor
CPU test. No encrypted helper is replaced or decrypted. Every primitive model
comes from the installed Efinity distribution. Not a full-board timing run.
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from run_units import ROOT, HERE, EF, sha
sys.path.insert(0, str(HERE.parent / 'pw_array_feed_abc'))
from run_native import BIN, VROOT, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--cpu-map', required=True, type=Path)
    a = p.parse_args()
    root, ref, cpu = a.root.resolve(), a.reference.resolve(), a.cpu_map.resolve()
    assert root.is_relative_to(ROOT / 'tmp') and not root.exists()
    root.mkdir(parents=True)
    try:
        meta = json.loads((cpu / 'result.json').read_text())
        assert meta['map_complete'] and meta['module'] == 'EfxSapphireSoc'
        image = json.loads((ref / 'firmware/boot.json').read_text())
        assert image['sha256'] == meta['boot_sha256'] == sha(ref / 'firmware/boot.bin')
        net = cpu / 'outflow/board_ip.map.v'
        assert sha(net) == meta['net_sha256']
        assert not re.search(r'pragma\s+protect|`protect', net.read_text())
        records, sources = [], []
        def copy(f, compile=True):
            dst = root / f.name
            assert not dst.exists(), dst
            shutil.copy2(f, dst)
            records.append(dict(source=str(f), snapshot=str(dst), sha256=sha(f)))
            if compile: sources.append(dst)
        kinds = set(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+', net.read_text(), re.M))
        for kind in sorted(kinds | {'RAM10'}): copy(EF / 'sim_models/verilog' / f'efx_{kind.lower()}.v')
        copy(net)
        for name in ('resident_weight_bank64.sv', 'coefficient_bank_store_fast.sv', 'cnn_apb_control.sv', 'sapphire_axi32_to_axi128.sv'):
            copy(ref / name)
        for name in ('jtag_memory_tasks.svh', 'boot_vectors.vh'): copy(ref / name, False)
        for f in ref.glob('EfxSapphireSoc*.bin'): copy(f, False)
        # The mapper removed only this alias. Verify it against vendor source.
        raw = (ref / 'EfxSapphireSoc.v').read_text()
        assert 'assign system_cores_0_dBus_cmd_ready = system_cores_0_dBus_cmd_rValidN;' in raw
        tb = (ref / 'tb_real_boot.sv').read_text().replace('dut.system_cores_0_dBus_cmd_ready', 'dut.system_cores_0_dBus_cmd_rValidN')
        (root / 'tb_real_boot.sv').write_text(tb)
        sources.append(root / 'tb_real_boot.sv')
        env = os.environ.copy()
        env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'], TEMP=str(root), TMP=str(root), TMPDIR=str(root),
                   MSYS2_ARG_CONV_EXCL='*', VERILATOR_ROOT=str(VROOT), MAKE=str(BIN/'mingw32-make.exe'))
        inputs = dict(inputs=records, reference_inputs_sha256=sha(ref/'inputs.json'), map_result_sha256=sha(cpu/'result.json'),
                      tb_sha256=sha(root/'tb_real_boot.sv'), runner_sha256=sha(__file__), primitive_kinds=sorted(kinds))
        (root/'inputs.json').write_text(json.dumps(inputs, indent=2))
        obj=root/'obj'
        cmd=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module','tb_real_boot',
             '--timescale-override','1ns/1ps','--output-split','12000','--output-split-cfuncs','500','-Wno-fatal',
             '--Mdir',obj.as_posix(),*[f.as_posix() for f in sources]]
        _,compile_seconds=run(cmd,root,root/'compile.log',env)
        opts='-O1 -munroll-only-small-loops -mvzeroupper'
        make=[BIN/'mingw32-make.exe','-f','Vtb_real_boot.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',
              f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}',
              'PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}',
              'SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',
              f'OPT_FAST={opts}',f'OPT_GLOBAL={opts}','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
        _,build_seconds=run(make,obj,root/'build.log',env)
        binary=obj/'Vtb_real_boot.exe'
        if not binary.exists():binary=obj/'Vtb_real_boot'
        results=[]
        for smoke in (1,0):
            log=root/('smoke.log' if smoke else 'simulation.log')
            body,seconds=run([binary,f'+SMOKE={smoke}'],root,log,env,timeout=10800)
            marker='PASS REAL_BOOT_SMOKE' if smoke else 'PASS REAL_BOOT_COMPLETE'
            assert marker in body and 'Error' not in body and 'Fatal' not in body
            results.append(dict(smoke=smoke,seconds=seconds,lines=re.findall(r'^(PASS .*)$',body,re.M),log_sha256=sha(log)))
            print(json.dumps(results[-1]),flush=True)
        for row in records:assert sha(row['snapshot'])==row['sha256']
        warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',(root/'compile.log').read_text(),re.M))
        result=dict(status='REAL_BOOT_MAPPED_TWO_STATE_PASS_WARNINGS_REVIEW_OPEN',results=results,
                    compile_seconds=compile_seconds,build_seconds=build_seconds,warnings=dict(warnings),
                    inputs_sha256=sha(root/'inputs.json'),binary_sha256=sha(binary),full_board=False,ddr_phy=False,board_release=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
