"""Short, isolated four-state regression of the mapped runtime colour stage.

Uses the installed ModelSim and official Efinity primitive models. Does not
build the board, program a device, or imply timing closure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    models = Path(r'E:\efinity-2026.1.132-windows-x64\efinity\2026.1\sim_models\verilog')
    simulator = Path(r'E:\altera\15.0\modelsim_ase\win32aloem')
    env = os.environ.copy()
    env.update(TEMP=str(out), TMP=str(out))
    names = ['efx_add.v', 'efx_ff.v', 'efx_lut4.v', 'efx_ram10.v']
    for name in names:
        shutil.copy2(models / name, out / name)
    shutil.copy2(root / 'outflow/style_colour_runtime.map.v', out)
    shutil.copy2(Path(__file__).with_name('fpga_style_port') / 'tb_colour_runtime.sv', out)

    def hex_file(source, target, limit=None):
        data = (root / source).read_bytes()
        if len(data) % 3:
            raise ValueError('RGB length')
        if limit is not None:
            data = data[:limit * 3]
        (out / target).write_text(''.join(data[i:i+3].hex() + '\n'
                                         for i in range(0, len(data), 3)), encoding='ascii')

    hex_file('table.rgb', 'table1.hex')
    hex_file('table2.rgb', 'table2.hex')
    hex_file('input.rgb', 'probe_input.hex', 128)
    hex_file('gold.rgb', 'probe_gold1.hex', 128)
    hex_file('gold2.rgb', 'probe_gold2.hex', 128)
    (out / 'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\n'
                              'run -all\nquit -code 0\n', encoding='ascii')
    stages = []
    steps = [
        ('library', ['vlib.exe', 'work']),
        # Official primitive files have no timescale of their own.
        ('compile', ['vlog.exe', '-sv', '-timescale', '1ns/1ps', *names,
                     'style_colour_runtime.map.v', 'tb_colour_runtime.sv']),
        ('simulation', ['vsim.exe', '-c', '-wlf', str(out / 'simulation.wlf'),
                        'work.tb_colour_runtime', '-do', 'run.do']),
    ]
    for name, command in steps:
        started = time.perf_counter()
        command[0] = str(simulator / command[0])
        with (out / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=out, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=120)
        stages.append(dict(stage=name, command=command, exit_code=result.returncode,
                           seconds=time.perf_counter() - started))
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {out / (name + ".log")}')
    log = (out / 'simulation.log').read_text(encoding='utf-8', errors='replace')
    marker = 'PASS FOURSTATE_RUNTIME_STYLE checked=98 old_new_bypass_reset=1'
    if marker not in log:
        raise RuntimeError('simulation returned without success marker')
    warning_lines = [line for line in log.splitlines() if '** Warning:' in line]
    # Mapped EFX_ADD intentionally leaves unused carry/sum output ports open.
    unexpected = [line for line in warning_lines
                  if '(vsim-3017)' not in line and '(vsim-3722)' not in line]
    if unexpected:
        raise RuntimeError('unreviewed simulation warnings: ' + repr(unexpected))
    result = dict(status='MAPPED_OFFICIAL_RAM_FOURSTATE_PASS', stages=stages,
                  checked=98, expected_unconnected_output_warnings=len(warning_lines),
                  full_style=False, physical_timing=False,
                  files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in out.iterdir() if p.suffix in ('.v', '.sv', '.hex', '.do')})
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(marker)
    print(out / 'result.json')


if __name__ == '__main__':
    main()
