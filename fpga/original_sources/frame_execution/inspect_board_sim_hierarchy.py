"""Read-only source-library hierarchy report in a separate compiler workspace."""
import argparse, json, os, subprocess, time
from pathlib import Path
from run_units import ROOT, sha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root, base = args.root.resolve(), args.base.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir()
    tools = Path('E:/quatra/questa_fse/win64')
    env = os.environ.copy()
    env['SALT_LICENSE_SERVER'] = 'E:/quatra/LR-195771_License.dat'
    start = time.monotonic()
    commands = [
        [tools/'vmap.exe', '-c'],
        [tools/'vlib.exe', 'work'],
        [tools/'vmap.exe', 'board_ref', base/'work'],
        [tools/'vopt.exe', 'board_ref.tb_board_startup', '-L', 'board_ref',
         '-warning', '14408', '-duselectreport=hierarchy.json', '-stats=all',
         '-o', 'board_profile_opt'],
    ]
    for index, cmd in enumerate(commands):
        with (root/f'command{index}.log').open('wb') as output:
            result = subprocess.run(list(map(str, cmd)), cwd=root, env=env,
                                    stdout=output, stderr=subprocess.STDOUT, timeout=240)
        if result.returncode:
            raise RuntimeError(f'Compiler command {index} failed: {result.returncode}')
    (root/'result.json').write_text(json.dumps(dict(
        status='HIERARCHY_COMPILE_REPORT_ONLY', base=str(base),
        source_inputs_sha256=sha(base/'inputs.json'),
        report_sha256=sha(root/'hierarchy.json'), seconds=time.monotonic()-start,
        simulation_started=False), indent=2))
    print('HIERARCHY_COMPILE_REPORT_ONLY', flush=True)

if __name__ == '__main__':
    main()
