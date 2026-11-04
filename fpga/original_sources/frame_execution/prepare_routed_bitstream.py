"""Generate an offline bitstream from a verified frozen route; never program.

The output is quarantined (board_release=false) until remaining release checks
are complete. The Efinity 'pgm' flow creates files; 'program' is never invoked.
"""
import argparse
import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from audit_canonical_multiseed import audit
from run_units import ROOT, EF, sha
from windows_process_tree_safe import stop_launched_tree


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--canonical', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    check = audit(args.canonical)
    assert check['current_timing_pass']
    base = Path(check['physical'])
    root = args.root.resolve()
    assert root.is_relative_to(ROOT / 'tmp') and not root.exists()
    shutil.copytree(base, root)
    # A copied result describes the source route, not this generation operation.
    (root / 'source_physical_result.json').write_bytes((base / 'result.json').read_bytes())
    (root / 'result.json').unlink()
    shutil.copy2(args.canonical / 'ti60f225_oob.sdc', root / 'ti60f225_oob.sdc')
    manifest = dict(status='OFFLINE_GENERATION_STARTED_NOT_RELEASED', base=str(base),
                    canonical=str(args.canonical.resolve()), acceptance=check,
                    source_result_sha256=sha(base / 'result.json'),
                    input_geometry={name: sha(base / 'outflow' / name) for name in
                                    ('ti60f225_oob.map.v', 'ti60f225_oob.place', 'ti60f225_oob.troutingtraces', 'ti60f225_oob.vdb')},
                    flow='pgm', board_release=False, hardware_access=False)
    (root / 'generation_inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (root / 'NOT_RELEASED.txt').write_text('OFFLINE CANDIDATE ONLY. Hardware programming is not released.\n')
    script = root / 'generate_offline.cmd'
    script.write_text('@echo off\ncall "' + str(EF / 'bin/setup.bat') +
                      '"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow pgm --timeout 240\nexit /b %errorlevel%\n')
    start = time.monotonic()
    try:
        with (root / 'generation.log').open('w') as log:
            proc = subprocess.Popen(['cmd.exe', '/d', '/c', str(script)], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=270)
            except subprocess.TimeoutExpired:
                (root / 'generation_timeout_cleanup.json').write_text(json.dumps(stop_launched_tree(proc), indent=2))
                raise
        assert code == 0, ('Offline generation failed', code)
        for name, digest in manifest['input_geometry'].items():
            assert sha(base / 'outflow' / name) == sha(root / 'outflow' / name) == digest
        artifacts = {str(f.relative_to(root)): dict(bytes=f.stat().st_size, sha256=sha(f))
                     for f in (root / 'outflow').iterdir() if f.suffix in ('.bit', '.hex') and f.stat().st_size > 0}
        assert artifacts, 'No bitstream output'
        result = dict(status='OFFLINE_BITSTREAM_GENERATED_NOT_RELEASED', seconds=time.monotonic()-start,
                      inputs_sha256=sha(root / 'generation_inputs.json'), artifacts=artifacts,
                      source_and_route_unchanged=True, board_release=False, hardware_access=False)
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    except Exception as exc:
        (root / 'generation_failure.json').write_text(json.dumps(dict(error=repr(exc), traceback=traceback.format_exc()), indent=2))
        raise


if __name__ == '__main__':
    main()
