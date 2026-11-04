"""Check the arena release adapter and refusal of negative original STA."""
import runpy
from run_units import ROOT, HERE
from control_command_repair import once


def main():
    source = HERE / 'preflight_registered_release.py'
    code = source.read_text()
    code = once(code, 'import finish_registered_boundary_release as release',
                'import finish_snapshot_arena_release as release')
    code = code.replace('registered-package-preflight1', 'snapshot-arena-package-preflight1')
    code = code.replace('must-not-exist-registered-preflight', 'must-not-exist-snapshot-arena-preflight')
    code = once(code, "'registered_boundary_transform_sha256'", "'base_inputs_sha256'")
    driver = ROOT / 'tmp/board-test/snapshot-arena-package-preflight-runner.py'
    assert not driver.exists()
    compile(code, str(driver), 'exec')
    driver.write_text(code, encoding='utf8')
    runpy.run_path(str(driver), run_name='__main__')


if __name__ == '__main__':
    main()
