"""Resume camera short gates after a verified, immutable same-map audit."""
import json, runpy, sys
from run_units import ROOT, HERE, sha
from control_command_repair import once
from camera_checked_v2 import BOARD


def main():
    prior = ROOT / 'tmp/frame-execution/camera-checked-evidence2/map-audit'
    root = ROOT / 'tmp/frame-execution/camera-checked-evidence4'
    driver = root.parent / (root.name + '-camera-short-wrapper.py')
    assert not root.exists() and not driver.exists()
    receipt = json.loads((prior / 'result.json').read_text())
    assert receipt['status'] == 'FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS'
    assert receipt['checks'] == 17504
    assert receipt['map_sha256'] == sha(BOARD / 'outflow/ti60f225_oob.map.v')
    assert receipt['board_inputs_sha256'] == sha(BOARD / 'inputs.json')
    for filename, digest in receipt['evidence'].items():
        assert sha(filename) == digest, filename
    held = json.loads((prior / 'held_data_structure.json').read_text())
    assert held['status'] == 'CAMERA_MAPPED_SYNC_AND_HELD_STRUCTURE_PASS'
    assert held['map_sha256'] == receipt['map_sha256']
    source = ROOT / 'tmp/frame-execution/camera-checked-evidence3-camera-short-wrapper.py'
    code = source.read_text()
    replacement = """  prior=Path(PRIOR_PATH)
  audit=root/'map-audit';shutil.copytree(prior,audit)
  reuse=dict(status='REUSED_VERIFIED_SAME_MAP_AUDIT',source=str(prior),
   map_sha256=mapped,evidence={str(f):sha(f) for f in prior.iterdir() if f.is_file()})
  save(audit/'reuse.json',reuse)
  stages.append(dict(stage='map-audit',status='finished',exit_code=0,reused=True,source=str(prior)))
  save(root/'stages.json',stages)""".replace('PRIOR_PATH', repr(str(prior)))
    code = once(code, "  run('map-audit','camera_checked_flow.py',['audit','--board',board,'--root',root/'map-audit'],100)", replacement)
    compile(code, str(driver), 'exec')
    driver.write_text(code, encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(
        source=str(source), source_sha256=sha(source), driver_sha256=sha(driver),
        resumer_sha256=sha(__file__), reused_result_sha256=sha(prior/'result.json'),
        hardware_access=False), indent=2))
    sys.argv = [str(driver), '--board', str(BOARD), '--root', str(root)]
    runpy.run_path(str(driver), run_name='__main__')


if __name__ == '__main__':
    main()
