"""Same RAM-snapshot map/C2/seed; free only old arena RAM anchors before place.

Stop after placement. No route, bitstream or hardware access is performed.
"""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once


def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    source=HERE/'place_registered_boundary_board.py';code=source.read_text()
    code=once(code,'from registered_boundary_board import audit_delta','from snapshot_ram_board import audit_delta')
    anchor="source=HERE/'probe_local_decode_place.py';code=source.read_text()"
    injection="""
    code=once(code,'from fixed_cnn_ram_locations import generate as retain_cnn_ram',
              'from partial_cnn_ram_locations import generate as retain_cnn_ram')
    code=once(code,'from fixed_cnn_ram_locations import verify_placed',
              'from partial_cnn_ram_locations import verify_placed')
"""
    code=once(code,anchor,anchor+injection)
    driver=root.parent/(root.name+'-arena-wrapper.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(
        source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),
        wrapper_sha256=sha(__file__),policy_sha256=sha(HERE/'partial_cnn_ram_locations.py'),
        change='Only 84 arena RAM placement anchors released; 29 other RAM and 288 DSP primitive assignments preserved.',
        hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]]
    runpy.run_path(str(driver),run_name='__main__')


if __name__=='__main__':main()
