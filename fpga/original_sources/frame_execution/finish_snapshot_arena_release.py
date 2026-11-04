"""Keep the RAM-snapshot release gates; bind the partial-arena placement.

This changes evidence provenance only. It never programs a board and does
not relax any functional, original-STA, CDC or bitstream gate.
"""
import inspect
import json
import runpy
import sys
from pathlib import Path
from run_units import ROOT, HERE, sha
from control_command_repair import once
import finish_snapshot_ram_release as previous
from finish_snapshot_ram_release import (
    BOARD, BASE, SHORTS, RTL, SERIAL, REGISTERED_PARENT, audit_delta,
    parent_evidence, verify_wrapper,
)


def extra_evidence(board, batch, physical, serial, bind, verify):
    code = inspect.getsource(previous.extra_evidence)
    # The route wrapper is unchanged. Only the placement wrapper differs.
    code = code.replace("physical.name+'-ram-wrapper.py'",
                        "physical.name+'-arena-wrapper.py'")
    code = once(code, "HERE/'place_snapshot_ram_board.py',bind,verify)",
                "HERE/'place_snapshot_arena_free.py',bind,verify)")
    scope = dict(previous.__dict__)
    exec(compile(code, __file__ + ':inherited-gates', 'exec'), scope)
    scope['extra_evidence'](board, batch, physical, serial, bind, verify)
    from partial_cnn_ram_locations import verify_placed
    measured = bind(physical / 'cnn_ram_actual_locations.json')
    assert json.loads(json.dumps(verify_placed(physical))) == measured
    assert measured['retained'] == 29 and measured['released'] == 84
    policy = bind(physical / 'cnn_ram_placement.json')
    wrapper = bind(physical.parent / (physical.name + '-arena-wrapper.json'))
    assert wrapper['policy_sha256'] == policy['generator_sha256']
    verify({str(HERE / 'partial_cnn_ram_locations.py'): wrapper['policy_sha256']})
    for suffix in ('map.v', 'vdb', 'raminfo.pb'):
        name = 'outflow/ti60f225_oob.' + suffix
        assert sha(board / name) == sha(physical / name)
        verify({str(board / name): sha(board / name),
                str(physical / name): sha(physical / name)})
    for name in ('finish_snapshot_arena_release.py', 'place_snapshot_arena_free.py',
                 'review_snapshot_arena_free_route.py'):
        verify({str(HERE / name): sha(HERE / name)})


def package_code():
    source, code = previous.package_code()
    code = code.replace('from finish_snapshot_ram_release import ',
                        'from finish_snapshot_arena_release import ')
    code = once(code, "str(HERE/'finish_snapshot_ram_release.py'):driver_meta['adapter_sha256']",
                "str(HERE/'finish_snapshot_arena_release.py'):driver_meta['adapter_sha256']")
    return source, code


def finish_code():
    source, code = previous.finish_code()
    code = code.replace("'finish_snapshot_ram_release.py'", "'finish_snapshot_arena_release.py'")
    return source, code


def main():
    mode = sys.argv.pop(1)
    if mode == 'pll':
        import audit_pll_shift_routed as check
        check.audit_delta = audit_delta
        check.main()
        return
    if mode == 'bundle':
        source, code = package_code()
        key = '--output'
    else:
        assert mode == 'finish'
        source, code = finish_code()
        key = '--root'
    args = sys.argv[1:]
    out = Path(args[args.index(key) + 1]).resolve()
    delta = audit_delta(BOARD)
    record = BOARD / 'delta_audit.json'
    if record.exists():
        assert json.loads(record.read_text()) == delta
    else:
        record.write_text(json.dumps(delta, indent=2))
    driver = out.parent / (out.name + '-arena-driver.py')
    assert not driver.exists()
    compile(code, str(driver), 'exec')
    driver.write_text(code, encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(
        source=str(source), source_sha256=sha(source), generated_sha256=sha(driver),
        adapter_sha256=sha(__file__)), indent=2))
    sys.argv = [str(driver), *args]
    runpy.run_path(str(driver), run_name='__main__')


if __name__ == '__main__':
    main()
