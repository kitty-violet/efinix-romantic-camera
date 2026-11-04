"""Retain every full-system release gate for the two bounded smaller deltas."""
from pathlib import Path
from control_command_repair import once

def code_for(variant):
    assert variant in ('minimal', 'data64')
    source = Path(__file__).with_name('finish_cnn_timing_closure_release.py')
    code = source.read_text(encoding='utf-8')
    old = "[('cnn-timing-closure-drivers1','ACTUAL_MAP_LOCAL_DRIVERS_FOURSTATE_PASS'),('cnn-timing-closure-controls1','MAPPED_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS')]"
    tests = [(f'cnn-{variant}-controls{2 if variant == "minimal" else 1}',
              'MAPPED_WRITER_ERROR_COFACTOR_OFFICIAL_LUT_SHORT_PASS')]
    if variant == 'data64':
        tests.insert(0, ('cnn-data64-drivers1', 'ACTUAL_MAP_LOCAL_DRIVERS_FOURSTATE_PASS'))
    code = once(code, old, repr(tests))
    code = code.replace('cnn-timing-closure-evidence1', f'cnn-{variant}-evidence1')
    code = once(code, " assert not (physical/'arena_island_placement.json').exists()", """ from cnn_run_budget_locations import verify_placed as check_arena
 assert check_arena(physical)['count']==36
 for name in ('arena_island_placement.json','arena_island_actual_locations.json'):
  item=bind(physical/name)
  if 'evidence' in item:verify(item['evidence'])""")
    for stem in ('finish_cnn_timing_closure_release', 'cnn_timing_closure_board',
                 'cnn_timing_closure_flow', 'cnn_timing_closure_implementation',
                 'cnn_timing_closure_route'):
        code = code.replace(stem, stem.replace('cnn_timing_closure', f'cnn_{variant}'))
    code = code.replace('cnn_timing_closure_native.py', 'cnn_small_native.py')
    code = code.replace('cnn_timing_closure_place.py', 'cnn_small_place.py')
    code = once(code, "'validate_cnn_timing_controls.py',", "'validate_cnn_timing_controls.py','validate_cnn_small_controls.py','cnn_small_release_factory.py','cnn_small_board_factory.py',")
    compile(code, str(source) + ':' + variant, 'exec')
    return code
