"""Use the seed1 C2 placement with its matching route recipe and all original gates."""
import json, runpy, sys
from pathlib import Path
import continuous_control_route as base
from run_units import ROOT, sha

def adapted(fn):
    source, code = fn()
    if fn == base.route_code:
        assert code.count('from windows_process_tree import stop_launched_tree')==1
        code=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')
        for old,new in (('canonical-seed3','canonical-seed1'),('gray-seed3','gray-seed1')):
            assert code.count(old)==1,(old,code.count(old))
            code=code.replace(old,new)
    compile(code,__file__,'exec')
    return source,code

def prepare_code(): return adapted(base.prepare_code)
def route_code(): return adapted(base.route_code)

def main():
    action = sys.argv.pop(1)
    if action == '--check-only':
        prepare_code(); route_code(); print('SEED1_MATCHING_ROUTE_ORIGINAL_GATES_COMPILE'); return
    if action == 'prepare':
        source, code = prepare_code()
        review = Path(sys.argv[sys.argv.index('--review')+1]).resolve()
        data = json.loads(review.read_text()); root = Path(data['physical']); suffix = '-prepare-cnn-budget'
        assert not (Path(data['placement'])/'arena_island_placement.json').exists()
    else:
        assert action == 'route'
        source, code = route_code()
        root = Path(sys.argv[sys.argv.index('--root')+1]).resolve(); suffix = '-local-driver-route'
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    driver = root.parent/(root.name+suffix+'.py'); assert not driver.exists()
    driver.write_text(code, encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),driver_sha256=sha(driver),wrapper_sha256=sha(__file__),adapter_sha256=sha(__file__),hardware_access=False),indent=2))
    sys.argv = [str(driver), *sys.argv[1:]]
    runpy.run_path(str(driver), run_name='__main__')

if __name__ == '__main__': main()
