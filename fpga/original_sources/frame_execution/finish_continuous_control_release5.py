"""Full release for the matching SEED1 placement/route; no hardware execution."""
import json, runpy, sys
from pathlib import Path
import continuous_seed1_release_evidence as evidence
from continuous_seed1_release_evidence import *
import finish_continuous_control_release3 as previous
from generate_continuous_seed1_release_evidence import verify_generated

SERIAL = ROOT/'tmp/frame-execution/continuous-control-serial3/new'

def payload_evidence(bind, verify):
    contract = evidence.payload_evidence(bind, verify)
    generated = verify_generated()
    bind(HERE/'continuous_seed1_release_evidence.json')
    assert generated['checks_removed'] is False
    safe=bind(ROOT/'tmp/board-test/safe-process-stop-test1.json'); verify(safe['evidence'])
    assert safe['status']=='PID_REUSE_EXCLUSION_AND_OWNED_CHILD_STOP_PASS' and safe['unrelated_sentinel_preserved']
    paths = bind(ROOT/'tmp/board-test/release-evidence-paths-test1.json'); verify(paths['evidence'])
    assert paths['status'] == 'RECORDED_SOURCE_PATH_POSITIVE_AND_NEGATIVE_PASS'
    for name in ('finish_continuous_control_release5.py','generate_continuous_seed1_release_evidence.py','continuous_seed1_release_evidence.py','continuous_control_seed1_route.py','continuous_control_seed1_place.py','CONTINUOUS_RUN_RELEASE.md'):
        verify({str(HERE/name):sha(HERE/name)})
    return contract

def package_code():
    source, code = previous.package_code()
    assert code.count('from finish_continuous_control_release3 import') == 1
    assert code.count("HERE/'finish_continuous_control_release3.py'") == 1
    code = code.replace('from finish_continuous_control_release3 import','from finish_continuous_control_release5 import')
    code = code.replace("HERE/'finish_continuous_control_release3.py'","HERE/'finish_continuous_control_release5.py'")
    assert code.count('gray-seed3')==1
    code=code.replace('gray-seed3','gray-seed1')
    compile(code, __file__+':package', 'exec')
    return source, code

def finish_code():
    source, code = previous.finish_code()
    assert 'finish_continuous_control_release3.py' in code
    code = code.replace('finish_continuous_control_release3.py','finish_continuous_control_release5.py')
    assert code.count('gray-seed3')==1
    code=code.replace('gray-seed3','gray-seed1')
    compile(code, __file__+':finish', 'exec')
    return source, code

def main():
    if sys.argv[1:] == ['--check-only']:
        verify_generated(); package_code(); finish_code()
        assert payload_prepare_code() == __import__('continuous_control_seed1_route').prepare_code()[1]
        print('SEED1_CURRENT_ENTRY_AND_ALL_RELEASE_GATES_COMPILE'); return
    mode = sys.argv.pop(1)
    if mode == 'pll':
        import audit_pll_shift_routed as check
        check.audit_delta = audit_delta; check.main(); return
    if mode == 'bundle': source,code=package_code(); key='--output'
    else:
        assert mode == 'finish'
        source,code=finish_code(); key='--root'
    args=sys.argv[1:]; out=Path(args[args.index(key)+1]).resolve(); audit_delta(BOARD)
    driver=out.parent/(out.name+'-cnn-release.py'); assert not driver.exists()
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args]; runpy.run_path(str(driver),run_name='__main__')

if __name__ == '__main__': main()
