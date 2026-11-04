"""Keep every full release gate and propagate the current package entry to children."""
import json,sys,runpy
from pathlib import Path
import finish_continuous_control_release2 as previous
from finish_continuous_control_release2 import *
SERIAL=ROOT/'tmp/frame-execution/continuous-control-serial3/new'

def package_code():
    source,code=previous.package_code()
    assert 'from finish_continuous_control_release import' in code
    code=code.replace('from finish_continuous_control_release import','from finish_continuous_control_release3 import')
    assert code.count("HERE/'finish_continuous_control_release.py'")==1
    code=code.replace("HERE/'finish_continuous_control_release.py'","HERE/'finish_continuous_control_release3.py'")
    assert "HERE/'CNN_RUN_BUDGET_RELEASE.md'" in code
    code=code.replace("HERE/'CNN_RUN_BUDGET_RELEASE.md'","HERE/'CONTINUOUS_RUN_RELEASE.md'")
    # Preserve the already-tested Windows path-key repair from the released
    # sparse package; otherwise the same packaging-only KeyError recurs.
    needle="mapped['evidence'][str(HERE/'continuous_control_board.py')]"
    assert code.count(needle)==1
    code=code.replace(needle,"recorded_source_digest(mapped['evidence'],HERE/'continuous_control_board.py')")
    needle='from run_units import ROOT,HERE,sha'
    assert code.count(needle)==1
    code=code.replace(needle,needle+'\nfrom release_evidence_paths import recorded_source_digest')
    compile(code,__file__+':package','exec')
    return source,code

def payload_evidence(bind,verify):
    contract=previous.payload_evidence(bind,verify)
    paths=bind(ROOT/'tmp/board-test/release-evidence-paths-test1.json');verify(paths['evidence'])
    assert paths['status']=='RECORDED_SOURCE_PATH_POSITIVE_AND_NEGATIVE_PASS'
    verify({str(Path(__file__)):sha(__file__),str(HERE/'CONTINUOUS_RUN_RELEASE.md'):sha(HERE/'CONTINUOUS_RUN_RELEASE.md')})
    return contract

def extra_evidence(board,batch,physical,serial,bind,verify):
    review=bind(physical/'route_review.json')
    profile=review['route_profile_cli_override']
    assert profile['before']=='CONGESTION_2' and profile['after'] in ('CONGESTION_2','TIMING_1')
    if profile['after']=='CONGESTION_2':
        return previous.extra_evidence(board,batch,physical,serial,bind,verify)
    # Match the actual recorded timing-route wrapper. Every inherited
    # placement, route, original STA/Gray, PLL and source check still executes.
    wrapper=HERE/'continuous_control_route_timing1.py'
    meta=bind(batch.parent/(batch.name+'-local-driver-route.json'))
    verify({str(wrapper):meta['wrapper_sha256']})
    import continuous_control_route_timing1 as route
    _,generated=route.route_code()
    driver=batch.parent/(batch.name+'-local-driver-route.py')
    assert driver.read_text(encoding='utf8')==generated
    verify({str(driver):meta['driver_sha256']})
    scope=previous.extra_evidence.__globals__;original=scope['wrappers']
    def selected_wrappers(base,pairs,bind,verify):
        if base==batch:
            pairs=[(suffix,wrapper if parent==HERE/'continuous_control_route.py' else parent) for suffix,parent in pairs]
        return original(base,pairs,bind,verify)
    scope['wrappers']=selected_wrappers
    try:return previous.extra_evidence(board,batch,physical,serial,bind,verify)
    finally:scope['wrappers']=original

def finish_code():
    import finish_local_driver_release as old
    source,code=old.finish_code()
    assert 'finish_local_driver_release.py' in code
    code=code.replace('finish_local_driver_release.py','finish_continuous_control_release3.py')
    # The prior successful full bundle audit needed404s. Preserve every
    # check, allow600s, and cache only immutable functional sources.
    assert 'a.output.resolve()],240)' in code
    code=code.replace('a.output.resolve()],240)','a.output.resolve()],600)')
    needle="            proc=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)"
    assert code.count(needle)==1
    code=code.replace(needle,"""            command=[sys.executable,str(HERE/script),*map(str,args)]
            if name=='bundle':
                command=[sys.executable,'-B',str(ROOT/'tmp/board-test/run-with-continuous-locked-evidence.py'),
                    '--cache-log',str(root/'bundle-hash-cache.json'),str(HERE/script),*map(str,args)]
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)""")
    compile(code,__file__+':finish','exec')
    return source,code

def main():
    if sys.argv[1]=='--check-only':
        package_code();finish_code();print('CONTINUOUS_CURRENT_ENTRY_AND_FULL_GATES_COMPILE');return
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='bundle':source,code=package_code();key='--output'
    else:
        assert mode=='finish';source,code=finish_code();key='--root'
    args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD)
    driver=out.parent/(out.name+'-cnn-release.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
