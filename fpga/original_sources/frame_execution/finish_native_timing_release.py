"""Retain all release gates for the one exact-placement timing-route variant."""
import inspect,json,runpy,sys
from pathlib import Path
import finish_native_split_release as prior
from finish_native_split_release import BASE,BOARD,PARENT,SHORTS,RTL,SERIAL,MAP_AUDIT,ADAPTER,STATUS,parent_evidence,warning_audit
from run_units import ROOT,HERE,sha
from control_command_repair import once
from native_split_board import audit_delta

def extra_evidence(board,batch,physical,serial,bind,verify):
    code=inspect.getsource(prior.extra_evidence)
    code=once(code,"(batch,'route_native_split_board.py')","(batch,'reroute_native_timing.py')")
    scope=dict(prior.__dict__);exec(compile(code,__file__+':unchanged-release-gates','exec'),scope)
    scope['extra_evidence'](board,batch,physical,serial,bind,verify)
    inheritance=bind(physical/'route_inheritance.json')
    assert all(inheritance[k] is False for k in ('map_changed','placement_changed','constraints_changed','seed_changed','hardware_access'))
    assert inheritance['route_profile_cli_override']==dict(before='CONGESTION_2',after='TIMING_1')
    verify(inheritance['source_files']);verify(inheritance['copied_files'])
    assert 'optimization_level=TIMING_1' in (physical/'route_only.cmd').read_text()
    assert 'optimization_level=TIMING_1' in (batch/'route.log').read_text()
    review=bind(physical/'route_review.json')
    assert review['route_profile_cli_override']==inheritance['route_profile_cli_override']
    verify({str(HERE/'reroute_native_timing.py'):review['required_evidence'][str(HERE/'reroute_native_timing.py')]})

def package_code():
    source,code=prior.package_code()
    code=code.replace('finish_native_split_release','finish_native_timing_release')
    code=once(code,'native_local_split_pairs=128,',
        "route_profile_cli_override={'before':'CONGESTION_2','after':'TIMING_1'},native_local_split_pairs=128,")
    return source,code

def finish_code():
    source,code=prior.finish_code()
    return source,code.replace('finish_native_split_release.py','finish_native_timing_release.py')

def main():
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='bundle':source,code=package_code();key='--output'
    else:
        assert mode=='finish';source,code=finish_code();key='--root'
    args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve()
    assert audit_delta(BOARD)==json.loads((BOARD/'delta_audit.json').read_text())
    driver=out.parent/(out.name+'-native-driver.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
