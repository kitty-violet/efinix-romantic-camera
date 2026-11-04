"""Complete compact-island release plus routing-target provenance, no bypass."""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
import finish_camera_factor_compact_island_release as prior
from camera_compact_timing_flow import EXTRA,PROBE,verified_probe

BASE,SHORTS,RTL,SERIAL=prior.BASE,prior.SHORTS,prior.RTL,prior.SERIAL
parent_evidence,warning_audit,camera_evidence=prior.parent_evidence,prior.warning_audit,prior.camera_evidence

def extra_evidence(board,batch,physical,serial,bind,verify):
    prior.extra_evidence(board,batch,physical,serial,bind,verify)
    verified_probe()
    receipt=bind(batch/'route_optimization_inputs.json')
    assert receipt['status']=='ORIGINAL_SDC_RESTORED_BEFORE_ACCEPTANCE'
    assert receipt['full_original_sta_required'] and not receipt['hardware_clock_changed'] and not receipt['placement_changed']
    assert receipt['extra']==EXTRA and receipt['probe']==str(PROBE/'result.json')
    acceptance=batch/'route_acceptance.sdc';optimization=batch/'route_optimization.sdc'
    verify({str(acceptance):receipt['canonical_sha256'],str(optimization):receipt['optimization_sha256'],
            receipt['probe']:receipt['probe_sha256'],str(HERE/'camera_compact_timing_flow.py'):receipt['adapter_sha256']})
    assert acceptance.read_bytes()==(physical/'ti60f225_oob.sdc').read_bytes()
    assert optimization.read_text()==acceptance.read_text()+EXTRA
    assert (batch/'canonical-seed3/ti60f225_oob.sdc').read_bytes()==acceptance.read_bytes()
    # Reconstruct every generated route instruction; the only new execution
    # surrounds route with a temporary stricter SDC and a finally restoration.
    import camera_compact_timing_flow as flow
    source=inspect.getsource(flow.route)
    source=once(source,"    generated(original,code,'-local-driver-route')","    return code")
    env=dict(flow.__dict__);exec(compile(source,__file__+':route-reconstruction','exec'),env)
    expected=env['route']()
    actual=batch.parent/(batch.name+'-local-driver-route.py')
    assert actual.read_text()==expected
    assert (batch/'runner.py').read_text()==expected
    verify(bind(PROBE/'result.json')['evidence'])

def package_code():
    source,code=prior.package_code()
    code=code.replace('finish_camera_factor_compact_island_release','finish_camera_compact_timing_release')
    code=once(code,"    manifest=dict(status=", "    manifest=dict(route_optimization=bind(batch/'route_optimization_inputs.json'),status=")
    compile(code,'compact-timing-complete-gates','exec')
    return source,code

def main():
    if sys.argv[1:]==['--check-only']:
        package_code();print('COMPACT_PROCESS_TARGET_ALL_RELEASE_GENERATORS_COMPILE');return
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=prior.prior.audit_delta;check.main();return
    if mode=='bundle':source,code=package_code();key='--output'
    else:
        assert mode=='finish'
        from finish_local_driver_release import finish_code
        source,code=finish_code()
        code=code.replace('finish_local_driver_release.py','finish_camera_compact_timing_release.py').replace('a.output.resolve()],240)','a.output.resolve()],360)');key='--root'
    args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve()
    prior.prior.audit_delta(prior.prior.BOARD);assert not out.exists()
    driver=out.parent/(out.name+'-camera-release.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
