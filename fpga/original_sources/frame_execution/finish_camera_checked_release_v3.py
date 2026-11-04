"""Current complete release gates, including explicit config-clock correction."""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once


def targeted_evidence(physical,batch,expected,bind,verify):
    """Bind routing-only constraints separately from the actual release STA."""
    sdc=physical/'ti60f225_oob.sdc'
    rule=bind(physical/'targeted_route_margin.json')
    meta=bind(physical/'physical_inputs.json')
    assert meta['targeted_route_margin_sha256']==sha(physical/'targeted_route_margin.json')
    from camera_targeted_route_margin import EXTRA
    assert rule['status']=='ONE_PATH_TIGHTENED_FOR_ROUTING_ONLY' and rule['extra']==EXTRA
    assert not rule['hardware_clocks_changed'] and rule['new_false_paths']==rule['new_multicycle_paths']==0
    acceptance=Path(rule['canonical_sdc'])
    assert acceptance==physical/'camera_acceptance.sdc'
    verify({str(acceptance):rule['canonical_sdc_sha256'],str(sdc):rule['optimization_sdc_sha256'],
            str(HERE/'camera_targeted_route_margin.py'):rule['script_sha256'],rule['probe']:rule['probe_sha256']})
    assert acceptance.read_text()==expected and sdc.read_text()==expected+EXTRA
    probe=bind(Path(rule['probe']));probe_inputs=Path(rule['probe']).with_name('inputs.json')
    pi=bind(probe_inputs)
    assert probe['status']=='TARGETED_MAX_DELAY_IS_STRICTER_SAME_ROUTE_PASS'
    assert probe['after_ns']<probe['before_ns'] and probe['inputs_sha256']==sha(probe_inputs)
    assert pi['script_sha256']==rule['script_sha256'] and pi['extra']==EXTRA
    verify(pi['original']);verify(probe['reports'])
    canonical=batch/'canonical-seed3'
    inputs=bind(canonical/'inputs.json');result=bind(canonical/'result.json')
    assert result['status']=='CANONICAL_ROUTED_TIMING_PASS_NOT_RELEASE'
    assert result['inputs_sha256']==sha(canonical/'inputs.json')
    assert inputs['canonical_sdc_sha256']==sha(acceptance)==sha(canonical/'ti60f225_oob.sdc')
    assert inputs['removed_setup_margin'] is None and inputs['min_setup_slack_ns']==inputs['min_hold_slack_ns']==0
    driver=batch/'canonical-seed3-targeted-canonical.py';dm=bind(driver.with_suffix('.json'))
    verify({str(driver):dm['generated_sha256'],dm['source']:dm['source_sha256'],
            str(HERE/'camera_targeted_route_margin.py'):dm['adapter_sha256']})
    assert Path(dm['source'])==HERE/'canonical_snapshot_sta.py'
    assert inputs['runner_sha256']==dm['generated_sha256']==sha(canonical/'runner.py')
    policy=sha(HERE/'timing_acceptance.py')
    verify({str(HERE/'timing_acceptance.py'):policy,str(batch/'timing_acceptance.py'):policy,
            str(canonical/'timing_acceptance.py'):policy})
    assert inputs['timing_policy_sha256']==policy
    return acceptance


def adapt_extra(code):
    code=once(code,"wrappers(batch,[('-local-driver-route',HERE/'camera_checked_flow.py')]",
                   "wrappers(batch,[('-local-driver-route',HERE/'camera_checked_flow.py')]")
    # generated() is still owned by camera_checked_flow; separately bind the
    # new caller responsible for constructing the reviewed route adapter.
    needle=" assert (physical/'ti60f225_oob.sdc').read_text()==expected and pins==rebound['pins'];"
    add=""" ratio=bind(physical/'config_clock_ratio.json')
 assert bind(physical/'physical_inputs.json')['config_clock_ratio_sha256']==sha(physical/'config_clock_ratio.json')
 verify({str(HERE/'camera_config_clock_ratio.py'):ratio['transform_sha256'],str(physical/'ti60f225_oob.peri.xml'):ratio['peri_sha256']})
 import hashlib,os
 assert hashlib.sha256(expected.replace('\\n',os.linesep).encode('utf8')).hexdigest()==ratio['before_sdc_sha256']
 from camera_config_clock_ratio import transform
 expected,change=transform(expected,(physical/'ti60f225_oob.peri.xml').read_text())
 assert change==ratio['change']
 acceptance=targeted_evidence(physical,batch,expected,bind,verify) if (physical/'targeted_route_margin.json').exists() else physical/'ti60f225_oob.sdc'
 assert sha(acceptance)==ratio['after_sdc_sha256']
 assert acceptance.read_text()==expected and pins==rebound['pins'];"""
    code=once(code,needle,add)
    code=once(code," camera_evidence(bind,verify)",""" camera_evidence(bind,verify)
 reuse=bind(MAP_AUDIT/'reuse.json');verify(reuse['evidence'])
 scope=bind(ROOT/'tmp/board-test/camera-observer-midline-scope1/result.json')
 assert scope['status']=='OBSERVER_MIDLINE_RESET_SCOPE_CONFIRMED'
 verify(scope['evidence'])
 negative=bind(ROOT/'tmp/board-test/camera-cdc-report-rejection1.json')
 assert negative['status']=='CDC_REPORT_NEGATIVE_GUARDS_PASS' and len(negative['cases'])==3
 verify(negative['evidence'])
 for n in ('camera_checked_route_v3.py','camera_checked_route_v4.py','camera_checked_route_v5.py','camera_targeted_route_margin.py','camera_config_clock_ratio.py','camera_checked_cdc_v2.py','camera_checked_gray_v2.py','audit_camera_implementation.py','CAMERA_CHECKED_RELEASE.md'):
  verify({str(HERE/n):sha(HERE/n)})""")
    return code


# Reuse the exact v2 evidence definitions while adapting only changed schema
# and provenance. The original scripts and their failed receipts stay intact.
_source=(HERE/'finish_camera_checked_release_v2.py').read_text()
_source=once(_source," exec(compile(code,__file__+':'+name,'exec'),globals())",
    " if name=='extra_evidence':code=adapt_extra(code)\n if name=='main':code=code.replace(\";key='--root'\",\";code=code.replace('a.output.resolve()],90)','a.output.resolve()],240)');key='--root'\")\n exec(compile(code,__file__+':'+name,'exec'),globals())")
_source=_source.replace('finish_camera_checked_release_v2','finish_camera_checked_release_v3')
# Rechecking the complete preserved parent plus this mapped delta reads several
# large netlists. Give the offline bundle audit its own finite 240-second cap.
_source=once(_source,"if __name__=='__main__':main()",'')
exec(compile(_source,__file__+':v2-definitions','exec'),globals())
_inherited_package=package_code


def package_code():
    source,code=_inherited_package()
    code=once(code,'from camera_checked_cdc import inventory','from camera_checked_cdc_v2 import inventory')
    code=once(code,'import audit_control_state_implementation as impl','import audit_camera_implementation as impl')
    # A camera payload replaces the obsolete cell probes; do not ship an old
    # snapshot decoder that would silently label camera bits as primitive data.
    code=once(code,"('read_startup_cause_snapshot','read_startup_cause_status')", "('read_startup_cause_status',)")
    start=code.index("    (out/'read_camera_checked_snapshot.py').write_text(")
    end=code.index('\n    manifest=dict(',start)
    code=code[:start]+"""    reader='import sys\\nsys.path.insert(0,'+repr(str(HERE))+')\\nfrom read_camera_checked_snapshot import main\\nif __name__==\"__main__\":main()\\n'
    compile(reader,str(out/'read_camera_checked_snapshot.py'),'exec')
    (out/'read_camera_checked_snapshot.py').write_text(reader,encoding='utf8')
"""+code[end:]
    code=once(code,'camera_checks=camera_evidence(bind,verify),',
        "camera_checks=camera_evidence(bind,verify),config_clock_analysis=bind(physical/'config_clock_ratio.json'),routing_only_target=bind(physical/'targeted_route_margin.json') if (physical/'targeted_route_margin.json').exists() else None,")
    code=once(code,"    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));verify(evidence)",
        "    for f in out.glob('*.py'):compile(f.read_text(),str(f),'exec')\n    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));verify(evidence)")
    compile(code,'camera-v3-package-preflight','exec')
    return source,code


if __name__=='__main__':main()
