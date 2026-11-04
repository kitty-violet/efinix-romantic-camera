"""Full current implementation release with explicit four-SRL proof lineage."""
import json,runpy,sys
from pathlib import Path
import continuous_srl_release_evidence as evidence
from continuous_srl_release_evidence import *
import finish_continuous_control_release3 as parent
from continuous_srl_register_board import BOARD,TRANSFORM,PARENT as OLD_BOARD,audit_delta,transform_check
from generate_continuous_srl_release_evidence import verify_generated

SHORTS=ROOT/'tmp/frame-execution/continuous-srl-register-evidence2'
RTL=SHORTS/'rtl';MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')
SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new')
for _name in ('SHORTS','RTL','MAP_AUDIT','ADAPTER','SERIAL'):setattr(evidence,_name,globals()[_name])

def payload_evidence(bind,verify):
    # Execute all historical numerical/controller proofs against their real
    # original map. The exact structural delta connects that map to this one.
    contract=parent.payload_evidence(bind,verify)
    verify_generated();bind(HERE/'continuous_srl_release_evidence.json')
    proof=transform_check();bind(TRANSFORM/'result.json');verify(proof['inputs'])
    assert proof['retained_primitives']==60128 and proof['latency_cycles_added']==0
    mapped=bind(BOARD/'result.json');verify(mapped['evidence']);audit_delta(BOARD)
    assert mapped['dead_generated_luts']==[] and len(mapped['added_cells'])==76
    for name in ('continuous-srl-register-short1',):
        root=ROOT/'tmp/frame-execution'/name;r=bind(root/'result.json');i=bind(root/'inputs.json')
        assert r['status']=='ACTUAL_FOUR_NATIVE_SRL_EXPANSIONS_OFFICIAL_MODEL_SHORT_PASS'
        assert r['inputs_sha256']==sha(root/'inputs.json')
        verify(i['files']);verify({str(root/'simulate.log'):r['simulation_sha256']})
        warning=bind(root/'warning-audit.json')
        assert warning['status']=='ONLY_FOUR_ORIGINAL_UNUSED_Q7_PORT_WARNINGS' and warning['other_model_warnings']==0
    address=bind(ROOT/'tmp/board-test/continuous-srl-address-known1.json');verify(address['evidence'])
    assert address['status']=='SRL_ADDRESS_STATE_AND_COMBINATIONAL_DRIVER_CONES_CLOSED'
    assert address['new_map_sha256']==mapped['map_sha256']
    for name in ('continuous_srl_register_native.py','continuous_srl_register_board.py','continuous_srl_register_flow.py','continuous_srl_register_flow_v2.py','continuous_srl_register_route.py','continuous_srl_register_place.py','continuous_srl_register_implementation.py','generate_continuous_srl_release_evidence.py','continuous_srl_release_evidence.py','finish_continuous_srl_release.py','validate_srl_register_serial.py','validate_srl_register_adapter.py','windows_process_tree_safe.py'):
        verify({str(HERE/name):sha(HERE/name)})
    safety=bind(ROOT/'tmp/board-test/safe-process-stop-test1.json');verify(safety['evidence'])
    assert safety['status']=='PID_REUSE_EXCLUSION_AND_OWNED_CHILD_STOP_PASS'
    return dict(contract,srl_timing_delta=dict(replaced=4,extra_ff=40,extra_lut=36,removed_srl=4,added_cycles=0),
        historical_control_map_sha256=sha(OLD_BOARD/'outflow/ti60f225_oob.map.v'),current_map_sha256=mapped['map_sha256'])

def parent_evidence(manifest,verify):
    result=parent.parent_evidence(manifest,verify);audit_delta(BOARD)
    return dict(result,current_structural_proof=str(TRANSFORM/'result.json'),current_structural_proof_sha256=sha(TRANSFORM/'result.json'),
        inherited_board_pass=False)

def camera_evidence(bind,verify):
    result=parent.camera_evidence(bind,verify)
    return dict(result,historical_continuous_map_sha256=sha(OLD_BOARD/'outflow/ti60f225_oob.map.v'),
        current_map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v'),current_srl_contract=payload_evidence(bind,verify),
        camera_ddr_display_connections_unchanged=True)

def warning_audit(board):
    assert board==BOARD;audit_delta(board)
    result=parent.warning_audit(OLD_BOARD)
    from probe_native_open_ports import protected_pins
    from audit_native_roundtrip import tokens
    import re
    a=(OLD_BOARD/'outflow/ti60f225_oob.warn.log').read_text();b=(board/'outflow/ti60f225_oob.warn.log').read_text()
    assert '[EFX-0654' not in b
    for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert a.count(marker)==b.count(marker)
    dangling=lambda s:set(re.findall(r'Net (.*?) is dangling',s))
    assert dangling(a)==dangling(b)
    names=re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",b)
    src=(board/'native_source.v').read_text();samples=tokens(protected_pins(src)['u_ddr3_top/samples'])
    assert len(names)==10 and len(samples)==40
    assert sorted(39-samples.index(n) for n in names)==[12,13,14,15,32,33,36,37,38,39]
    assert all(len(re.findall(r'\b'+re.escape(n)+r'\b',src))==1 for n in names)
    result.update(status='FOUR_SRL_DELTA_SAME_PROTECTED_OPEN_BITS_AND_WARNING_CLASSES_PASS')
    result['evidence'].update({str(f):sha(f) for f in (OLD_BOARD/'outflow/ti60f225_oob.warn.log',board/'outflow/ti60f225_oob.warn.log',board/'public_connectivity_audit.json',Path(__file__))})
    return result

# Physical gates retain their complete original bodies and use the current
# paths. Only their functional-proof callbacks follow the explicit lineage.
evidence.payload_evidence=payload_evidence
evidence.parent_evidence=parent_evidence
evidence.camera_evidence=camera_evidence
evidence.warning_audit=warning_audit

def extra_evidence(board,batch,physical,serial,bind,verify):
    evidence.extra_evidence(board,batch,physical,serial,bind,verify)
    payload_evidence(bind,verify)

def package_code():
    source,code=parent.package_code()
    for old,new in (('finish_continuous_control_release3','finish_continuous_srl_release'),
                    ('continuous_control_board','continuous_srl_register_board'),
                    ('continuous_control_implementation','continuous_srl_register_implementation'),
                    ('CONTINUOUS_RUN_RELEASE.md','CONTINUOUS_SRL_RELEASE.md')):
        assert old in code;code=code.replace(old,new)
    needle="functional_evidence_scope='Full video/A150MHz preserved; bounded issuer deadline and 26-bit count, hardware frame admission and atomic progress page changed. Exact late-writer-error factorization and targeted equivalent data-driver replication added without latency. Current-map regression and physical gates required; complete frame and board picture remain unverified'"
    assert code.count(needle)==1
    code=code.replace(needle,"functional_evidence_scope='Full video/A150MHz and finite512-frame control retained. Four dynamic-address SRLs replaced by verified same-cycle FF/mux structures under closed known-address control cones. Exact unchanged graph binds historical numerical evidence; current interface/reset, original STA/CDC/Gray/PLL and bitstream gates independently checked. Continuous picture/FPS awaits supervised board test'")
    compile(code,__file__+':package','exec');return source,code

def finish_code():
    source,code=parent.finish_code()
    assert 'finish_continuous_control_release3.py' in code
    code=code.replace('finish_continuous_control_release3.py','finish_continuous_srl_release.py')
    code=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')
    compile(code,__file__+':finish','exec');return source,code

def main():
    if sys.argv[1:]==['--check-only']:
        verify_generated();package_code();finish_code()
        assert evidence.payload_prepare_code()==__import__('continuous_srl_register_route').prepare_code()[1]
        print('SRL_FULL_OFFLINE_RELEASE_ENTRY_GATES_COMPILE');return
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='bundle':source,code=package_code();key='--output'
    else:
        assert mode=='finish';source,code=finish_code();key='--root'
    args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD)
    driver=out.parent/(out.name+'-cnn-release.py');assert not driver.exists()
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
