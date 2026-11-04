"""Complete release gates for the proved RAM-data specialization; offline only."""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_payload_board import BOARD,PARENT as CAMERA_PARENT,audit_delta
import finish_camera_early_ratio_release as prior
import finish_camera_checked_release as original
from finish_split_logic_outputs_release import archived_placement_evidence
from finish_local_driver_release import wrappers

BASE=original.BASE;PARENT=original.PARENT
SHORTS=ROOT/'tmp/frame-execution/camera-payload-evidence1'
RTL=SHORTS/'rtl';SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new')
MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')

def payload_evidence(bind,verify):
    r=bind(ROOT/'tmp/board-test/camera-payload-reuse1/result.json')
    assert r['status']=='UNCHANGED_CAMERA_MAP_TESTS_BOUND_TO_PAYLOAD_MAP' and r['camera_cells']==992
    assert r['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v') and r['camera_or_observer_pins_changed']==0
    verify(r['evidence'])
    native=ROOT/'tmp/frame-execution/camera-payload-native2';n=bind(native/'result.json')
    assert n['status']=='PAYLOAD_SPECIALIZATION_GRAPH_PASS' and n['clones']==59 and n['changed_payload_muxes']==768
    assert n['latency_cycles_added']==0 and n['changed_other_pins']==0 and n['max_local_payload_fanout']==18
    verify(n['inputs']);verify({str(native/f):h for f,h in n['files'].items()})
    s=ROOT/'tmp/board-test/camera-payload-short4';t=bind(s/'result.json');i=bind(s/'inputs.json')
    assert t['status']=='PAYLOAD_PAIRED_OFFICIAL_RAM_FOURSTATE_PASS' and len(t['results'])==3
    assert t['inputs_sha256']==sha(s/'inputs.json') and i['source_result_sha256']==sha(native/'result.json')
    verify({str(s/f):h for f,h in i['files'].items()})
    verify({str(s/f"seed{x['seed']}.log"):x['log_sha256'] for x in t['results']})
    return dict(clones=59,payload_bits=768,max_driver_fanout=18,added_cycles=0,official_ram_seeds=3)

def parent_evidence(parent,verify):
    r=prior.parent_evidence(parent,verify);audit_delta(BOARD)
    r.update(status='FULL_VIDEO_A_CAMERA_AND_PAYLOAD_LINEAGE_BOUND',
      scope='Parent full video/A plus checked camera; only RAM payload selection specialized under proved CLKEB-disabled condition. Current map and physical gates required; no inherited board PASS.')
    return r

def warning_audit(board):
    old=prior.warning_audit(CAMERA_PARENT);audit_delta(board)
    from probe_native_open_ports import protected_pins
    from audit_native_roundtrip import tokens
    f=board/'outflow/ti60f225_oob.warn.log';body=f.read_text();source=(board/'native_source.v').read_text()
    assert '[EFX-0654' not in body
    names=re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",body)
    samples=tokens(protected_pins(source)['u_ddr3_top/samples']);assert len(samples)==40 and len(names)==10
    bits={n:39-samples.index(n) for n in names};assert sorted(bits.values())==[12,13,14,15,32,33,36,37,38,39]
    assert all(len(re.findall(r'\b'+n+r'\b',source))==1 for n in names)
    previous=(CAMERA_PARENT/'outflow/ti60f225_oob.warn.log').read_text()
    for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert body.count(marker)==previous.count(marker)
    dangling=lambda t:set(re.findall(r'Net (.*?) is dangling',t))
    groups=json.loads((ROOT/'tmp/frame-execution/camera-payload-native2/groups.json').read_text())
    allowed={r['old_select'][1] for r in groups}
    assert dangling(body)-dangling(previous)<=allowed,dangling(body)-dangling(previous)-allowed
    r=json.loads((board/'result.json').read_text())
    assert r['protected_boundary']==dict(ports=1201,bits=4128,disconnected_bits=1374,other_connections_equal=True)
    return dict(status='SAME_DISCONNECTED_BITS_AND_UNUSED_OLD_DATA_SELECTORS_CHECKED',sample_bits=bits,
      retired_selector_dangling=sorted(dangling(body)-dangling(previous)),new_functional_unconnected_inputs=False,
      evidence={**old['evidence'],str(f):sha(f),str(board/'result.json'):sha(board/'result.json'),
                str(board/'public_connectivity_audit.json'):sha(board/'public_connectivity_audit.json'),str(HERE/'finish_camera_payload_release.py'):sha(__file__)})

def camera_evidence(bind,verify):
    r=prior.camera_evidence(bind,verify);r['unchanged_camera_reuse']=payload_evidence(bind,verify);return r

def payload_prepare_code():
    from camera_early_ratio_route import prepare_code
    return once(prepare_code()[1],'from camera_checked_board import audit_delta','from camera_payload_board import audit_delta')

def extra_code():
    code=inspect.getsource(original.extra_evidence)
    # These are the same v2/v3 safety additions, rebound through the exact
    # unchanged camera graph instead of pretending old results use the new map.
    code=code.replace(' camera_evidence(bind,verify)'," camera_evidence(bind,verify)\n wiring=bind(CAMERA_PARENT/'camera_wiring_inputs.json');verify(wiring['evidence'])\n external=bind(CAMERA_PARENT/'camera_external_input_audit.json');assert external['status']=='ALL_CAMERA_MAPPED_INPUTS_DRIVEN_RAW6_OR_RAW5_ZERO_PASS';verify(external['evidence'])")
    code=prior.adapt_extra(code)
    code=once(code,"reuse=bind(MAP_AUDIT/'reuse.json');verify(reuse['evidence'])","payload_evidence(bind,verify)")
    code=code.replace("'-prepare-ratio.py'","'-prepare-payload.py'")
    code=code.replace("HERE/'camera_ratio_before_place.py'","HERE/'camera_payload_flow.py'")
    code=code.replace("HERE/'camera_checked_flow.py'","HERE/'camera_payload_flow.py'")
    code=code.replace("HERE/'camera_early_ratio_route.py'","HERE/'camera_payload_flow.py'")
    code=once(code,'from camera_early_ratio_route import prepare_code\n assert preparation.read_text()==prepare_code()[1]',
                   'assert preparation.read_text()==payload_prepare_code()')
    return code

exec(compile(extra_code(),__file__+':full-extra-gates','exec'),globals())

def package_code():
    source,code=prior.package_code()
    code=code.replace('finish_camera_early_ratio_release','finish_camera_payload_release')
    code=once(code,'from camera_checked_board import audit_delta','from camera_payload_board import audit_delta')
    code=once(code,'import audit_camera_implementation as impl','import camera_payload_implementation as impl')
    code=once(code,"verify({str(HERE/'camera_checked_board.py'):mi['camera_builder_sha256']})",
                   "verify({str(HERE/'camera_payload_board.py'):mapped['auditor_sha256']})")
    code=once(code,"functional_evidence_scope='Full parent video/A retained except checked camera configuration and passive observation; current-map tests and physical gates required'",
                   "functional_evidence_scope='Full video/A retained, checked camera configuration plus proved local RAM payload specialization; current-map tests and physical gates required'")
    compile(code,'payload-package-preflight','exec');return source,code

def main():
    if sys.argv[1:]==['--check-only']:
        _,code=package_code();compile(code,'payload-package','exec')
        print('PAYLOAD_COMPLETE_RELEASE_GENERATORS_COMPILE');return
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='bundle':source,code=package_code();key='--output'
    else:
        assert mode=='finish'
        import finish_local_driver_release as old
        source,code=old.finish_code();code=code.replace('finish_local_driver_release.py','finish_camera_payload_release.py')
        code=code.replace('a.output.resolve()],90)','a.output.resolve()],300)');key='--root'
    args=sys.argv[1:];out=Path(args[args.index(key)+1]).resolve();audit_delta(BOARD)
    driver=out.parent/(out.name+'-camera-release.py');assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
