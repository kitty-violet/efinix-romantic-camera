"""Bind local-driver proofs to the established complete release gates.

No hardware access; negative routed timing still stops before bitstream release.
"""
import inspect,json,re,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from local_driver_board import BASE,PARENT as NATIVE_PARENT,MAPPED,FOUR,audit_delta
from finish_split_logic_outputs_release import archived_placement_evidence

BOARD=ROOT/'tmp/frame-execution/connected-board-local-drivers1'
PARENT=ROOT/'output/deployment/board-full-150-startup-cause-20261011'
SHORTS=ROOT/'tmp/frame-execution/native-local-driver-evidence3'
RTL=SHORTS/'rtl';SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new')
MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')

def parent_evidence(parent,verify):
    assert Path(parent['board'])==BASE
    verify(parent['required_evidence']);audit_delta(BOARD)
    return dict(status='FULL_RTL_NATIVE_SPLIT_AND_LOCAL_DRIVER_LINEAGE_BOUND',
        manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
        scope='Original video/A RTL, pins and clocks unchanged. Current native map and physical evidence required; no inherited board PASS.')

def warning_audit(board):
    from finish_native_split_release import warning_audit as previous
    prior=previous(NATIVE_PARENT);audit_delta(board)
    mapped=json.loads((MAPPED/'result.json').read_text())
    assert mapped['lexical_audit']['status']=='LEXICAL_PUBLIC_NET_BIJECTION_PASS'
    assert mapped['protected_boundary']==dict(ports=1201,bits=4128,disconnected_bits=1374,other_connections_equal=True)
    assert mapped['final_auditor_sha256']==sha(HERE/'map_native_local_drivers.py')
    evidence=dict(prior['evidence'])
    old=(NATIVE_PARENT/'outflow/ti60f225_oob.warn.log').read_text()
    current=board/'outflow/ti60f225_oob.warn.log';body=current.read_text()
    assert '[EFX-0654' not in body
    names=lambda text:re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",text)
    assert names(body)==names(old)==['Open_'+str(n) for n in range(8,18)]
    for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):
        assert body.count(marker)==old.count(marker),(marker,body.count(marker),old.count(marker))
    for f in (MAPPED/'result.json',MAPPED/'inputs.json',MAPPED/'public_connectivity_audit.json',current,
              HERE/'map_native_local_drivers.py',HERE/'native_local_drivers.py'):
        evidence[str(f)]=sha(f)
    assert evidence[str(MAPPED/'public_connectivity_audit.json')]==mapped['public_audit_sha256']
    return dict(status='PARENT_OPEN_INPUT_PROOF_REBOUND_BY_CURRENT_LEXICAL_GRAPH',
        inherited_input_ports=prior['checked_input_ports'],inherited_bits=prior['checked_bits'],
        same_protected_boundary=True,new_functional_unconnected_inputs=False,
        private_vendor_warnings_retained=True,warning_list_may_be_capped=True,
        evidence=evidence,ddr_calibration_proven=False)

def wrappers(root,pairs,bind,verify):
    for suffix,wrapper in pairs:
        driver=root.parent/(root.name+suffix+'.py');m=bind(driver.with_suffix('.json'))
        verify({str(driver):m['driver_sha256'],m['source']:m['source_sha256'],str(wrapper):m['wrapper_sha256']})

def extra_evidence(board,batch,physical,serial,bind,verify):
    assert board==BOARD and audit_delta(board)==bind(board/'delta_audit.json')
    ma=bind(MAP_AUDIT/'result.json');mapped=bind(board/'result.json')
    assert ma['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS' and ma['checks']==17504
    assert ma['map_sha256']==mapped['map_sha256'] and ma['board_inputs_sha256']==sha(board/'inputs.json')
    verify(ma['evidence'])
    for n,checker in (('new_diagnostic_cells.json','audit_native_split_map.py'),('held_data_structure.json','native_split_cdc.py')):
        r=bind(MAP_AUDIT/n);assert r['map_sha256']==mapped['map_sha256']
        verify({str(HERE/checker):r['auditor_sha256']})
    unit=ROOT/'tmp/frame-execution/startup-cause-shorts4';ur=bind(unit/'result.json');ui=bind(unit/'inputs.json')
    assert ur['status']=='STARTUP_CAUSE_TRACE_CELLS_SERIAL_ADAPTER_PASS' and len(ur['results'])==15
    assert ur['inputs_sha256']==sha(unit/'inputs.json') and ui['board_inputs_sha256']==sha(BASE/'inputs.json')
    verify({str(unit/n):h for n,h in ui['files'].items()});verify({str(unit/x['log']):x['sha256'] for x in ur['results']})
    assert all(x['passes'] for x in ur['results'])
    for folder,status in ((SHORTS/'observe','MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'),(ADAPTER,'ACTUAL_MAPPED_CONTROL_ADAPTER_IDLE_THREE_RESETS_PASS')):
        r=bind(folder/'result.json');i=bind(folder/'inputs.json')
        assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
        assert i.get('source_map_sha256',i.get('original_map_sha256'))==mapped['map_sha256']
        verify({str(folder/n):h for n,h in i['files'].items()})
        verify({str(folder/'simulation.log'):r.get('log_sha256',r.get('simulation_sha256'))})
    # Retain the complete serial/physical gate, with the actual wrapper lineage.
    import finish_local_decode_release as old
    code=inspect.getsource(old.extra_evidence)
    code='def inherited_physical(board,batch,physical,serial,bind,verify):\n'+code[code.index("    si=bind(serial/'inputs.json')"):]
    code=code.replace('from inline_official_state_luts import inline','from inline_fault_probe_luts import inline')
    code=code.replace("HERE/'inline_official_state_luts.py'","HERE/'inline_fault_probe_luts.py'")
    code=once(code,"len(details['instances'])==23","len(details['instances'])==proof['primitive_instances_removed']")
    for name in ('route','review'):
        code=once(code,"verify("+name+"['required_evidence'])","verify(archived_placement_evidence("+name+"['required_evidence'],physical,bind))")
    needle="    verify({str(driver):driver_meta['driver_sha256'],driver_meta['source']:driver_meta['source_sha256'],str(wrapper):driver_meta['wrapper_sha256']})"
    code=once(code,needle,"    wrapper=physical.parent/(physical.name+'-place-wrapper.py')\n"+needle)
    scope=dict(old.__dict__,archived_placement_evidence=archived_placement_evidence)
    exec(compile(code,__file__+':inherited-physical','exec'),scope)
    scope['inherited_physical'](board,batch,physical,serial,bind,verify)
    wrappers(physical,[('-place-wrapper',physical.parent/(physical.name+'-local-driver-wrapper.py')),
        ('-local-driver-wrapper',HERE/'local_driver_flow.py')],bind,verify)
    wrappers(batch,[('-local-driver-route',HERE/'route_local_driver_board.py')],bind,verify)
    inv=bind(batch/'gray-seed3/clock_crossing_inventory.json')
    assert inv['cell_probe_bundled_settling']['max_delay_ns']<13.332
    assert inv['cell_probe_structure']['map_sha256']==mapped['map_sha256']
    complete=bind(SHORTS/'result.json');assert complete['status']=='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert complete['map_sha256']==mapped['map_sha256'];verify(complete['evidence'])
    rebound=bind(physical/'endpoint_rebinding.json');origin=Path(rebound['source'])
    assert bind(physical/'physical_inputs.json')['endpoint_rebinding_sha256']==sha(physical/'endpoint_rebinding.json')
    verify(rebound['original']);verify({str(HERE/'prepare_local_driver_route.py'):rebound['script_sha256'],str(HERE/'rebind_native_placement.py'):rebound['translator_sha256']})
    from rebind_native_placement import translate
    expected,pins=translate((origin/'ti60f225_oob.sdc').read_text(),(origin/'outflow/ti60f225_oob.map.v').read_text(),(origin/'outflow/ti60f225_oob.out.placeloc').read_text())
    assert (physical/'ti60f225_oob.sdc').read_text()==expected and pins==rebound['pins']
    verify({str(physical/n):h for n,h in rebound['geometry'].items()})
    from audit_native_chain import audit as native_chain
    verify(native_chain()['required_evidence']);verify(warning_audit(board)['evidence'])
    for folder,status,checks in ((ROOT/'tmp/frame-execution/native-logic-split-fourstate128','LOGIC_SPLIT_128_FOURSTATE_65536_PASS',None),
                                  (FOUR,'ACTUAL_MAP_LOCAL_DRIVERS_FOURSTATE_PASS',75264)):
        r=bind(folder/'result.json');i=bind(folder/'inputs.json')
        assert r['status']==status and r['inputs_sha256']==sha(folder/'inputs.json')
        if checks:assert r['checks']==checks and i['map_sha256']==mapped['map_sha256']
        verify({str(folder/n):h for n,h in i['files'].items()});verify({str(folder/'simulation.log'):r['log_sha256']})
    for n in ('local_driver_board.py','local_driver_short_flow.py','review_local_driver_route.py','prepare_local_driver_route.py',
              'route_local_driver_board.py','finish_local_driver_release.py','native_bound_cdc.py','audit_native_split_gray.py'):
        verify({str(HERE/n):sha(HERE/n)})

def package_code():
    import finish_native_split_release as previous
    source,code=previous.package_code()
    code=once(code,'from native_split_board import audit_delta','from local_driver_board import audit_delta')
    code=code.replace('finish_native_split_release','finish_local_driver_release')
    code=once(code,"verify({str(HERE/'native_split_board.py'):mi['native_transform_sha256']})",
        "verify({str(HERE/'local_driver_board.py'):mi['local_driver_packager_sha256']})")
    code=once(code,'native_local_split_pairs=128,','native_local_split_pairs=128,local_driver_extra_luts=242,local_driver_extra_ffs=52,')
    code=code.replace('128处局部原语拆分的完整视频/A候选','局部原语拆分与驱动复制的完整视频/A候选')
    code=code.replace('功能RTL保持原版；','功能RTL保持原版，33条慢网增加已验证的同功能驱动，不增加流水周期；')
    return source,code

def finish_code():
    import finish_native_split_release as previous
    source,code=previous.finish_code()
    return source,code.replace('finish_native_split_release.py','finish_local_driver_release.py')

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
    driver=out.parent/(out.name+'-local-driver-release.py');assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*args];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
