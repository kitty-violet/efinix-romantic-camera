"""Release the locally split native candidate with unchanged full RTL evidence.

All original resource/timing/CDC/pin/PLL/host/bitstream gates remain. The native
reimport lineage, four-state proof and warning sensitivity are additional gates.
This tool never accesses hardware.
"""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from native_split_board import BASE,audit_delta
from finish_split_logic_outputs_release import archived_placement_evidence
from audit_native_chain import audit as native_chain

BOARD=ROOT/'tmp/frame-execution/connected-board-native-split1'
PARENT=ROOT/'output/deployment/board-full-150-startup-cause-20261011'
SHORTS=ROOT/'tmp/frame-execution/native-split-evidence1'
RTL=SHORTS/'rtl'
SERIAL=ROOT/'tmp/frame-execution/native-split-evidence1-serial/new'
MAP_AUDIT=SHORTS/'map-audit'
ADAPTER=ROOT/'tmp/frame-execution/native-split-evidence1-adapter'
STATUS='LOCAL_DECODE_ROUTED_ORIGINAL_STA_GRAY_PASS_PENDING_RELEASE_AUDITS'

def parent_evidence(parent,verify):
    assert Path(parent['board'])==BASE
    verify(parent['required_evidence']);audit_delta(BOARD)
    return dict(status='ORIGINAL_RTL_PARENT_AND_NATIVE_SPLIT_BOUND',
        manifest=str(PARENT/'manifest.json'),sha256=sha(PARENT/'manifest.json'),
        scope='Original full video/A RTL, pins, clocks and assets retained. Native primitive replacement has separate current proofs. No inherited board PASS.')

def warning_audit(board):
    import re
    probe=ROOT/'tmp/frame-execution/native-all-open-inputs1'
    r=json.loads((probe/'result.json').read_text());i=json.loads((probe/'inputs.json').read_text())
    assert r['status']=='EXPORTED_OPEN_INPUTS_INSENSITIVE_PRIVATE_VENDOR_WARNINGS_RETAINED'
    assert r['inputs_sha256']==sha(probe/'inputs.json')
    assert i['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    original=board/'outflow/ti60f225_oob.warn.log';body=original.read_text()
    assert sha(original)==i['warning_sha256']
    assert '[EFX-0654' not in body
    names=re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",body)
    assert names==['Open_'+str(n) for n in range(8,18)]
    evidence={**r['evidence'],str(probe/'result.json'):sha(probe/'result.json'),
        str(probe/'inputs.json'):sha(probe/'inputs.json'),str(original):sha(original),
        str(probe/'public_connectivity_audit.json'):r['public_audit_sha256'],
        **{str(probe/n):h for n,h in i['files'].items()}}
    driver=probe.parent/(probe.name+'-driver.py');dm=json.loads(driver.with_suffix('.json').read_text())
    evidence.update({str(driver):dm['driver_sha256'],str(driver.with_suffix('.json')):sha(driver.with_suffix('.json')),
        dm['source']:dm['source_sha256'],str(HERE/'probe_native_all_open_inputs.py'):dm['wrapper_sha256']})
    assert r['private_vendor_warnings'] and r['private_warning_list_may_be_capped']
    for path,h in evidence.items():assert sha(path)==h,path
    return dict(status='ALL_EXPORTED_DISCONNECTED_INPUT_GROUPS_CHECKED',
        evidence=evidence,visible_warning_count=10,frontend_warning_groups=r['frontend_warning_groups'],
        checked_input_ports=r['input_ports'],checked_bits=r['tied_bits'],loops=0,
        private_vendor_warnings_retained=r['private_vendor_warnings'],ddr_calibration_proven=False)

def extra_evidence(board,batch,physical,serial,bind,verify):
    # Keep the existing complete actual-map/serial/physical evidence checks,
    # remove only proofs of datapath edits this candidate does not contain.
    import finish_registered_boundary_release as old
    code=inspect.getsource(old.extra_evidence)
    code=code[:code.index('    alias=bind(SERIAL.parent.parent')]
    changes={
        "'audit_split_logic_outputs_map.py'":"'audit_native_split_map.py'",
        "'split_logic_outputs_cdc.py'":"'native_split_cdc.py'",
        "'tmp/frame-execution/split-outputs-shorts2'":"'tmp/frame-execution/startup-cause-shorts4'",
        "sha(DATAPATH_PARENT/'inputs.json')":"sha(BASE/'inputs.json')",
        "str(HERE/'registered_boundary_route.py')":"str(batch.parent/(batch.name+'-native-wrapper.py'))",
        "str(HERE/'place_registered_boundary_board.py')":"str(physical.parent/(physical.name+'-native-wrapper.py'))",
        "'REGISTERED_BOUNDARY_SAME_MAP_SHORT_INTEGRATION_PASS'":"'NATIVE_SPLIT_SAME_MAP_SHORT_INTEGRATION_PASS'"
    }
    for before,after in changes.items():code=once(code,before,after)
    scope=dict(globals());exec(compile(code,__file__+':reused-complete-gates','exec'),scope)
    scope['extra_evidence'](board,batch,physical,serial,bind,verify)
    for root,wrapper in ((physical,'place_native_split_board.py'),(batch,'route_native_split_board.py')):
        driver=root.parent/(root.name+'-native-wrapper.py');m=bind(driver.with_suffix('.json'))
        verify({str(driver):m['driver_sha256'],m['source']:m['source_sha256'],str(HERE/wrapper):m['wrapper_sha256']})
    verify(native_chain()['required_evidence']);verify(warning_audit(board)['evidence'])
    from rebind_native_placement import audit as rebind_audit
    rebound=rebind_audit(physical);bind(physical/'endpoint_rebinding.json')
    assert bind(physical/'physical_inputs.json')['endpoint_rebinding_sha256']==sha(physical/'endpoint_rebinding.json')
    verify(rebound['original']);verify({str(HERE/'rebind_native_placement.py'):rebound['script_sha256']})
    proof=ROOT/'tmp/frame-execution/native-logic-split-fourstate128'
    r=bind(proof/'result.json');i=bind(proof/'inputs.json')
    assert r['status']=='LOGIC_SPLIT_128_FOURSTATE_65536_PASS' and r['inputs_sha256']==sha(proof/'inputs.json')
    verify({str(proof/n):h for n,h in i['files'].items()})
    verify({str(proof/'simulation.log'):r['log_sha256'],i['manifest']:i['manifest_sha256'],
            str(HERE/'validate_native_logic_split.py'):i['runner_sha256']})
    for n in ('native_split_board.py','native_split_flow.py','native_split_cdc.py','audit_native_split_map.py',
              'audit_native_split_gray.py','audit_native_chain.py','audit_native_roundtrip.py','native_logic_split.py',
              'run_native_split_evidence.py','finish_native_split_release.py','probe_native_open_ports.py',
              'probe_native_open_inputs.py','native_bound_cdc.py'):
        verify({str(HERE/n):sha(HERE/n)})

def package_code():
    import finish_registered_boundary_release as old
    source,code=old.package_code()
    code=once(code,'from registered_boundary_board import audit_delta','from native_split_board import audit_delta')
    code=code.replace('finish_registered_boundary_release','finish_native_split_release')
    code=once(code,'extra_evidence,parent_evidence\n','extra_evidence,parent_evidence,warning_audit\n')
    code=once(code,"verify({str(HERE/'prepare_split_registered_board.py'):mi['registered_boundary_transform_sha256']})",
        "verify({str(HERE/'native_split_board.py'):mi['native_transform_sha256']})")
    code=once(code,'from split_logic_outputs_cdc import inventory','from native_bound_cdc import inventory')
    code=once(code," and not mapped['comb_loop_warning'] and not mapped['new_undriven_signatures']",
        "\n    verified_warnings=warning_audit(board);verify(verified_warnings['evidence'])")
    code=once(code,"functional_evidence_scope='Parent evidence retained; modified datapaths separately checked, no inherited throughput claim'",
        "functional_evidence_scope='Original full RTL retained; native remap and current interfaces separately checked, no inherited board PASS'")
    code=once(code,'registered_datapath_boundaries=True,independent_outputs_workaround=True,',
        'native_local_split_pairs=128,independent_outputs_workaround=True,native_warning_audit=verified_warnings,')
    code=code.replace('零COMB4与RQ/source寄存边界的完整视频/A候选','128处局部原语拆分的完整视频/A候选')
    code=code.replace('使用独立LUT输出，RQ反馈/输出及source增加已验证寄存边界；',
        '仅将128处LOGIC双输出原语转换为独立LUT，并由官方重新生成VDB；功能RTL保持原版；')
    return source,code

def finish_code():
    import finish_registered_boundary_release as old
    source,code=old.finish_code()
    return source,code.replace('finish_registered_boundary_release.py','finish_native_split_release.py')

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
