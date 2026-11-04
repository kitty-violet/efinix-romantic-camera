"""Full observer release gates plus exact sparse-control delta and short proof."""
from pathlib import Path
from control_command_repair import once
source=Path(__file__).with_name('finish_cnn_observer_trim_free_release.py')
body=source.read_text(encoding='utf8')

replacement=r'''
def payload_evidence(bind,verify):
    import finish_cnn_observer_trim_free_release as parent
    contract=parent.payload_evidence(bind,verify)
    from cnn_sparse_control_board import transform_check
    proof=transform_check();bind(TRANSFORM/'result.json');verify(proof['inputs'])
    verify({str(TRANSFORM/'native_source.v'):proof['source_sha256'],str(TRANSFORM/'replications.json'):proof['replications_sha256']})
    assert proof['added_kinds']==dict(EFX_FF=3,EFX_LUT4=27)
    assert proof['identical_clock_reset_ce_and_init_parameters'] and proof['protected_blocks_byte_equal']
    assert proof['latency_cycles_added']==0 and proof['selected_nets']==13
    mapped=bind(BOARD/'result.json');verify(mapped['evidence']);audit_delta(BOARD)
    root=ROOT/'tmp/frame-execution/cnn-sparse-control-drivers1'
    r=bind(root/'result.json');i=bind(root/'inputs.json')
    assert r['status']=='ACTUAL_MAP_LOCAL_DRIVERS_FOURSTATE_PASS' and r['checks']==30*256
    assert r['inputs_sha256']==sha(root/'inputs.json')
    assert i['map_sha256']==mapped['map_sha256'] and i['map_result_sha256']==sha(BOARD/'result.json')
    assert i['replications_sha256']==sha(BOARD/'replications.json')
    verify({str(root/f):h for f,h in i['files'].items()})
    verify({str(root/'simulation.log'):r['log_sha256'],str(HERE/'validate_cnn_sparse_control_drivers.py'):i['runner_sha256'],str(HERE/'validate_native_local_drivers.py'):i['base_runner_sha256']})
    return contract

def warning_audit(board):
    import finish_cnn_observer_trim_free_release as parent
    from cnn_observer_trim_board import BOARD as old_board
    assert board==BOARD;audit_delta(board)
    result=parent.warning_audit(old_board)
    a=(old_board/'outflow/ti60f225_oob.warn.log').read_text()
    b=(board/'outflow/ti60f225_oob.warn.log').read_text()
    assert '[EFX-0654' not in b
    for marker in ('[EFX-0200 WARNING]','[EFX-0201 WARNING]'):assert a.count(marker)==b.count(marker)
    from probe_native_open_ports import protected_pins
    from audit_native_roundtrip import tokens
    from native_local_drivers import graph
    undriven=lambda s:re.findall(r"\[EFX-0201 WARNING\] Re-wiring to GND non-driven net/port '([^']+)'",s)
    for root,log in ((old_board,a),(board,b)):
        text=(root/'native_source.v').read_text();names=undriven(log)
        samples=tokens(protected_pins(text)['u_ddr3_top/samples'])
        assert len(set(names))==len(names)==10 and len(samples)==40
        bits=sorted(39-samples.index(n) for n in names)
        assert bits==[12,13,14,15,32,33,36,37,38,39]
        assert all(len(re.findall(r'\b'+re.escape(n)+r'\b',text))==1 for n in names)
        mapped=tokens(protected_pins((root/'outflow/ti60f225_oob.map.v').read_text())['u_ddr3_top/samples'])
        assert len(mapped)==40 and all(re.fullmatch(r'Open_\d+',mapped[39-bit]) for bit in bits)
    dangling=lambda s:set(re.findall(r'Net (.*?) is dangling',s))
    assert dangling(a)==dangling(b)
    result.update(status='SPARSE_CONTROL_IDENTICAL_WARNING_CLASSES_AND_PROTECTED_OPEN_BITS_PASS',new_functional_unconnected_inputs=False)
    result['evidence'].update({str(f):sha(f) for f in (old_board/'outflow/ti60f225_oob.warn.log',board/'outflow/ti60f225_oob.warn.log',board/'public_connectivity_audit.json',Path(__file__))})
    return result

'''
injection="""
begin=tail.index('def payload_evidence(bind,verify):')
end=tail.index('_observer_extra_parent=extra_evidence',begin)
tail=tail[:begin]+REPLACEMENT+tail[end:]
tail=once(tail,"    proof=bind(TRANSFORM/'result.json');verify(proof['inputs'])",'''    proof=bind(TRANSFORM/'result.json');verify(proof['inputs'])
    for name in ('cnn_sparse_control_native.py','cnn_sparse_control_board.py','cnn_sparse_control_flow.py','cnn_sparse_control_place.py','cnn_sparse_control_implementation.py','cnn_sparse_control_route.py','validate_cnn_sparse_control_drivers.py'):
        verify({str(HERE/name):sha(HERE/name)})
    canonical=batch/'canonical-seed3';cm=bind(canonical/'inputs.json')
    verify({str(HERE/'canonical_snapshot_sta_isolated.py'):cm['runner_sha256'],str(HERE/'canonical_snapshot_sta.py'):cm['canonical_source_sha256']})
    assert cm['process_isolation']=='timing_then_checks_both_required'
    stages=bind(canonical/'isolated_stages.json')
    assert [s['name'] for s in stages]==['sta','checks'] and all(s['returncode']==0 for s in stages)
    for stage in stages:verify({str(canonical/(stage['name']+'.log')):stage['log_sha256']})''')
"""
body=once(body,"code=code_for('minimal')",injection+"\ncode=code_for('minimal')")
marker='inner="exec(compile(code,str(SOURCE)+\':joint-closure-release\',\'exec\'),globals())"'
switches="""
for old,new in (
 ('finish_cnn_observer_trim_free_release','finish_cnn_sparse_control_release'),
 ('cnn_observer_trim_free_route','cnn_sparse_control_route'),
 ('cnn_observer_trim_board','cnn_sparse_control_board'),
 ('cnn_observer_trim_implementation','cnn_sparse_control_implementation'),
 ('cnn_observer_trim_place.py','cnn_sparse_control_place.py'),
 ('cnn_observer_trim_flow','cnn_sparse_control_flow'),
 ('cnn-observer-trim-evidence3','cnn-sparse-control-evidence1')):
    assert old in code,old
    code=code.replace(old,new)
"""
body=once(body,marker,switches+'\n'+marker)
REPLACEMENT=replacement
exec(compile(body,str(source)+':sparse-control-full-release','exec'),globals())
