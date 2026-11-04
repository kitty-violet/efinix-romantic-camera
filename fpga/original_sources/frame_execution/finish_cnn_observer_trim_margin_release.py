"""Full system release plus proved placement-only target and restored SDC."""
from pathlib import Path
from control_command_repair import once
source=Path(__file__).with_name('finish_cnn_observer_trim_t1_release.py')
body=source.read_text()
body=body.replace('finish_cnn_observer_trim_t1_release','finish_cnn_observer_trim_margin_release')
body=body.replace('cnn_observer_trim_t1_route','cnn_observer_trim_margin_route')
body=body.replace("after='TIMING_1'","after='CONGESTION_2'")
anchor="body=body.replace('cnn_observer_trim_free_route','cnn_observer_trim_margin_route')"
body=once(body,anchor,anchor+"\nbody=body.replace('cnn_observer_trim_place.py','cnn_observer_margin_place.py')")
anchor='    for stage in stages:verify({str(canonical/(stage[\'name\']+\'.log\')):stage[\'log_sha256\']})'
extra='''
    from cnn_observer_process_margin import EXTRA,PROBE,verify_probe
    proof=verify_probe();verify(proof['evidence']);bind(PROBE/'result.json')
    pm=bind(physical/'physical_inputs.json')
    origin=Path(pm['placement_origin'])
    receipt=bind(origin/'placement_optimization_inputs.json')
    assert receipt['status']=='ORIGINAL_SDC_RESTORED_AFTER_PLACEMENT'
    assert receipt['extra']==EXTRA and not receipt['hardware_clock_changed']
    assert receipt['ordinary_anchors']==0 and receipt['full_original_sta_required']
    assert pm['placement_optimization_inputs_sha256']==sha(origin/'placement_optimization_inputs.json')
    assert sha(physical/'placement_optimization_inputs.json')==sha(origin/'placement_optimization_inputs.json')
    acceptance=origin/'placement_acceptance.sdc';optimization=origin/'placement_optimization.sdc'
    verify({str(acceptance):receipt['canonical_sha256'],str(optimization):receipt['optimization_sha256'],
            str(HERE/'cnn_observer_margin_place.py'):receipt['adapter_sha256'],receipt['proof']:receipt['proof_sha256']})
    assert (origin/'ti60f225_oob.sdc').read_bytes()==acceptance.read_bytes()
    assert optimization.read_text()==acceptance.read_text()+EXTRA
    restore=bind(ROOT/'tmp/board-test/cnn-observer-margin-restore2/result.json')
    assert restore['status']=='NORMAL_AND_EXCEPTION_SDC_BYTE_RESTORATION_PASS' and restore['cases']==2
    verify(restore['evidence'])
'''
# Insert into the generated full extra_evidence function, before package code.
body=once(body,anchor,anchor+extra)
exec(compile(body,str(source)+':margin-placement-full-release','exec'),globals())
