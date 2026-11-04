"""Complete selected-implementation gates plus the bounded continuous delta."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('finish_cnn_sparse_control_release.py')
body=SOURCE.read_text(encoding='utf8').replace('cnn_sparse_control','continuous_control').replace('cnn-sparse-control','continuous-control')
start=body.index("replacement=r'''")
end=body.index('injection="""',start)
original=SOURCE.read_text(encoding='utf8')
wstart=original.index('def warning_audit(board):')
wend=original.index("\n'''",wstart)
warnings=original[wstart:wend].replace('finish_cnn_observer_trim_free_release as parent','finish_cnn_sparse_control_release as parent').replace('from cnn_observer_trim_board import BOARD as old_board','from cnn_sparse_control_board import BOARD as old_board')
replacement=r'''
def payload_evidence(bind,verify):
    import finish_cnn_sparse_control_release as parent
    contract=parent.payload_evidence(bind,verify)
    from continuous_control_board_v2 import transform_check,audit_delta as current_audit
    proof=transform_check();bind(TRANSFORM/'result.json');verify(proof['inputs'])
    verify({str(TRANSFORM/'native_source.v'):proof['source_sha256'],str(TRANSFORM/'boundary.json'):proof['boundary_sha256']})
    assert proof['unchanged_primitives']==59191 and proof['protected_byte_identical']
    cut=bind(TRANSFORM/'expanded_cut_audit2.json');verify(cut['inputs'])
    assert cut['status']=='EXPANDED_NATIVE_CUT_PASS' and cut['crossing_loads']==407 and not cut['missing']
    mapped=bind(BOARD/'result.json');verify(mapped['evidence']);current_audit(BOARD)
    assert mapped['removed_dead_old_valid_luts']==[f'LUT__{i}' for i in range(159749,159754)]
    assert mapped['postmap_auditor_sha256']==sha(HERE/'continuous_control_board_v2.py')
    for name,status in (
        ('continuous-control-short2','CONTINUOUS_PREVIEW_CONTROL_SHORT_PASS'),
        ('preview-boundary-short7','ORIGINAL_MAPPED_CONTROLLER_EXPORT_MEANINGS_PASS'),
        ('preview-owner-short1','CONTINUOUS_OWNER_ADMISSION_DRAIN_PASS'),
        ('preview-mapped-control3','CONTINUOUS_ACTUAL_MAP_LOAD_VERIFY_REARM_PASS'),
        ('preview-mapped-admission1','ACTUAL_MAP_ADMISSION_ALL_BUDGETS_AND_SATURATION_PASS')):
        root=ROOT/'tmp/frame-execution'/name;r=bind(root/'result.json');i=bind(root/'inputs.json')
        assert r['status']==status and r['inputs_sha256']==sha(root/'inputs.json')
        verify(i.get('files',i))
        if 'map_sha256' in r:assert r['map_sha256']==mapped['map_sha256']
        if 'log_sha256' in r:
            log=root/'simulate.log'
            verify({str(log):r['log_sha256']})
        for stage in r.get('stages',[]):
            if 'markers' in stage:
                log=root/(stage['stage']+'.log');bind_log=log.read_text()
                assert all(marker in bind_log for marker in stage['markers'])
                verify({str(log):sha(log)})
        if (root/'simulate.log').exists():
            verify({str(root/'simulate.log'):sha(root/'simulate.log')})
    decoder=bind(ROOT/'tmp/board-test/continuous-progress-decoder1.json');verify(decoder['evidence'])
    assert decoder['status']=='CURRENT_MAPPED_PROGRESS_POLARITY_AND_SATURATION_PASS'
    assert decoder['map_sha256']==mapped['map_sha256']
    host=bind(ROOT/'tmp/board-test/continuous-run-host-tests1.json');verify(host['evidence'])
    assert host['status']=='BOUNDED_CONTINUOUS_HOST_POLICY_PASS' and host['tests']==13
    recovery=bind(ROOT/'tmp/frame-execution/continuous-control-evidence1/recovery.json');verify(recovery['evidence'])
    return dict(contract,hardware_frame_limit=512,frame_budget_bits=10,boot_run_watchdog_cycles=9000000000,
        explicit_success_rearm=True,full_reload_on_rearm=True,diagnostic_admitted='saturated31_not_full_count',
        progress_schema='D-continuous-progress-v1',single_frame_cnn_deadline_cycles=33554432,
        actual_continuous_fps_validated=False,screen_frame_validated=False)

'''
body=body[:start]+'replacement='+repr(replacement+warnings+'\n')+'\n'+body[end:]
body=body.replace("'validate_continuous_control_drivers.py'","'validate_preview_mapped_control.py','validate_preview_admission_mapped.py','audit_continuous_vectors.py','validate_continuous_adapter.py','read_continuous_progress.py','continuous_control_board_v2.py'")
# The final package's reader and schema must agree with the new gate contract.
marker='body=once(body,"code=code_for(\'minimal\')",injection+"\\ncode=code_for(\'minimal\')")'
profile_patch="\ntail=tail.replace('cnn-observer-live-levels-only-v1','cnn-continuous-progress-v1').replace('read_cnn_observer_trim_snapshot','read_continuous_snapshot')\n"
body=once(body,marker,'injection += '+repr(profile_patch)+'\n'+marker)
exec(compile(body,str(SOURCE)+':continuous-full-release','exec'),globals())
