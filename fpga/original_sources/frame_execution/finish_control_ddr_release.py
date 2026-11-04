"""Finish the independently bound combined revision, offline only."""
import argparse,json,sys,runpy
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_ddr_combined import audit_delta,once

def execute(source,code,driver,args):
    assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        generated_sha256=sha(driver),adapter_sha256=sha(__file__)),indent=2))
    sys.argv=[str(driver),*map(str,args)];runpy.run_path(str(driver),run_name='__main__')

def main():
    mode=sys.argv.pop(1)
    if mode=='pll':
        import audit_pll_shift_routed as check
        check.audit_delta=audit_delta;check.main();return
    if mode=='bundle':
        args=sys.argv[1:];out=Path(args[args.index('--output')+1]).resolve()
        source=HERE/'prepare_command_progress_release.py';code=source.read_text()
        pairs=[('from control_command_repair import BASE,audit_delta','from control_ddr_combined import BASE,audit_delta'),
            ('output/deployment/board-full-150-ddr-shift-guard-20261010','output/deployment/board-full-150-command-progress-20261010'),
            ('tmp/frame-execution/command-progress-short2','tmp/frame-execution/control-ddr-combined-rtl1'),
            ("'COMMAND_PROGRESS_RTL_SERIAL_RESET_PASS'","'CONTROL_DDR_RTL_SERIAL_EVENTS_RESET_PASS'"),
            ("'REAL_MAPPED_PROGRESS_COUNTERS_PASS'","'CONTROL_DDR_MAPPED_SERIAL_IDLE_RESET_PASS'"),
            ("for name in ('read_control_progress','read_command_progress_status'):","for name in ('read_control_ddr_progress','read_control_ddr_status'):"),
            ("name=='read_command_progress_status'","name=='read_control_ddr_status'"),
            ("'test_stable_jtag_reply','-v'","'test_stable_jtag_reply','test_control_ddr_host','-v'"),
            ("'Ran 18 tests'","'Ran 24 tests'")]
        for before,after in pairs:code=once(code,before,after)
        before="    out.mkdir(parents=True);shutil.copy2(bit,out/bit.name)"
        after="""    from run_control_ddr_physical import audit_comb4
    combined=audit_comb4(board);assert combined==bind(board/'combined_map_audit.json')
    observe=ROOT/'tmp/frame-execution/control-ddr-combined-observe1'
    observed=bind(observe/'result.json');observed_inputs=bind(observe/'inputs.json')
    assert observed['status']=='MAPPED_DDR_STATUS_AND_EVENTS_CONTROL_RESET_PASS'
    assert observed['inputs_sha256']==sha(observe/'inputs.json')
    assert observed_inputs['source_map_sha256']==mapped['map_sha256']
    verify({str(observe/n):h for n,h in observed_inputs['files'].items()})
    verify({str(observe/'simulation.log'):observed['log_sha256']})
    reused=bind(batch/'short_evidence_reuse.json');verify(reused)
    for name in ('control_ddr_combined.py','read_control_ddr_progress.py','read_control_ddr_status.py','validate_control_ddr.py','validate_control_ddr_observation.py','test_control_ddr_host.py','control_ddr_clock_profile.py','probe_control_ddr_physical.py','run_control_ddr_physical.py','finish_control_ddr_release.py'):
        evidence[str(HERE/name)]=sha(HERE/name)
    ancestry=batch
    if (batch/'route_continuation.json').exists():
        continuation=bind(batch/'route_continuation.json')
        ancestry=Path(continuation['source_batch'])
        verify({str(ancestry/'inputs.json'):continuation['source_inputs_sha256'],str(batch/'route_result.json'):continuation['route_result_sha256'],str(batch/'validation_runner.py'):terminal['validation_runner_sha256']})
        rr=bind(batch/'route_result.json');assert rr['status']=='ROUTE_PASS_PENDING_STA' and rr['exit_code']==0
        assert rr['source_and_copy_frozen_hashes_pass'] and Path(rr['source'])==ancestry
        verify({str(physical/'outflow/ti60f225_oob.route.out'):rr['route_sha256']})
        verify({str(physical/n):h for n,h in rr['frozen'].items()})
    for driver in (ancestry.parent/(ancestry.name+'-combined-driver.py'),serial.parent/(serial.name+'-combined-driver.py')):
        evidence[str(driver)]=sha(driver);evidence[str(driver.with_suffix('.json'))]=sha(driver.with_suffix('.json'))
    recipe=ancestry.parent/(ancestry.name+'-recipe-runner.json')
    if recipe.exists():
        recipe_data=bind(recipe);verify(recipe_data['required_evidence'])
    dispatch=ancestry.parent/(ancestry.name+'-recipe-dispatch.py')
    if dispatch.exists():
        dm=bind(dispatch.with_suffix('.json'))
        verify({str(dispatch):dm['generated_sha256'],dm['source']:dm['source_sha256'],str(HERE/'run_control_ddr_congestion1.py'):dm['adapter_sha256']})
    if (physical/'reuse_placement.json').exists():
        from fixed_control_ddr_reuse_locations import verify_placed
        reuse_places=bind(physical/'reuse_placement.json')
        assert verify_placed(physical)==bind(physical/'reuse_actual_locations.json')
        assert reuse_places['new_map_sha256']==mapped['map_sha256']
        ref=Path(reuse_places['reference']);ref_selected=canonical_audit(ref)
        lut_only=reuse_places['status']=='SAME_MAP_LUT_ONLY_REPAIR_NOT_TIMING_PASS'
        if lut_only:
            # This is only the starting geometry. The selected final route
            # has already been independently required to pass above.
            assert reuse_places['old_map_sha256']==reuse_places['new_map_sha256']
            from fixed_control_ddr_lut_locations import plan as local_plan
            assert local_plan(board)=={k:v for k,v in reuse_places.items() if k!='assignments_sha256'}
            verify({reuse_places['rejected_proof']:reuse_places['rejected_proof_sha256'],
                str(Path(reuse_places['survey'])/'result.json'):reuse_places['survey_result_sha256'],
                str(Path(reuse_places['survey'])/'outflow/canonical_setup.rpt'):reuse_places['survey_report_sha256']})
            evidence[str(HERE/'fixed_control_ddr_reuse_locations.py')]=sha(HERE/'fixed_control_ddr_reuse_locations.py')
        else:
            assert ref_selected['current_timing_pass']
        ref_physical=Path(ref_selected['physical'])
        extra_pairs='additional_register_driver_pairs' in reuse_places
        placement_module='fixed_control_ddr_lut_locations.py' if lut_only else 'fixed_control_ddr_control_locations.py' if extra_pairs else 'fixed_control_ddr_reuse_locations.py'
        verify({str(ref/'result.json'):reuse_places['reference_result_sha256'],
            str(ref_physical/'outflow/ti60f225_oob.map.v'):reuse_places['old_map_sha256'],
            str(ref_physical/'outflow/ti60f225_oob.out.placeloc'):reuse_places['old_placement_sha256'],
            str(physical/'fixed_dsp.placeloc'):reuse_places['assignments_sha256'],
            str(physical/'reuse_placement_generator.py'):reuse_places['generator_sha256'],
            str(HERE/placement_module):reuse_places['generator_sha256']})
        run_wrapper=HERE/'run_control_ddr_reuse_physical.py'
        probe_wrapper=HERE/'probe_control_ddr_reuse_physical.py'
        if extra_pairs or lut_only:
            if extra_pairs:
                verify({str(HERE/'fixed_control_ddr_reuse_locations.py'):reuse_places['base_generator_sha256']})
            variant='lut' if lut_only else 'control'
            run_wrapper=ancestry.parent/(ancestry.name+'-'+variant+'-dispatch-wrapper.py')
            probe_wrapper=physical.parent/(physical.name+'-'+variant+'-probe-wrapper.py')
            for wrapped,adapter in ((run_wrapper,'run_control_ddr_'+variant+'_physical.py'),(probe_wrapper,'probe_control_ddr_'+variant+'_physical.py')):
                wm=bind(wrapped.with_suffix('.json'))
                verify({str(wrapped):wm['generated_sha256'],wm['source']:wm['source_sha256'],str(HERE/adapter):wm['adapter_sha256']})
        dispatch=ancestry.parent/(ancestry.name+'-reuse-dispatch.py');dm=bind(dispatch.with_suffix('.json'))
        verify({str(dispatch):dm['generated_sha256'],dm['source']:dm['source_sha256'],
            str(run_wrapper):dm['wrapper_sha256'],
            str(HERE/('probe_control_ddr_lut_physical.py' if lut_only else 'probe_control_ddr_control_physical.py' if extra_pairs else 'probe_control_ddr_reuse_physical.py')):dm['probe_sha256'],
            str(HERE/placement_module):dm['placement_sha256']})
        probe=physical.parent/(physical.name+'-reuse-probe.py');pm=bind(probe.with_suffix('.json'))
        verify({str(probe):pm['generated_sha256'],pm['source']:pm['source_sha256'],
            str(probe_wrapper):pm['wrapper_sha256']})
    out.mkdir(parents=True);shutil.copy2(bit,out/bit.name)"""
        code=once(code,before,after)
        code=code.replace('OFP2','OFP3').replace('386位','450位')
        code=once(code,'命令两级同步器强制映射为两个真实FF；低192位首错记录保留，新增128位当前状态与命令计数。',
            '关闭COMB4合并；保留两级命令FF、192位首错和128位控制进度，增加DDR公开原始状态及采样事件。')
        code=code.replace('read_control_progress.py只读','read_control_ddr_progress.py只读').replace('read_command_progress_status.py读','read_control_ddr_status.py读')
        execute(source,code,out.parent/(out.name+'-package-driver.py'),args);return
    assert mode=='finish'
    args=sys.argv[1:];root=Path(args[args.index('--root')+1]).resolve()
    source=HERE/'finish_command_progress_release.py';code=source.read_text()
    code=once(code,"'audit_command_progress_pll.py',['--board'","'finish_control_ddr_release.py',['pll','--board'")
    code=once(code,"'prepare_command_progress_release.py',['--board'","'finish_control_ddr_release.py',['bundle','--board'")
    execute(source,code,root.parent/(root.name+'-combined-driver.py'),args)
if __name__=='__main__':main()
