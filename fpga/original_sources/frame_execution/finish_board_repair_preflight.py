"""Queue one-license simulation work behind the current strict DDR regression.

This prepares and checks a board candidate; it never starts an unbounded run,
downloads hardware, changes licenses, or upgrades resource/physical gates.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_ddr516_board import verified_io_bridge
from board_host_speedups import proof as verify_host_proof


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--ddr',type=Path,required=True);p.add_argument('--board',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--dpram-proof',type=Path)
    p.add_argument('--ram10-proof',type=Path)
    a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    ddr=a.ddr.resolve();board=a.board.resolve();control=a.control.resolve()
    inputs=dict(ddr=str(ddr),ddr_inputs_sha256=sha(ddr/'inputs.json'),board=str(board),
                board_inputs_sha256=sha(board/'inputs.json'),control=str(control),
                control_inputs_sha256=sha(control/'inputs.json'),runner_sha256=sha(__file__),
                queued_at=time.time(),full_board_run_allowed=False)
    inputs['reused_proofs']={name:dict(root=str(path.resolve()),result_sha256=sha(path/'result.json'))
        for name,path in [('dpram',a.dpram_proof),('ram10',a.ram10_proof)] if path}
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    print('PREFLIGHT_WAITING_FOR_EXISTING_DDR_RESULT',flush=True)
    deadline=time.monotonic()+3600
    while not (ddr/'result.json').exists():
        assert time.monotonic()<deadline,'DDR prerequisite wait timed out; no second Questa started'
        time.sleep(2)
    assert sha(ddr/'inputs.json')==inputs['ddr_inputs_sha256']
    results=[]
    def run(name,script,*args):
        command=[sys.executable,'-B',str(HERE/script),*map(str,args)]
        start=time.monotonic()
        print('PREFLIGHT_START '+name,flush=True)
        with(root/(name+'.log')).open('wb')as log:
            r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        row=dict(stage=name,returncode=r.returncode,seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log')))
        results.append(row);(root/'stages.json').write_text(json.dumps(results,indent=2))
        assert r.returncode==0,f'{name} failed; inspect {root/(name+".log")}'
        print('PREFLIGHT_PASS '+name+' '+str(round(row['seconds'],3)),flush=True)
    sim='E:/quatra/questa_fse/win64'
    # Host-only proofs are useful even if the independent DDR candidate fails.
    proof_roots={}
    for name,script in [('dpram','probe_dpram_word_sim.py'),('ram10','probe_ram10_word_sim.py'),
                        ('ring','probe_board_ring_scoreboard.py'),('micron_lookup','validate_micron_lookup.py')]:
        reused=getattr(a,name+'_proof',None)
        if reused:
            status='PUBLIC_DPRAM_WORD_EQUIVALENCE_PASS' if name=='dpram' else 'PUBLIC_RAM10_WORD_EQUIVALENCE_PASS'
            target,_,_=verify_host_proof(reused,status)
            assert sha(target/'result.json')==inputs['reused_proofs'][name]['result_sha256']
            proof_roots[name]=target
            print('PREFLIGHT_REUSED_VERIFIED_PROOF '+name,flush=True)
            continue
        target=root/name;proof_roots[name]=target
        run(name,script,'--root',target,'--sim-bin',sim)
    regression,proof=verified_io_bridge(ddr)
    assert sha(board/'inputs.json')==inputs['board_inputs_sha256']
    for row in json.loads((board/'inputs.json').read_text())['inputs']:
        assert sha(row['snapshot'])==row['sha256'],row['snapshot']
    for name in ('ddr516_axi_address_bridge.sv','ddr516_rx_retime.sv','ddr516_v4_controller.sv','ddr3_controller.bin','ddr3_controller.vh'):
        assert sha(board/name)==sha(regression/name),name
    assert sha(control/'ddr516_reference.sv')==sha(regression/'ddr516_reference.sv')
    run('pin_audit','audit_ddr516_turnaround_pins.py','--root',ddr)
    proof.update(board_inputs_sha256=inputs['board_inputs_sha256'],board=str(board))
    (root/'board_ddr_regression_proof.json').write_text(json.dumps(proof,indent=2))
    fast=root/'accelerated_board'
    # Equivalence alone does not justify an optimization. Keep the official
    # view when the actual Questa/O5 isolated trial did not show a benefit.
    host_options=[];host_selection={}
    for name,flag,key in [('dpram','--ram-word-proof','host_speed_ratio'),
                          ('ram10','--simple-ram-word-proof','host_speed_ratio'),
                          ('ring','--ring-checker-proof','isolated_host_speed_ratio')]:
        trial=json.loads((proof_roots[name]/'result.json').read_text())
        use=trial[key]>1.05
        host_selection[name]=dict(selected=use,isolated_host_speed_ratio=trial[key],whole_board_speed_measured=False)
        if use:host_options.extend([flag,proof_roots[name]])
    (root/'host_selection.json').write_text(json.dumps(host_selection,indent=2))
    run('prepare_accelerated','prepare_board_frame_run.py','--root',fast,'--base',board,
        '--flat-dsp-proof',ROOT/'tmp/frame-execution/dsp24-compact-board-proof1',
        '--camera-before-loader','--warmup-lines','4',*host_options)
    run('source_binding','audit_ddr516_board_inputs.py','--map',board,'--simulation',fast)
    # Equal initial 100 us checks startup, hierarchy and model integration, not
    # active CNN speed. It cannot authorize the project's one-hour exception.
    for name,target in [('control_startup',control),('accelerated_startup',fast)]:
        run(name,'run_board_frame.py','--root',target,'--duration-us','100','--optimize-level','5')
        status=json.loads((target/'run_result_100us.json').read_text())
        assert status['status']=='BOUNDED_SPEED_PROBE_NOT_FULL_FRAME' and status['ddr_model_error_total']==0
    result=dict(status='BOARD_REPAIR_AND_HOST_ACCEL_PREFLIGHT_PASS',stages=results,
                inputs_sha256=sha(root/'inputs.json'),accelerated_board=str(fast),
                accelerated_inputs_sha256=sha(fast/'inputs.json'),ddr_proof_sha256=sha(root/'board_ddr_regression_proof.json'),
                full_frame_pass=False,active_cnn_speed_measured=False,one_hour_exception_proven=False,
                host_selection=host_selection,
                resource_fit=False,board_download_allowed=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
