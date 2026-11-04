"""Continue a passed vendor partition gate into frozen whole-board measurement.

No simulator is launched until the supplied completed gate is verified. This
is a sequential job runner (not an agent); it never retries a failed stage.
Every stop uses the project's existing architecture-review memory workflow.
"""
import argparse
import copy
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
from run_units import ROOT,HERE,sha
from production_board_binding import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--vendor',type=Path,required=True);p.add_argument('--board',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--reuse-preflight',type=Path);a=p.parse_args()
    root=a.root.resolve();vendor=a.vendor.resolve();board=a.board.resolve();report=a.report.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    assert board.is_relative_to(ROOT/'tmp') and (board.is_dir() if a.reuse_preflight else not board.exists())
    assert report.is_relative_to(ROOT/'output') and not report.exists()
    proof=json.loads((vendor/'result.json').read_text())
    assert proof['status']=='PRODUCTION_DPI_REAL_DDR_FOUR_MASTER_EQUIVALENCE_PASS'
    assert proof['log_sha256']==sha(vendor/'simulation.log')
    root.mkdir();stages=[]
    base=ROOT/'tmp/frame-execution/board-repair-preflight2/accelerated_board'
    native=ROOT/'tmp/frame-execution/production-cosim2'
    mapped=ROOT/'tmp/frame-execution/connected-board-native-lock1'
    scripts=['prepare_production_board_run.py','production_board_binding.py','audit_ddr516_board_inputs.py',
             'run_board_frame.py','finish_board_measurement.py','publish_board_measurement.py',
             'summarize_board_measurement.py','summarize_board_traffic.py','summarize_board_command_profile.py',
             'probe_board_dpi_cost.py','owned_simulator_process.py','host_runtime_budget.py','run_units.py']
    script_hashes={name:sha(HERE/name) for name in scripts}
    input_data=dict(vendor=str(vendor),vendor_proof_sha256=sha(vendor/'result.json'),board=str(board),
                    report=str(report),runner_sha256=sha(__file__),scripts=script_hashes,hardware_modified=False)
    (root/'inputs.json').write_text(json.dumps(input_data,indent=2))
    def checkpoint(stage,summary,next_step):
        data=dict(stage=stage,summary=summary,next=next_step,
                  facts=[summary],pending=[],resources=[str(root/'stages.json'),str(board),str(report)])
        path=root/('checkpoint-'+str(len(stages))+'.json');path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
        # Use the same installed PowerShell 7 host as the approved project
        # shell. Do not change execution policies or fall back to Windows PS5.
        host=shutil.which('pwsh.exe');assert host,'PowerShell 7 host unavailable'
        subprocess.run([host,'-NoProfile','-File',str(ROOT/'tmp/checkpoint-active-work.ps1'),'-DataPath',str(path)],check=True,cwd=ROOT)
    def execute(name,script,*arguments):
        for file,expected in script_hashes.items():assert sha(HERE/file)==expected,file
        begin=time.monotonic();path=root/(name+'.log')
        item=dict(name=name,status='running',log=str(path));stages.append(item)
        (root/'stages.json').write_text(json.dumps(stages,indent=2));print('STAGE_BEGIN '+name,flush=True)
        with path.open('wb') as stream:
            result=subprocess.run([sys.executable,'-B',str(HERE/script),*map(str,arguments)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        item.update(status='pass' if result.returncode==0 else 'fail',returncode=result.returncode,
                    seconds=time.monotonic()-begin,log_sha256=sha(path))
        (root/'stages.json').write_text(json.dumps(stages,indent=2));print('STAGE_END '+json.dumps(item),flush=True)
        assert result.returncode==0,(name,path)
    try:
        if a.reuse_preflight:
            prior=a.reuse_preflight.resolve();old=json.loads((prior/'inputs.json').read_text())
            assert Path(old['board'])==board and old['vendor_proof_sha256']==sha(vendor/'result.json')
            assert old['scripts']==script_hashes
            for row in json.loads((prior/'stages.json').read_text()):
                assert row['status']=='pass' and row['log_sha256']==sha(row['log'])
            startup=json.loads((board/'run_result_100us.json').read_text())
            assert startup['status']=='BOUNDED_SPEED_PROBE_NOT_FULL_FRAME' and startup['returncode']==0
            assert not startup['failures'] and not startup['ddr_model_error_total']
            assert startup['inputs_sha256']==sha(board/'inputs.json')
            assert startup['log_sha256']==sha(board/'simulation_100us.log')
            assert not (board/'run_result.json').exists()
            stages.append(dict(name='reuse_preflight',status='pass',prior=str(prior),
                               prior_inputs_sha256=sha(prior/'inputs.json'),board_inputs_sha256=sha(board/'inputs.json')))
            (root/'stages.json').write_text(json.dumps(stages,indent=2))
        else:
            execute('prepare','prepare_production_board_run.py','--root',board,'--base',base,'--native',native,
                '--boundary',ROOT/'tmp/frame-execution/production-cosim-boundary4',
                '--active',ROOT/'tmp/frame-execution/production-cosim-active2',
                '--unknown',ROOT/'tmp/frame-execution/production-cosim-unknown1','--vendor',vendor)
        execute('source_binding','audit_ddr516_board_inputs.py','--map',mapped,'--simulation',board)
        meta=json.loads((board/'inputs.json').read_text());validate(board,meta)
        # Tampering and accidentally compiling both CNN implementations must
        # be rejected before a long run can consume the manifest.
        rejected=[]
        for name in ('dll','proof','duplicate_cnn'):
            altered=copy.deepcopy(meta)
            if name=='dll':altered['native_dll_sha256']='0'*64
            elif name=='proof':altered['partition_proofs']['vendor']['sha256']='0'*64
            else:altered['source_files'].append(str(board/'assembly/cnn_production_frame.sv'))
            try:validate(board,altered)
            except AssertionError:rejected.append(name)
            else:raise AssertionError('Invalid partition accepted: '+name)
        (root/'binding_negative_tests.json').write_text(json.dumps(dict(status='PRODUCTION_BINDING_TAMPER_REJECTION_PASS',
             cases=rejected,board_inputs_sha256=sha(board/'inputs.json')),indent=2))
        if not a.reuse_preflight:
            execute('short_startup','run_board_frame.py','--root',board,'--duration-us','100','--optimize-level','5')
        checkpoint('真实DDR并发与联合边界通过，启动冻结整帧测量',
                   '2026-10-03 真实DDR四主短集成、联合边界同源绑定、篡改拒绝及100us整板短启动均通过；即将运行同一冻结版本的完整CSI输入至三个不同TMDS屏幕帧，尚无完整帧耗时结论。',
                   '继续同次冻结完整测量；首个错误立即停止。完成后汇总输入至输出、AI提交/新屏幕帧间隔及connected-board-native-lock1同源资源，容量失败保持原判。')
        execute('full_frames','run_board_frame.py','--root',board,'--optimize-level','5')
        execute('accounting','finish_board_measurement.py','--board',board,'--map',mapped,
                '--root',root/'accounting','--report',report)
        checkpoint('完整CSI至TMDS联合测量与同源资源报告完成',
                   '2026-10-03 '+str(board)+'完整连续测量已结束，统一报告见'+str(report)+'；实际完整结果以报告数据为准，仍未取得容量适配/P&R/STA或板测通过。',
                   '向用户提交本次完整帧延迟、间隔、分段热点及同源资源结果；在这份完整测量基础上确定后续优化。')
        (root/'result.json').write_text(json.dumps(dict(status='UNIFIED_PRODUCTION_BOARD_MEASUREMENT_COMPLETE',report=str(report),
                      report_sha256=sha(report),inputs_sha256=sha(root/'inputs.json')),indent=2))
    except Exception as exc:
        data=dict(status='PRODUCTION_BOARD_PIPELINE_STOPPED',error=str(exc),stages=stages,inputs_sha256=sha(root/'inputs.json'))
        (root/'result.json').write_text(json.dumps(data,indent=2))
        checkpoint('联合整板测量流程遇到门禁停点，未计完整通过',
                   '2026-10-03 本次顺序流程在'+str(stages[-1]['name'] if stages else '准备')+'停止：'+str(exc)+'。未自动重试，证据目录'+str(root)+'。',
                   '读取该次阶段日志与真实首个失败原因，短复现解决后再决定继续；不得把未完成测量写成完整通过。')
        raise


if __name__=='__main__':main()
