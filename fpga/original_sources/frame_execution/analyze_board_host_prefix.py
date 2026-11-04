"""Attribute host time in an interrupted run; never manufacture frame time."""
import argparse
import json
from pathlib import Path
import re
from run_units import ROOT, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();src=a.run.resolve();out=a.out.resolve()
    assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    stop=json.loads((src/'host_budget_interruption.json').read_text())
    assert stop['status']=='INTERRUPTED_USER_HOST_BUDGET_PREFIX_PRESERVED'
    assert not stop['observed_errors_before_shutdown']
    prefix=Path(stop['evidence_prefix'])
    for item in stop['files']:
        assert sha(prefix/item['name'])==item['sha256']
    run=json.loads((src/'run_result.json').read_text())
    assert run['inputs_sha256']==stop['inputs_sha256']
    progress=run['progress']
    assert progress[-1]==json.loads((prefix/'progress.json').read_text())
    assert all(x['wall_seconds']<=stop['elapsed_host_seconds'] for x in progress)
    milestones=[x for x in progress if re.search(
        r'BOARD_CAL_DONE|BOARD_CPU_RUN|RAW_FRAME_STORED|RGB_FRAME_(BEGIN|COMPLETE)|CNN_BEGIN',x['event'])]
    points=[dict(sim_ms=float(re.search(r't_ns=([\d.]+)',x['event'])[1])/1e6,
                 wall_seconds=x['wall_seconds']) for x in progress if 'BOARD_PROGRESS' in x['event']]
    windows=[]
    for label,lo,hi in [('capture_and_loader',8,39),('raw_to_rgb',40,74),('cnn_active_with_other_traffic',77,106)]:
        q=[x for x in points if lo<=x['sim_ms']<=hi]
        assert len(q)>=3
        ds=q[-1]['sim_ms']-q[0]['sim_ms'];dw=q[-1]['wall_seconds']-q[0]['wall_seconds']
        windows.append(dict(stage=label,first=q[0],last=q[-1],sim_ms=ds,wall_seconds=dw,
                            host_seconds_per_sim_ms=dw/ds))
    begin=next(x for x in milestones if 'CNN_BEGIN' in x['event'])
    result=dict(status='INTERRUPTED_HOST_PREFIX_ANALYSIS_COMPLETE',source=str(src),
                inputs_sha256=stop['inputs_sha256'],run_result_sha256=sha(src/'run_result.json'),
                interruption_sha256=sha(src/'host_budget_interruption.json'),
                host_budget_seconds=14400,host_before_first_cnn_seconds=begin['wall_seconds'],
                old_partition_pre_cnn_alone_over_budget=begin['wall_seconds']>14400,
                milestones=milestones,windows=windows,complete_frames=0,full_board_pass=False,
                limitation='Host timestamps are log-arrival timestamps; wide windows reduce buffering bias. Not per-module CPU profiling, not the new candidate, not an FPGA frame-latency result.')
    out.mkdir();(out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
