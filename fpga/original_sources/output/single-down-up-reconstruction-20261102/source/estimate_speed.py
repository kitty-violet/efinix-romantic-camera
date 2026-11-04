"""Conditional speed estimate anchored to the existing 150 MHz board ledger."""
from pathlib import Path
import json,math,hashlib,ast
ROOT=Path(r'E:\fpga');D=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def bulk(rows):
    shapes={'input':(720,1280,3)};costs=[]
    for r in rows:
        h,w,c=shapes[r['inputs'][0]]
        if 'kernel' in r:
            h=(h+r['stride']-1)//r['stride'];w=(w+r['stride']-1)//r['stride']
            # Current row-pair core emits three rows for s1, two for s2.
            # For a proposed s4, assume one output row/window and no extra
            # stalls beyond the current core's per-column service policy.
            out_rows=3 if r['stride']==1 else 2 if r['stride']==2 else 1
            parts=3 if r['kind']=='conv3x3' else math.ceil(c/8) if r['kind']=='pw1x1' else 1
            reuse=(r['kind']=='pw1x1' and parts==1) or (r['kind']=='dw3x3' and r['stride']==1)
            vectors=h*w*math.ceil(r['channels']/8)/out_rows
            cycles=vectors*(parts+1-int(reuse))
            macs=h*w*r['channels']*(c//r['groups'])*r['kernel']**2
            costs.append(dict(node=r['id'],macs=macs,model_bulk_cycles=cycles,output_rows=out_rows,parts=parts))
        elif r['kind']=='nearest':h*=r['factor'];w*=r['factor']
        shapes[r['id']]=(h,w,r['channels'])
    return dict(macs=sum(x['macs'] for x in costs),cycles=sum(x['model_bulk_cycles'] for x in costs),nodes=costs)

def main():
    audit=ROOT/'workspaces/camera-color-calibration/architecture-audit-20261101/analysis.json'
    binding=ROOT/'workspaces/camera-color-calibration/architecture-audit-20261101/release-binding.json'
    a=read(audit);b=read(binding)
    protected=[x for x in b['files'] if x['path'].endswith(('compute222.bin','dma222.bin'))]
    assert len(protected)==2 and all(sha(Path(x['path']))==x['sha256'] for x in protected)
    old=bulk(read(D/'original2x2x-r3/graph.json')['nodes']);new=bulk(read(D/'single4x-r2/graph.json')['nodes'])
    assert old['macs']==966758400 and new['macs']==640051200
    period=58.31;observed_bulk=a['timing']['analytical_bulk_ms']
    # Existing 42.72 ms is analytical service, not a measured pure-compute
    # fraction. Retain its unknown remainder unchanged as a stated scenario.
    calibrated_new_bulk=observed_bulk*new['cycles']/old['cycles']
    saved=observed_bulk-calibrated_new_bulk
    new_period=period-saved
    result=dict(status='CONDITIONAL_ESTIMATE_NOT_HARDWARE_MEASUREMENT',clock_mhz=150,
        input_shape=[720,1280],old=old,new=new,
        mac_reduction_pct=100*(1-new['macs']/old['macs']),
        mac_proportional_speedup=old['macs']/new['macs'],
        baseline_period_ms=period,baseline_fps=1000/period,
        mac_proportional_whole_frame_fps=1000/period*old['macs']/new['macs'],
        bulk_service_scenario=dict(old_analytical_bulk_ms=observed_bulk,
            new_calibrated_bulk_ms=calibrated_new_bulk,saved_service_ms=saved,
            retained_other_ms=period-observed_bulk,conditional_frame_ms=new_period,
            conditional_fps=1000/new_period,conditional_speedup=period/new_period),
        assumptions=['Same 150MHz core, grouping and local SRAM policy.',
            's4 produces one row per source window; direct nearest4 supports existing reader-style reuse.',
            'Existing tile/halo overhead scales with nominal row service for this estimate.',
            'Unattributed wait, DMA, control and input scheduling remainder retained unchanged.',
            'No claim of measured new DDR traffic or newly compiled descriptor count.'],
        limitations=['New stride4/nearest4 hardware and scheduling have not been implemented or timed.',
            'Shared-array service differs from MAC count; output/RQ/row access can dominate.',
            'DMA overlap and residual implementation may change exposed stalls; this is not a bound or confidence interval.',
            'Do not add other streams\u0027 6.628ms optimization independently; savings overlap.',
            'Reconstruction quality does not demonstrate retention of existing five styles.'],
        evidence={str(p):sha(p) for p in [audit,binding,D/'original2x2x-r3/graph.json',D/'single4x-r2/graph.json']},
        hardware_or_flash_modified=False)
    (D/'speed-estimate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=f'''# 精简三段后的条件速度估计

按新实验结构，合并一次下采样、删一个残差块、合并一次上采样：24个卷积减到15个，权重与偏置10819减到6787。总下采样仍4倍，实际处理仍720p。

MAC从{old['macs']:,}减到{new['macs']:,}，减少{result['mac_reduction_pct']:.2f}%。若整帧时间都随MAC同比缩放，速度为{result['mac_proportional_speedup']:.3f}倍，即提升{100*(result['mac_proportional_speedup']-1):.1f}%，17.15FPS可条件推算到约25.9FPS。此情景把搬运/控制也同比缩放，不能当板测预报或严格上限。

更贴近当前150MHz共享行计算核心的情景：原已绑定程序的静态服务42.71968ms；根据各层输出向量、8通道组、输入分段与RQ复用模型，新的服务约{calibrated_new_bulk:.2f}ms，少{saved:.2f}ms。其余{period-observed_bulk:.2f}ms保持原样，则58.31ms/帧变成{new_period:.2f}ms/帧，约{1000/new_period:.2f}FPS，速度提升约{100*(period/new_period-1):.1f}%。

这是真实旧程序账本校准的静态情景估算，不是新结构RTL、完整DMA调度或实板测量。stride4和nearest4尚未适配；残差、银行争用、搬运、采集等待可能改变结果。不能保证30FPS，也不能把其他对话约6.628ms的精简收益再直接相加。

实际推理预览已完成，说明重建任务中大体画面能保留；五套已有风格能否迁移到新结构仍需重训验收。本轮仅计算估计，不改硬件、部署与Flash。

逐层模型和证据SHA见speed-estimate.json；复现：使用既有隔离Python运行source/estimate_speed.py，无新依赖。
'''
    (D/'精简结构速度估计.md').write_text(text,encoding='utf-8')
    ast.parse(Path(__file__).read_text('utf-8'))
    print(json.dumps({k:result[k] for k in ['mac_reduction_pct','mac_proportional_speedup','baseline_fps','mac_proportional_whole_frame_fps','bulk_service_scenario']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
