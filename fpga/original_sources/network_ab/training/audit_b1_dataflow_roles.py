"""Read-only full B1 trace and residual-branch ablations on actual bytes.

Disabling a branch is a diagnostic counterfactual, never a claimed solution.
The stage table reports observed data ranges and each stage's explicit role.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_projected_count_regions import region_geometry
from .train_b1_contour_coherence import contour_coherence
from .fpga_video_resize import verify
from .runtime import write_json,file_hash


def role(name):
    if name=='input':return '已按硬件方式缩到720p的RGB；减128，作为有符号输入'
    if name=='stem':return '720p局部颜色/边缘特征；无旁路保留原照片'
    if name.startswith('down1'):return '第一次步幅2采样到360p；逐通道空间核与通道混合'
    if name.startswith('down2'):return '第二次步幅2采样到180p；32通道特征入口'
    if name.startswith('res'):return '180p非线性邻域特征修正；残差支路按现有整数参数对齐相加'
    if name in ('up1_low_dw','up1_low_pw'):return '180p预测15个分组色阶场；此处仍非RGB图像'
    if name=='up1_high_dw':return '360p色阶场覆盖/夹紧，尚未组合RGB'
    if name=='up2_low_dw':return '360p每场取0/1/2，15场分成7/4/4组，形成1215个颜色组合'
    if name=='up2_low_pw':return '360p把色阶计数组合成颜色坐标；所有后续层共同解码'
    if name=='up2_high_dw':return '720p空间覆盖与边界插值，保留色块内部'
    if name=='rgb_head':return '720p坐标转RGB；加128后再按硬件整数插值到1080p'
    return '最近邻2倍上采样，复制现有特征；自身不恢复新细节'


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    source=m.load_data(a.data);rows=[r for r in source['validation'] if r['id'] in ('val_new_hallway_120','val_stairs_090','val_plaza_210')]
    assert len(rows)==3;records=[];traces=[]
    for branch in (0,1,2,3):
        model=copy.deepcopy(net)
        if branch:
            key=f'res{branch}_conv2__pw';model.weight[key].zero_();model.bias[key].zero_()
            assert model.package['residual_encoding'][f'res{branch}_add']['main_scaled']
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(model.export()),device='cpu')
        for row in rows:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()-128
            target=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            nodes,stats=actual(x,True);p=nodes['rgb_head']/128
            same=model(x/128,all_nodes=True);assert all(torch.equal(v,same[k]) for k,v in nodes.items())
            if not branch:
                sequence=[]
                for name,v in nodes.items():
                    sequence.append(dict(node=name,shape=list(v.shape[1:]),minimum=float(v.min()),maximum=float(v.max()),
                        median_spatial_std=float(v.std((-2,-1)).median()),saturated_preclamp_fraction=stats.get(name),role=role(name)))
                traces.append(dict(id=row['id'],nodes=sequence))
            z=nodes['up2_low_dw'][0,:15];codes=(81*z[:7].sum(0)+9*z[7:11].sum(0)+z[11:15].sum(0)).long().numpy()
            record=dict(id=row['id'],disabled_residual_branch=branch,mae=float((p-target).abs().mean()*128),
                short_contour_turn_excess=float(contour_coherence(p,target)),**region_geometry(codes))
            records.append(record)
        print('DATAFLOW_BRANCH_ABLATION',branch,flush=True)
    write_json(dest/'audit.json',dict(checkpoint_sha256=file_hash(a.checkpoint),manifest_sha256=file_hash(a.data),
        traces=traces,branch_ablations=records,resize_check=verify(),actual_byte_all_nodes_equal=True,
        branch_ablation_is_diagnostic=True,person_identity_objective=False,
        person_note='目标图和原图不同；当前实际输出人物细节过多属于未充分简化，不是用户要求保留身份。'))
    lines=['# 固定B1数据流与作用核对','',
        '实际权重逐节点与字节重载模型相等。下表来自一幅真实720p输入的观测；残差关闭仅用于定位，不是交付方案。','',
        '|节点|C×H×W|实际范围|空间标准差中位数|作用|','|---|---|---|---|---|']
    for row in traces[0]['nodes']:
        lines.append('|'+row['node']+'|'+'×'.join(map(str,row['shape']))+f'|{row["minimum"]:g}…{row["maximum"]:g}|{row["median_spatial_std"]:.3f}|'+row['role']+'|')
    lines.extend(['','输入1080→720先使用3:2整数面积采样；CNN内部仍有360/180两个尺度；输出720→1080是整数双线性。训练目标、连通区标签、光流、轮廓一致性只参与训练/诊断，不进入推理。',
        '','人物与其他物体使用同一风格目标，没有原照片身份保持损失。后续应压制多余局部色变、保留形态和主要明暗，不通过复制原人物绕过风格化。',
        '','## 残差作用实测','', '|关闭支路|三个场景MAE平均|小碎块比例|局部转折超量|','|---|---|---|---|'])
    for branch in (0,1,2,3):
        rr=[r for r in records if r['disabled_residual_branch']==branch]
        vals=[np.mean([r[k] for r in rr]) for k in ('mae','small_region_pixels','short_contour_turn_excess')]
        lines.append(f'|{branch or "不关闭"}|{vals[0]:.4f}|{vals[1]:.6f}|{vals[2]:.6f}|')
    lines.extend(['','关闭支路会同时改变颜色和结构；单项指标下降不代表该支路无用。具体场景见audit.json。'])
    (dest/'数据流与支路作用.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
