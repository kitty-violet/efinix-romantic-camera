"""Record the later MC experiment separately from the R20 review videos."""
from core import *
import ast

def main():
    d=OUT/'exports/mc-geometry-only-r25';r=OUT/'r25-geometry-only-support';review=OUT/'pixel-a-shapes-and-mc-review'
    v=json.loads((d/'verification.json').read_text('utf-8'));c=json.loads((r/'calibrated/calibration.json').read_text('utf-8'))
    inv=json.loads((r/'invariance-audit.json').read_text('utf-8'));assert v['sha256_padded']==c['coefficient_sha256']==inv['coefficient_sha256']
    assert v['small_oracle_scalar_comparisons']==68800 and all(x['max_error']==0 for x in v['full720_qat_vs_decoded_integer_nodes'].values())
    files=list((OUT/'source').glob('*.py'))
    for p in files:ast.parse(p.read_text('utf-8-sig'),filename=str(p))
    snapshot=d/'source';snapshot.mkdir(exist_ok=True)
    for p in files:shutil.copy2(p,snapshot/p.name)
    progress=dict(latest_MC_research='R25 geometry-only support',coefficient_path=str(d/'coefficients-padded.bin'),crc32=v['crc32_padded'],sha256=v['sha256_padded'],
        training_steps=600,warmup_steps=150,quantized_steps=450,source_checkpoint=str(r/'diagnostic-last.pt'),
        original_half_scale_gate_pass=False,existing_readout_calibrated=True,
        evaluation=c['unseen_test'],fresh_test_seed=c['unseen_test_seed'],counterexamples=c['counterexamples'],
        structural_support_additive_RGB_offset_invariant=True,constant_colour_support_zero=True,
        selected_lighting_negatives_pass=c['selected_lighting_negatives_pass'],positive_geometry_controls_pass=c['positive_geometry_controls_pass'],
        complete_MC_style_accepted=False,video_review_contains_previous_R20_not_R25=True,
        limitation='Hard shadows still falsely extrude. Main colour/height construction and global cuboid arrangement are not solved by the invariant support branch.',
        protected_deployment_modified=False,extra_inference_operators=False,source_AST_pass=len(files),disk=disk_guard())
    save_json(OUT/'mc-progress-latest.json',progress);save_json(review/'MC_FOLLOWUP.json',progress)
    text='''# MC结构研究R25：非风格成品

本包仅为研究候选，不能用来宣称MC已完成。固定B1仍为10819参数、24卷积、29节点、1280×720。

本轮将已有9个低分辨率通道用于结构支持。该分支不能读取原RGB/亮度特征；INT8输入边缘卷积逐输入通道的直流和严格为0。导出BIN实测结构支持对整体RGB加偏置不变，恒色输入支持为0。这一性质只约束支持分支，不能据此宣称整个RGB输出不受光照影响或已恢复真实几何。

训练600步（浮点热身150、普通B1 QAT450），原始0.5支持阈值仅命中约39%，未通过原门槛；保存末次诊断checkpoint，并校准已有PW读出阈值24/增益8后，新192个构造样本命中约66.6%、假阳性约13.1%。没有改写失败历史。

选定彩色墙面柔光反例不再挤出，灰色/彩色物体仍有侧面；硬阴影仍误判，且还没有全局方块场景重建。不得将小方块三面体或局部分类通过当作自然场景MC成功。

独立标准库参考68800标量比较零误差，720p所有节点QAT与解码BIN一致；CRC32 4c66dbd6。有效17792字节、补齐20480字节，无额外推理算子/滤镜/分割/光流/帧历史，未上板。

pixel-a-shapes-and-mc-review中的MC视频是前一轮R20。R25静图、反例及校准证据位于r25-geometry-only-support，不混用这两组权重。
'''
    (d/'README.md').write_text(text,encoding='utf-8');shutil.copy2(r/'invariance-audit.json',d/'invariance-audit.json');shutil.copy2(r/'calibrated/calibration.json',d/'calibration.json')
    save_json(d/'manifest.json',dict(files={str(p.relative_to(d)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in d.rglob('*') if p.is_file() and p.name!='manifest.json'}))
    readme=review/'README.md';body=readme.read_text('utf-8');marker='## 后续MC研究R25'
    if marker not in body:readme.write_text(body+'\n'+marker+'\n\n结构支持分支已封闭RGB/亮度捷径并通过加色偏置不变性检查。彩色柔光反例不挤出、灰色物体有侧面，硬阴影仍未解决；详见MC_FOLLOWUP.json。上面的MC视频仍是R20，不能拿它冒充R25。\n',encoding='utf-8')
    save_json(review/'manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in review.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print('MC_FOLLOWUP_RECORDED',json.dumps(dict(CRC=v['crc32_padded'],test=c['unseen_test'],MC_accepted=False,disk=disk_guard())),flush=True)

if __name__=='__main__':main()
