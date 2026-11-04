"""Verify real package/video provenance and produce a concise evidence report."""
import json
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from . import train_b1_style_recovery as m
from . import train_b1_reconstruction as r
from .publish_b1_reconstruction_research import ActualBytes
from .fpga_video_resize import area_3to2,bilinear_2to3
from .runtime import configure_workspace,contract_guard,file_hash,write_json


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


@torch.inference_mode()
def main():
    configure_workspace();contract_guard();root=m.OUT
    final=load(root/'final_results.json');media=load(root/'videos/results.json');package=root/'delivery/coefficients-padded.bin'
    assert file_hash(package)==final['package_sha256']==media['sha256']
    review=load(root/'delivery/review.json');assert review['independent_oracle_error']==0
    assert len(review['records'])==6 and all(x['export_node_error']==0 and x['nodes']==29 for x in review['records'])
    assert file_hash(r.ROOT/'network_ab/model_b_contract.json')==r.CONTRACT_SHA
    net=ActualBytes(package);videos=[]
    for row in media['results']:
        scene=row['id'];v=row['video'];count=360 if scene=='stairs' else 450
        path=root/'videos'/(scene+'_integer1080.mp4')
        assert v['frames']==count and v['seconds']==count/30 and v['size']==[1920,1080] and v['fps']==30
        assert file_hash(path)==v['sha256'] and path.stat().st_size==v['bytes']
        assert file_hash(row['source'])==row['source_sha256']
        source=np.array(Image.open(root/f'videos/{scene}_090_source1080.png'))
        replay=bilinear_2to3(np.array(r.c.rgb_image(net(r.tensor(area_3to2(source))[None])[0][0])))
        saved=np.array(Image.open(root/f'videos/{scene}_090_integer1080.png'));assert np.array_equal(replay,saved)
        old=load(r.OUT.parent/'b1-sampling-colour-v2-20261005/nineteen-levels/crisp-edges/videos/results.json')
        prior=next(z for z in old['results'] if z['id']==scene)
        videos.append(dict(scene=scene,**v,reloaded_frame_exact=True,
            motion_output=float(np.mean([x['output']['mean'] for x in row['motion']])),
            motion_source=float(np.mean([x['source']['mean'] for x in row['motion']])),
            motion_previous=float(np.mean([x['output']['mean'] for x in prior['motion']])),
            encoding_mae=float(np.mean([x['mae'] for x in row['encoding']]))))
    assert {v['scene'] for v in videos}=={'stairs','plaza'}
    trials=[]
    for p in root.glob('*/completion.json'):
        cfg=load(p.parent/'config.json');done=load(p);pre=load(p.parent/'preflight.json')
        assert pre['status']=='PASS' and pre['resume_error']<1e-5
        code=p.parent/'train_b1_style_recovery.py'
        assert file_hash(code)==cfg['code'][str(Path(m.__file__))]
        assert cfg['data_sha256']==file_hash(root/'data.json') and cfg['contract_sha256']==r.CONTRACT_SHA
        logs=[json.loads(x) for x in (p.parent/'metrics.jsonl').read_text().splitlines() if x]
        trials.append(dict(name=p.parent.name,updates=len(logs),qat=cfg['args']['qat'],
            selected_step=done['selected_step'],score=done['validation']['mean'],resume_error=pre['resume_error']))
    page=(root/'watch.html').read_text(encoding='utf-8')
    for rel in re.findall(r'src="([^"]+)"',page):assert (root/rel).resolve().is_file(),rel
    result=dict(status='TECHNICAL_PASS_ONLY',technical_status='PASS',goal_status=final.get('goal_status','candidate_review'),
        videos=videos,trials=trials,actual_training_updates=sum(x['updates'] for x in trials),
        package_sha256=file_hash(package),contract_sha256=r.CONTRACT_SHA,actual_board=False,visual_acceptance_pending=True,
        code={str(p):file_hash(p) for p in Path(__file__).parent.glob('*b1_style_recovery.py')})
    write_json(root/'final_verification.json',result)
    rows=['# 固定 B1 清晰风格结果','',final.get('assessment','画质待验收。'),'','本轮在原硬件网络内优化权重：24 卷积、29 节点、17792B 有效系数、20480B 传输。未增加旁路或外部后处理；实际板级资源和时序未由本轮验证。',
          '',f'实际训练更新 {result["actual_training_updates"]} 次，不含预检。所有正式路线完成 GPU 原生720前后向、小批次、恢复下一步检验。',
          '','## 相同验证画面的目标保真与边界','',
          '| 模型 | 对目标PSNR↑ | SSIM↑ | RGB MAE↓ | 边界损失↓ | 块内多余纹理↓ |','|---|---:|---:|---:|---:|---:|']
    for name,v in final['models'].items():
        z=v['validation']['mean'];rows.append(f'| {name} | {z["pooled_psnr"]:.3f} | {z["ssim"]:.4f} | {z["mae"]:.3f} | {z["boundary"]:.5f} | {z["flat"]:.5f} |')
    rows+=['','以上指标是对认可风格目标的误差，不能替代原尺寸观感，也不是对原片越相似越好。','',
           '## 实际视频与稳定性','','| 场景 | 时长 | 帧数 | 原片运动补偿误差 | 旧19级 | 本轮 | 编码MAE |','|---|---:|---:|---:|---:|---:|---:|']
    for v in videos:rows.append(f'| {v["scene"]} | {v["seconds"]}s | {v["frames"]} | {v["motion_source"]:.3f} | {v["motion_previous"]:.3f} | {v["motion_output"]:.3f} | {v["encoding_mae"]:.3f} |')
    rows+=['','运动补偿误差包含真实外观变化、光流误差和模型变化；不能单凭该数值宣布无闪烁。视频为真实字节包的软件整数离线推理，30fps为播放帧率。',
           '','## 可复现产物','','- watch.html：无损原尺寸图片、原片与12/15秒实际整数视频。','- delivery：FP32/QAT断点、真实bin和2560字hex。',
           '- final_results.json：全部6验证图和10预留图指标；预留图未用于模型选择。','- final_verification.json：视频全帧数、哈希、训练次数和源码绑定。',
           f'- 权重SHA-256：`{file_hash(package)}`。','',
           '原始方法依据与边界见 experiment-plan.md。本轮结果是已测试候选的工程比较，不证明全局最优。具体外观观察另行记录。']
    (root/'RESULTS.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('STYLE_DELIVERY_VERIFIED',result['actual_training_updates'],flush=True)


if __name__=='__main__':main()
