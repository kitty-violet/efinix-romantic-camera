"""Verify final research media and record reproducible, bounded conclusions."""
import json
import math
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from . import train_b1_reconstruction as r
from . import research_b1_reconstruction as s
from .publish_b1_reconstruction_research import ActualBytes
from .fpga_video_resize import area_3to2, bilinear_2to3
from .runtime import configure_workspace, contract_guard, file_hash, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def distortion(a, b):
    delta = a.astype(np.float64)-b.astype(np.float64)
    mse = float((delta*delta).mean())
    return dict(mse=mse, mae=float(np.abs(delta).mean()), psnr=10*math.log10(255**2/max(mse, 1e-12)))


@torch.inference_mode()
def main():
    configure_workspace(); contract_guard()
    root=s.OUT; final=load(root/'final_results.json'); media=load(root/'videos/results.json')
    package=root/'delivery/coefficients-padded.bin'
    assert file_hash(package)==final['package_sha256']==media['sha256']
    assert file_hash(r.ROOT/'network_ab/model_b_contract.json')==r.CONTRACT_SHA
    review=load(root/'delivery/review.json')
    assert review['independent_oracle_error']==0 and all(x['export_node_error']==0 for x in review['records'])
    rows=[]
    for row in media['results']:
        v=row['video']; p=root/'videos'/(row['id']+'_integer1080.mp4')
        count=360 if row['id']=='stairs' else 450
        assert v['frames']==count and v['seconds']==count/30 and v['fps']==30 and v['size']==[1920,1080]
        assert file_hash(p)==v['sha256'] and p.stat().st_size==v['bytes']
        assert file_hash(row['source'])==row['source_sha256']
        assert len(row['encoding'])==4
        rows.append(dict(scene=row['id'],path=str(p),**v,
            encoding_mae=[x['mae'] for x in row['encoding']],
            motion_source=float(np.mean([x['source']['mean'] for x in row['motion']])),
            motion_output=float(np.mean([x['output']['mean'] for x in row['motion']]))))
    assert {x['scene'] for x in rows}=={'stairs','plaza'}

    # Isolate the outer 1080 -> 720 -> 1080 resize, without attributing it to B1.
    resize=[]; net=ActualBytes(package)
    board=Image.new('RGB',(1440,380*2),'#17202a'); draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for i,scene in enumerate(('stairs','plaza')):
        original=np.array(Image.open(root/f'videos/{scene}_090_source1080.png'))
        down=area_3to2(original); base=bilinear_2to3(down)
        output=r.c.rgb_image(net(r.tensor(down)[None])[0][0])
        replay=bilinear_2to3(np.array(output))
        saved=np.array(Image.open(root/f'videos/{scene}_090_integer1080.png'))
        assert np.array_equal(replay,saved),'Video evidence does not match package'
        Image.fromarray(base).save(root/f'final-review/{scene}_resize_only1080.png')
        resize.append(dict(scene=scene,frame=90,resize_only=distortion(base,original),
                           actual_network=distortion(replay,original),reloaded_frame_exact=True))
        box=(600,180,1080,520) if scene=='stairs' else (570,450,1050,790)
        for j,(pixels,title) in enumerate(zip((original,base,replay),('原始1080p','只做1080→720→1080','再经过B1实际整数包'))):
            board.paste(Image.fromarray(pixels).crop(box),(480*j,380*i+40))
            draw.text((480*j+5,380*i+8),scene+' | '+title,font=font,fill='white')
    board.save(root/'final-review/resize-versus-network.png')

    trials=[]
    for path in sorted(root.glob('*/completion.json')):
        d=path.parent; config=load(d/'config.json'); complete=load(path)
        preflight=load(d/'preflight.json')
        assert preflight['status']=='PASS' and preflight['resume_error']==0
        assert config['contract_sha256']==r.CONTRACT_SHA
        assert file_hash(d/'research_b1_reconstruction.py')==config['code_sha256']
        history=load(d/'history.json')
        lines=[json.loads(x) for x in (d/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if x]
        best=max(history,key=lambda x:x['validation']['pooled_psnr'])
        trials.append(dict(name=d.name,qat=config['qat'],executed_updates=len(lines),
            final_step=complete['executed_steps'],selected_step=complete['selected_step'],
            best_psnr=best['validation']['pooled_psnr'],best_psnr_step=best['step'],
            stop=complete['stop'],preflight_resume_error=0,
            config_sha256=file_hash(d/'config.json'),completion_sha256=file_hash(path)))
    page=root/'watch.html'; text=page.read_text(encoding='utf-8')
    text=text.replace('软件整数回放，未上板。','软件整数离线回放、30fps播放，未上板。')
    if 'resize-versus-network.png' not in text:
        text=text.replace('<h2>两个来源隔离视频的10张预留测试</h2>',
            '<h2>外部缩放与网络损失分开看</h2><p>下面是原始1080p、仅缩小放大、再经过B1的原尺寸裁图。</p><img src="final-review/resize-versus-network.png"><h2>两个来源隔离视频的10张预留测试</h2>')
    for rel in re.findall(r'src="([^"]+)"',text):assert (root/rel).resolve().is_file(),rel
    page.write_text(text,encoding='utf-8')
    preview=root/'videos/preview.html'
    preview.write_text(preview.read_text(encoding='utf-8').replace('B1颜色分区学习试验','B1原图重建研究').replace('B1真实整数包试验','本轮B1实际整数包'),encoding='utf-8')
    code=['research_b1_reconstruction.py','balance_b1_reconstruction.py','correct_b1_reconstruction_bias.py',
          'prepare_b1_reconstruction_holdout.py','probe_b1_reconstruction_capacity.py',
          'audit_b1_reconstruction_research.py','publish_b1_reconstruction_research.py',
          'verify_b1_reconstruction_research.py','report_b1_reconstruction_research.py','render_b1_region_trial.py',
          'train_b1_learned_colour.py','probe_b1_cartoon_int8.py','b1_trained_image.py','fpga_video_resize.py']
    verification=dict(status='DELIVERY_VERIFIED_NOT_BOARD_OR_STYLE_PASS',
        package_sha256=final['package_sha256'],contract_sha256=r.CONTRACT_SHA,
        weights_bytes=package.stat().st_size,video_rows=rows,resize_diagnostic=resize,
        trials=trials,total_executed_updates=sum(x['executed_updates'] for x in trials),
        preflight_updates_excluded=True,diagnostic_oracle_updates_excluded=True,
        replayed_updates_included=True,watch_sources_exist=True,actual_board=False,
        temporal_error_not_proof_of_no_flicker=True,theoretical_maximum_proven=False,
        code_hashes={name:file_hash(r.ROOT/'network_ab/training'/name) for name in code})
    write_json(root/'final_verification.json',verification)
    print('RECON_RESEARCH_FINAL_VERIFIED',json.dumps(dict(videos=rows,resize=resize,updates=verification['total_executed_updates']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
