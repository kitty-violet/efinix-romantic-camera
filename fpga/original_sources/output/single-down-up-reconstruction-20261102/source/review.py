"""Native reconstruction pixels, independent scenes, exact source PTS."""
from pathlib import Path
import sys,time,json,copy,shutil
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import reconstruct as r
from reconstruct import ROOT,D,torch,setup,save_json,sha,tx,rgb,metrics,IntegerModel,Model
sys.path.insert(0,str(ROOT/'output/b1-color-cycle-negative-20261027/source'))
from build_four_scene_acceptance import StreamEncoder
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2
import evaluation_utils as ev

OLDOUT=ROOT/'output/b1-color-cycle-negative-20261027'
FONT='C:/Windows/Fonts/msyh.ttc'
KEYS=['original2x2x-r3','single4x-r2']
def load_fp(key):
    ck=torch.load(D/key/'fp32.pt',map_location='cpu',weights_only=True)
    m=Model(ck['single']).cuda();m.load_state_dict(ck['state_dict']);m.qat=False;m.eval();return m
def manifest():return json.loads((OLDOUT/'acceptance-four-scenes-5s-20261029/source_manifest.json').read_text('utf-8'))['scenes']
def framehash(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()

def run():
    setup(20261102);start=time.perf_counter();dest=D/'review';dest.mkdir(exist_ok=False)
    sources=manifest();engines={k:IntegerModel(D/k) for k in KEYS};fp={k:load_fp(k) for k in KEYS}
    frozen={str(p):sha(p) for p in (ROOT/'model-training-workspace/accepted').glob('B1*/coefficients-padded.bin')}
    records=[];contact=Image.new('RGB',(1440,4*302),'#111924');dr=ImageDraw.Draw(contact);font=ImageFont.truetype(FONT,20)
    for row,s in enumerate(sources):
        scene=s['scene'];assert sha(s['source'])==s['source_sha256']
        count=next(i+1 for i,t in enumerate(s['pts']) if t>=3.)
        indices=s['frame_indices'][:count];pts=s['pts'][:count]
        writers={k:StreamEncoder(dest/f'{scene}_{k}.mp4',pts) for k in KEYS}
        compare=StreamEncoder(dest/f'{scene}_三列对比.mp4',pts,1920,392)
        hashes={k:[] for k in KEYS};measurements={k:[] for k in KEYS};timing={k:[] for k in KEYS};temps={k:[] for k in KEYS};previous=None;stills={}
        for n,(index,stamp,a1080) in enumerate(decode_frames(s['source'],indices=indices,hlg_exposure_stops=s['exposure_stops'])):
            assert time.perf_counter()-start<1200,'20 minute review cap'
            assert index==indices[n] and abs(stamp-s['absolute_pts'][n])<1e-8
            assert shutil.disk_usage(D).free>5*2**30
            a=area_3to2(a1080);assert framehash(a)==s['raw_input_frame_sha256'][n]
            x=tx(a[None]);outputs={}
            for k,e in engines.items():
                torch.cuda.synchronize();tick=time.perf_counter();out=rgb(e(x));torch.cuda.synchronize()
                timing[k].append(time.perf_counter()-tick);outputs[k]=out;writers[k].write(out);hashes[k].append(framehash(out));measurements[k].append(metrics(a,out))
            # Half-size panels only for display; native videos above remain 720p.
            panel=Image.new('RGB',(1920,392),'#111924');pd=ImageDraw.Draw(panel)
            labels=['原图','原结构重建：两次2倍 / 三残差','新结构重建：一次4倍 / 双残差']
            for col,(label,img) in enumerate(zip(labels,[a]+[outputs[k] for k in KEYS])):
                pd.text((col*640+6,3),label,font=font,fill='white');panel.paste(Image.fromarray(img).resize((640,360),Image.Resampling.BOX),(col*640,32))
            compare.write(np.asarray(panel))
            if n in [0,count//2,count-1]:
                Image.fromarray(a).save(dest/f'{scene}_{n:03d}_original.png')
                for k,out in outputs.items():Image.fromarray(out).save(dest/f'{scene}_{n:03d}_{k}.png')
            if n==count//2:
                stills=dict(input=a,**outputs)
                for col,(label,img) in enumerate(zip(labels,[a]+[outputs[k] for k in KEYS])):
                    dr.text((col*480+4,row*302+2),s['label']+' / '+label.split('：')[0],font=font,fill='white')
                    contact.paste(Image.fromarray(img).resize((480,270),Image.Resampling.BOX),(col*480,row*302+29))
                for k in KEYS:
                    with torch.inference_mode():floatout=rgb(fp[k](x))
                    save_json(dest/f'{scene}_{k}_float-comparison.json',dict(float32=metrics(a,floatout),integer=metrics(a,outputs[k]),integer_vs_float=metrics(floatout,outputs[k])))
            if previous is not None and n%30==0:
                mx,my,valid,stationary,_=ev.flow_mask(previous[0],a)
                for k in KEYS:temps[k].append(dict(frame=n,metrics=ev.temporal(previous[1][k],outputs[k],mx,my,valid,stationary)))
            previous=(a,outputs)
            if n%30==0:print('REVIEW_FRAME',scene,n,'sec',round(time.perf_counter()-start,1),flush=True)
        videos={k:w.close() for k,w in writers.items()};panelinfo=compare.close()
        sceneinfo=dict(scene=scene,label=s['label'],source=s['source'],source_sha256=s['source_sha256'],frames=count,pts=pts,input_frame_hashes=s['raw_input_frame_sha256'][:count],same_original_pts=True,models={})
        for k in KEYS:
            assert len(set(hashes[k]))==count
            sceneinfo['models'][k]=dict(video=videos[k],raw_hashes=hashes[k],mean_mae=float(np.mean([x['mae'] for x in measurements[k]])),mean_psnr_db=float(np.mean([x['psnr_db'] for x in measurements[k]])),temporal_flow_evaluation_only=temps[k],host_seconds_per_frame_mean=float(np.mean(timing[k])),host_timing_is_not_hardware_speed=True)
        sceneinfo['comparison_video']=panelinfo;records.append(sceneinfo);save_json(dest/'progress.json',dict(complete=len(records)==4,scenes=records))
        # Native 1:1 crop sheet highlights fine detail without enlarged interpolation.
        crops=Image.new('RGB',(1200,450),'#111924');cd=ImageDraw.Draw(crops)
        for col,(label,k) in enumerate(zip(labels,['input']+KEYS)):
            cd.text((col*400+4,2),label.split('：')[0],font=font,fill='white')
            im=Image.fromarray(stills[k]);c=im.crop((440,140,840,540));crops.paste(c,(col*400,32))
        crops.save(dest/f'{scene}_原生局部.png')
        print('REVIEW_SCENE_COMPLETE',scene,{k:round(sceneinfo['models'][k]['mean_psnr_db'],2) for k in KEYS},flush=True)
    contact.save(dest/'四场景重建对比.jpg',quality=96)
    assert all(sha(p)==h for p,h in frozen.items())
    summary=dict(status='COMPLETED_SOFTWARE_RECONSTRUCTION_EXPERIMENT',training_summary=str(D/'training-summary.json'),scenes=records,protected_pack_sha_unchanged=frozen,no_inference_filters=True,no_input_output_shortcut=True,no_inference_flow=True,no_repeat_or_drop=True,outputs_from_exported_integer_coefficients=True,hardware_or_flash_modified=False,hardware_compatibility_verified=False,seconds=time.perf_counter()-start,output_bytes=sum(p.stat().st_size for p in D.rglob('*') if p.is_file()))
    assert summary['output_bytes']<512*2**20
    save_json(D/'verification.json',summary)
    make_readme(summary)

def make_readme(s):
    tr=json.loads((D/'training-summary.json').read_text('utf-8'));models={x['model']:x for x in tr['models']}
    old=models[KEYS[0]];new=models[KEYS[1]]
    lines=['# 单次4倍缩放、双残差：原图重建实验','',
      '用户2026-11-02授权软件结构实验。大下采样解释为stride=4，大上采样为nearest×4；总缩放仍4倍。保持stem8、瓶颈32、输出前8通道，残差由3块减为2块。',
      '','新结构：RGB → 3×3卷积3→8 → DW3×3/s4 → PW8→32 → 两个32通道残差块 → DW3×3 → PW32→8 → 最近邻×4 → DW3×3 → RGB1×1。',
      '','每个残差块为DW/PW/ReLU、DW/PW线性、clamp(main+skip)。对照是旧B1的两次2倍、三残差拓扑，用相同训练预算从重建初始化训练；不是拿油画风格权重作重建对照，也不是复现其历史残差缩放编码。',
      '','两者均无IN/BN、无原图直通输出、无额外锐化或滤镜。固定signed INT8激活，逐输出通道INT8权重，INT32偏置和Q17半离零舍入。当前NPZ是实验参数格式，不能替换现有固定B1部署包。',
      '', '| 指标 | 原结构重建 | 新结构重建 |','|---|---:|---:|',
      f"| 参数（含偏置） | {old['cost']['parameters']} | {new['cost']['parameters']} |",
      f"| 720p MAC | {old['cost']['macs_per_720p']:,} | {new['cost']['macs_per_720p']:,} |",
      f"| 验证集MAE（0—255） | {old['selected_validation_mae']:.3f} | {new['selected_validation_mae']:.3f} |",
      '',f"新结构理论MAC减少{100*(1-new['cost']['macs_per_720p']/old['cost']['macs_per_720p']):.1f}%；这不是实板提速实测。",
      '', '| 独立场景 | 原结构PSNR | 新结构PSNR | 原结构MAE | 新结构MAE |','|---|---:|---:|---:|---:|']
    for x in s['scenes']:
        a=x['models'][KEYS[0]];b=x['models'][KEYS[1]]
        lines.append(f"| {x['label']} | {a['mean_psnr_db']:.2f} | {b['mean_psnr_db']:.2f} | {a['mean_mae']:.2f} | {b['mean_mae']:.2f} |")
    lines+=['','固定数据：32张训练图、3张验证图按源视频隔离，4段验收视频与它们源视频不重合。160×160训练块256个、验证块24个；每个模型1000步FP32+300步QAT，按验证集选择。没有借用四景视频反向训练。',
       '','独立Python整数卷积参考核对小张量全部节点；量化训练与重新载入NPZ输出逐像素相同。四景各约3秒、原生1280×720、原PTS和全部输入帧保留。三列对比仅显示缩小，另存原生重建视频和无损首/中/尾PNG。光流仅用于评价。',
       '','查看review/四场景重建对比.jpg和review/*_三列对比.mp4；逐景详细指标、输入哈希、输出哈希及PTS见verification.json。浮点与量化中帧对照在review/*_float-comparison.json。',
       '','此轮没有改硬件、程序、Flash、任何已冻结风格包或其他对话记录。stride4/nearest4和新调度尚未做硬件实现与验证，不是可直接上板包。',
       '', '复现（既有隔离环境，无新依赖）：', '```powershell',
       f"& 'E:\\fpga\\model-training-workspace\\env\\Scripts\\python.exe' -B '{D / 'source/reconstruct.py'}' train",
       f"& 'E:\\fpga\\model-training-workspace\\env\\Scripts\\python.exe' -B '{D / 'source/reconstruct.py'}' review",
       '```','一轮输出目录采用防覆盖保护；重跑需在独立目录复制source后执行，不能原地覆盖既有结果。','']
    (D/'实验说明.md').write_text('\n'.join(lines),encoding='utf-8')
