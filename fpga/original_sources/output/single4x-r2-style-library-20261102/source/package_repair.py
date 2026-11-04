"""Two-style override plus naming correction; keep the first 16-pack frozen."""
from pathlib import Path
import json,sys,shutil,zipfile,subprocess,ast,hashlib,importlib.util
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import library as l
import repair_styles as rs

BASE=l.D;D=BASE/'revisions/film-pixel-r2';P=D/'delivery/S4R2_老片像素返修及名称修正'

def verify_reel(key,sources):
    path=D/'review/reels'/f'{key}_新结构_四场景各5秒.mp4';cap=cv2.VideoCapture(str(path));observed=[];bad=0
    while True:
        ok,b=cap.read()
        if not ok:break
        a=cv2.cvtColor(b,cv2.COLOR_BGR2RGB);observed.append(hashlib.sha256(a.tobytes()).hexdigest())
        if key=='PIXEL_R39':
            z=a.reshape(180,4,320,4,3);bad+=int((z.max((1,3))!=z.min((1,3))).any(2).sum())
    cap.release();expected=[]
    for s in sources:expected+=json.loads((D/'review/native'/key/(s['scene']+'.json')).read_text('utf-8'))['new_raw_sha256']
    assert observed==expected and len(observed)==608 and len(set(observed))==608 and bad==0
    return dict(style_id=key,reel=str(path),reference_file_sha256=l.sha(path),coefficient_sha256=l.sha(D/'models'/key/'coefficients-padded.bin'),raw_hash_list_sha256=hashlib.sha256(''.join(expected).encode('ascii')).hexdigest(),frames=608,decoded_rgb_matches_all_bin_frame_sha256=True,unequal_4x4_cells_over_all_frames=bad if key=='PIXEL_R39' else None)

def crop_comparison():
    # A nearest-neighbour inspection figure, not an inference filter.
    old=np.array(Image.open(BASE/'review/stills/PIXEL_R39/outdoor_076_old.png'))
    failed=np.array(Image.open(BASE/'review/stills/PIXEL_R39/outdoor_076_new.png'))
    new=np.array(Image.open(D/'review/stills/PIXEL_R39/outdoor_076_new.png'))
    box=(800,280,960,440);im=Image.new('RGB',(1440,515),'#111924');dr=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for i,(name,a) in enumerate([('原R39 / 旧B1',old),('被否定的新结构初版',failed),('返修 / 实际BIN输出',new)]):
        dr.text((i*480+5,3),name,font=font,fill='white');im.paste(Image.fromarray(a).crop(box).resize((480,480),Image.Resampling.NEAREST),(i*480,35))
    im.save(D/'像素块面_原始像素整数倍放大.png')

def write_index():
    inventory=json.loads((D/'inventory.json').read_text('utf-8'))['styles']
    lines=['# 十六款视频索引：两款返修与名称修正','','最新用户命名：红色R8-06为热成像，R8-07为漫画。编号与权重不互换。清朝老片和像素链接指向本次返修，其余14款系数与视频保持初版。','','每款原生视频依次室内、室外、人群、水面，各约5秒。像素返修原生视频为RGB无损，608帧解码逐像素与BIN相同；对照缩图不能用于判断4×4格内RGB。','','|编号|名称|原生四景20秒|原图/旧版/新结构对照|','|---|---|---|---|']
    for e in inventory:
        key=e['id']
        if key=='BW_LINE':continue
        root=D if key in ['OLD_FILM','PIXEL_R39'] else BASE
        media_dir='review/display-reels' if key in ['OLD_FILM','PIXEL_R39'] else 'review/reels'
        native=root/media_dir/f'{key}_新结构_四场景各5秒.mp4';comparison=root/media_dir/f'{key}_原图旧版新版_四场景各5秒.mp4'
        assert native.exists() and comparison.exists()
        label=e.get('current_display_name',e['name'])
        lines.append(f'|{key}|{label}|[原生]({native.as_posix()})|[对照]({comparison.as_posix()})|')
    lines+=['','初版R8两款对照视频中的旧文字标签可能仍相反；以本页最新命名和编号为准，原生视频没有错名字幕。','']
    result='\n'.join(lines);(BASE/'16款视频验收索引_返修与更名版.md').write_text(result,encoding='utf-8');(P/'全部16款最新视频入口.md').write_text(result,encoding='utf-8')

@l.torch.inference_mode()
def event_checks(entries,sources):
    rows=[]
    entry=next(e for e in entries if e['id']=='OLD_FILM');old=l.teacher_parameters(entry['source']);new=rs.empty_model();desc,ps=l.decode(D/'models/OLD_FILM');new.residual=desc['residual']
    for n,p in ps.items():
        scale=l.torch.tensor(p['multiplier'],device='cuda').float()/131072
        new.w[n].copy_(l.torch.tensor(p['weight'],device='cuda')*scale[:,None,None,None]);new.b[n].copy_(l.torch.tensor(p['bias'],device='cuda')*scale);getattr(new,n+'_m').copy_(l.torch.tensor(p['multiplier'],device='cuda'))
    for s in sources:
        scene=s['scene'];n=s['frames']//2;a=np.array(Image.open(l.OLD/'acceptance-four-scenes-5s-20261029/originals'/f'{scene}_{n:03d}.png'));x=l.tx(a[None]);z=old(x,True);q=new(x,True)
        rec=dict(scene=scene)
        for tag,model,nodes,feature,out in [('old',old,z,'up2_high_dw','rgb_head'),('repair',new,q,'up1_high','rgb')]:
            f=nodes[feature][0];bars=((f[5]>16)|(f[6]>16)).cpu().numpy().astype(np.uint8);ink=(f[7]>16).cpu().numpy().astype(np.uint8)
            delta=(nodes[out]/128-rs.background(model,nodes)).mean(1)[0].cpu().numpy()*128
            count,_,stats,_=cv2.connectedComponentsWithStats(bars,8);thin=stats[1:];thin=thin[(thin[:,cv2.CC_STAT_HEIGHT]<=8)&(thin[:,cv2.CC_STAT_WIDTH]>=8)]
            rec[tag]=dict(bar_feature_coverage=float(bars.mean()),visible_positive_noise_coverage=float((delta>10).mean()),visible_negative_noise_coverage=float((delta< -10).mean()),ink_feature_coverage=float(ink.mean()),thin_bar_count=len(thin),thin_bar_median_width=float(np.median(thin[:,cv2.CC_STAT_WIDTH])) if len(thin) else 0,noise_positive_mean=float(np.maximum(delta,0).mean()),noise_negative_mean=float(np.maximum(-delta,0).mean()))
        rows.append(rec)
    # Flat colours are not allowed to create fake stripes just to boost density.
    flat=[]
    for value in [0,32,64,128,192,255]:
        x=l.tx(np.full((1,32,32,3),value,np.uint8));z=new(x,True);f=z['up1_high'][:,5:8];assert f.max()==0;flat.append(value)
    return dict(records=rows,uniform_greys_without_events=flat,no_inference_rng_or_overlay=True,old_and_new_event_locations_not_required_identical=True)

def main():
    l.setup(20261102);l.guard();entries=[e for e in json.loads((D/'inventory.json').read_text('utf-8'))['styles'] if e['id'] in ['OLD_FILM','PIXEL_R39']]
    sources=json.loads((D/'review/source_manifest.json').read_text('utf-8'))['scenes']
    cached=D/'rgb-reference-reel-proof.json'
    video_checks=json.loads(cached.read_text('utf-8'))['records'] if cached.exists() else [verify_reel(e['id'],sources) for e in entries]
    for r in video_checks:
        expected=[]
        for s in sources:expected+=json.loads((D/'review/native'/r['style_id']/(s['scene']+'.json')).read_text('utf-8'))['new_raw_sha256']
        assert r['raw_hash_list_sha256']==hashlib.sha256(''.join(expected).encode('ascii')).hexdigest()
        assert r['coefficient_sha256']==l.sha(D/'models'/r['style_id']/'coefficients-padded.bin')
        r['rgb_reference_cache_retained']=Path(r['reel']).exists()
        r['rgb_reference_note']='actually verified before pointwise display conversion; large cache removed afterwards when absent'
    display_checks=json.loads((D/'display-codec-verification.json').read_text('utf-8'))
    assert len(display_checks['records'])==12
    for r in display_checks['records']:
        assert r['maximum_rgb_codec_difference']<=4
        if r['unequal_4x4_cells'] is not None:assert r['unequal_4x4_cells']==0
    events=event_checks(entries,sources);l.save_json(D/'old-film-event-checks.json',events);crop_comparison();P.mkdir(parents=True,exist_ok=True)
    rows=[]
    for e in entries:
        key=e['id'];src=D/'models'/key;dest=P/'styles'/key;dest.mkdir(parents=True,exist_ok=True)
        assert l.sha(e['source'])==e['source_sha256']
        for name in ['coefficients.bin','coefficients-padded.bin','layout.json','graph.json','load_words.hex','integer-coefficients.npz','reference-vectors.json','selected.pt','training.json']:shutil.copy2(src/name,dest/name)
        for s in sources:
            scene=s['scene'];r=json.loads((D/'review/native'/key/(scene+'.json')).read_text('utf-8'))
            assert r['native_video']['decoded_rgb_all_frame_hashes_exact'] and r['frames']==152
            assert r['coefficient_sha256']==l.sha(src/'coefficients-padded.bin')
            (dest/'test-vectors').mkdir(exist_ok=True);shutil.copy2(D/'review/stills'/key/f'{scene}_076_new.png',dest/'test-vectors'/f'{scene}_output.png')
            (P/'common/test-inputs').mkdir(parents=True,exist_ok=True);shutil.copy2(l.OLD/'acceptance-four-scenes-5s-20261029/originals'/f'{scene}_076.png',P/'common/test-inputs'/f'{scene}.png')
        rows.append(dict(id=key,name=e['name'],directory='styles/'+key,parameters=6787,coefficient_sha256=l.sha(src/'coefficients-padded.bin'),replaces_initial_new_structure_sha256=l.sha(BASE/'models'/key/'coefficients-padded.bin'),visual_accepted=False))
    l.save_json(P/'catalog.json',dict(accepted_or_retained_style_count=2,styles=rows,patch_for_initial_16_style_zip=True,original_B1_hardware_compatible=False,not_new_hardware_release=True))
    shutil.copy2(D/'name-corrections.json',P/'name-corrections.json');shutil.copy2(D/'pixel-cell-audit.json',P/'previous-video-cell-audit.json')
    shutil.copy2(D/'old-film-event-checks.json',P/'old-film-event-checks.json');l.save_json(P/'lossless-video-verification.json',dict(status='PASS',scope='internal RGB numerical references; not compatible user previews',records=video_checks,total_native_frames=1216))
    shutil.copy2(D/'display-codec-verification.json',P/'display-codec-verification.json')
    if (D/'ui-video-colour-feedback.json').exists():shutil.copy2(D/'ui-video-colour-feedback.json',P/'ui-video-colour-feedback.json')
    (P/'previews').mkdir(exist_ok=True)
    for e in entries:shutil.copy2(D/'review/static'/e['id']/'四场景对比.jpg',P/'previews'/(e['id']+'.jpg'))
    shutil.copy2(D/'像素块面_原始像素整数倍放大.png',P/'previews/像素块面.png')
    for name in ['verify_package.py','infer_package.py']:shutil.copy2(BASE/'source'/name,P/name)
    (P/'source').mkdir(exist_ok=True)
    for name in ['repair_styles.py','render_repair.py','correct_repair_names.py','audit_pixel_cells.py','display_compatible_repair.py','package_repair.py']:
        src=BASE/'source'/name;ast.parse(src.read_text('utf-8'));shutil.copy2(src,P/'source'/name)
    native={e['id']:str(D/'review/display-reels'/f"{e['id']}_新结构_四场景各5秒.mp4") for e in entries}
    comparison={e['id']:str(D/'review/display-reels'/f"{e['id']}_原图旧版新版_四场景各5秒.mp4") for e in entries}
    l.save_json(P/'media-links.json',dict(native=native,comparisons=comparison))
    doc='''# S4R2两款返修及名称修正

本补丁只替代第一版新结构16款包中的OLD_FILM和PIXEL_R39。先前16款ZIP、原B1采用包、硬件、Flash和现行部署均不修改。油画、水墨本组视频用户反馈无明显差别，保持原版新结构系数。

名称按2026-11-02用户最新纠正：红色调R8-06为“热成像”，R8-07为“漫画”。只换名称，不交换编号或权重；原板标签未写入。本名称不表示从RGB测得物理温度。其他风格的实际系数保持16款初版，新name-corrections.json优先于初版的相反名称。

老片根因：通用初始化截去旧曲率事件通道，并省去第三残差的横向延展；稀疏事件又被整图逐点误差忽略。此次在相同15卷积、两残差图内重新安排曲率有限窗口、亮暗触发、横向延展和暗点通道，单独校准分布；500步真实QAT只拟合底色，不能压低事件读出权重。最后采用第0步的较好底色系数，不冒称第500步更好。噪条位置与原版不同，属同类局部事件，不是逐像素相同。约30像素理论横向支撑、实际连通宽度见old-film-event-checks.json。没有推理RNG或噪条叠加。

像素重新保留原R39离散配色、形态决策和最后硬阈值，将配色放在硬阈值后；800步真实QAT只调整共享RGB取色范围并保持整数DC。四分辨率RGB验证误差1.219到约0.530。高分辨DW严格中心项，任意输入的4×4格内完全同色。原R39限幅锐化没有逐算子等价复刻，不能称精确R39；本候选只保留其配色和像素形状方向，待用户验收。

第一版原始BIN PNG虽同色，但有损H264/YUV420视频在人群中帧破坏37662个格子，最大通道差40。首次返修libx264rgb CRF0在本地数值解码精确，却在用户对话播放器显示粉绿，作为预览已作废。正式观看路径为review/display与display-reels中的YUV444 CRF0视频，明确BT601/SMPTE170M矩阵与sRGB传递；不做色度子采样，没有空间滤镜。实际与内部RGB数值参考逐帧测量转换误差，最大不超过4级，详见display-codec-verification.json，不能宣称显示视频RGB字节完全相同。像素所有608帧、35020800个4×4格解码后仍严格同色。三列对照有缩小排版，不能据其缩图判断原生格内RGB。

RGB视频只用于内部数值参考，不作为用户播放器预览。全部参考重新解码，与BIN原始帧RGB SHA逐项相同；验证及展示转换后清理这些大体积缓存，保留每帧SHA、系数SHA及rgb-reference-reel-proof.json证据。PNG与test-vectors是字节精确的可直接查看结果。正式视频仅链接review/display和display-reels。

每款仍6432权重+355偏置=6787参数、同一S4R2PKG1格式16KB镜像，旧B1硬件不能直接装载。独立小张量整数参考、整720p检查点与BIN、包内独立BIN复演、CRC/HEX/SHA/ZIP和视频PTS均核验。无新增推理滤镜/外部锐化/时序滤波/光流，不改算子或分辨率。

视频均依次室内、室外、人群、水面，各152帧约5.061秒，四景608帧约20.245秒。完整路径见media-links.json。优先看原生像素视频；放大查看源图采用整数倍最近邻，避免播放器缩放再次插值。

包内verify_package.py --full-reference只依赖Python标准库；infer_package.py --style-root styles/PIXEL_R39 --input common/test-inputs/indoor.png --output check.png使用已有torch/numpy/Pillow。source用于原工程追溯，不是完整独立训练环境。用户新视觉验收仍待定，软件数值PASS不等于硬件放行。
'''
    (P/'交给主项目.md').write_text(doc,encoding='utf-8');(D/'交给主项目_返修及名称修正.md').write_text(doc,encoding='utf-8')
    write_index()
    manifest={str(p.relative_to(P)).replace('\\','/'):l.sha(p) for p in P.rglob('*') if p.is_file() and p.name!='manifest-sha256.json'};l.save_json(P/'manifest-sha256.json',manifest)
    p=subprocess.run([sys.executable,'-B',str(P/'verify_package.py'),'--root',str(P),'--full-reference'],capture_output=True,text=True,encoding='utf-8',timeout=180);assert p.returncode==0,(p.stdout,p.stderr)
    result=json.loads(p.stdout);assert result['style_count']==2;l.save_json(D/'portable-verification.json',result)
    sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('repair_standalone',P/'infer_package.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for r in rows:
        engine=mod.Engine(P/r['directory'])
        for s in sources:
            x=np.array(Image.open(P/'common/test-inputs'/f"{s['scene']}.png"));want=np.array(Image.open(P/r['directory']/'test-vectors'/f"{s['scene']}_output.png"));assert np.array_equal(engine(x),want)
    zpath=D/'delivery/S4R2_老片像素返修及名称修正_20261102.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,str(Path(P.name)/p.relative_to(P)))
    with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
    l.save_json(D/'delivery-verification.json',dict(status='PASS_TWO_REPAIRS_AND_NAME_CORRECTIONS_NOT_VISUAL_ACCEPTANCE',zip=str(zpath),zip_sha256=l.sha(zpath),zip_bytes=zpath.stat().st_size,native_clips=8,native_frames=1216,standalone_full720_cases=8,standalone_full720_scalars=8*720*1280*3,lossless_rgb_reference_checks=video_checks,compatible_display_checks=display_checks,original_frozen_16_zip_sha256=l.sha(BASE/'delivery/新结构S4R2_16款风格包_20261102.zip')))
    print('TWO_STYLE_REPAIR_PACKAGE_COMPLETE',zpath,flush=True)

if __name__=='__main__':main()
