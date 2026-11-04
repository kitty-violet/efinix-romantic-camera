"""Freeze the pixel-only revision after complete video and integer checks."""
import shutil,statistics,zipfile,subprocess,importlib.util,ast,time
from PIL import ImageDraw,ImageFont
from common import *

NEW=D/'models/PIXEL_COHERENT'
P=D/'delivery/S4R1_像素块间接续修正'

def main():
    o.setup(20261102);guard();r=json.loads((D/'review/complete.json').read_text('utf-8'));assert r['source_frames']==1073 and len(r['videos'])==3
    for v in r['videos']:assert v['frames']==1073 and v['pts_error_seconds']<1e-5 and sha(v['path'])==v['sha256']
    protected=[]
    for q in [BASE/'protected-r2-artifacts.json',D/'protected-r1-originals.json']:
        for f in json.loads(q.read_text('utf-8'))['files']:
            assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],f['path']
            protected.append(f)
    save(D/'preservation-verification.json',dict(status='PASS',original_R1_files=161,original_R2_files=575,files=len(protected),all_sha256_unchanged=True))
    g,a=o.decode(R1);h,b=o.decode(NEW);assert g==h
    changes=[]
    for n in a:
        for key in a[n]:
            if not np.array_equal(a[n][key],b[n][key]):changes.append((n,key))
    assert changes==[('up1_low_dw','weight')]
    w=b['up1_low_dw']['weight'];assert np.all(w[:24]>=0) and np.all(w[:24].sum((1,2,3))==127) and np.array_equal(w[24:],a['up1_low_dw']['weight'][24:])
    save(D/'coefficient-change-proof.json',dict(status='PASS',only_changed_fields=changes,graph_identical=True,parameters=4035,operators=11,residual_blocks=1,int8_weights_changed=int((w!=a['up1_low_dw']['weight']).sum()),kernel=w[0,0].tolist(),DC_integer_sum=127,all_biases_Q17_and_other_weights_unchanged=True,final_palette_and_high_resolution_centre_weights_unchanged=True))
    style=P/'styles/PIXEL_COHERENT';style.mkdir(parents=True,exist_ok=True)
    for p in NEW.iterdir():
        if p.is_file():shutil.copy2(p,style/p.name)
    for name in ['infer_package.py','verify_package.py']:shutil.copy2(BASE/'source'/name,P/name)
    sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('standalone_pixel_coherence',P/'infer_package.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);e=mod.Engine(style)
    cases={p.stem:p for p in (D/'inputs').glob('*.png')}
    original=o.r2.OLD/'acceptance-four-scenes-5s-20261029/originals'
    cases.update({n:original/f'{n}_076.png' for n in ['indoor','outdoor','crowd','water']})
    checks=[];engine=o.Engine(NEW);(P/'test-inputs').mkdir(exist_ok=True);(P/'test-outputs').mkdir(exist_ok=True)
    with torch.inference_mode():
        for case,path in cases.items():
            a=np.array(Image.open(path).convert('RGB'));result=e(a);expected=rgb(engine(o.tx(a[None])));assert np.array_equal(result,expected)
            if case.startswith('video_') or case=='photo':assert np.array_equal(result,np.array(Image.open(D/'review'/f'{case}_one_after.png')))
            shutil.copy2(path,P/'test-inputs'/f'{case}.png');Image.fromarray(result).save(P/'test-outputs'/f'{case}.png');checks.append(dict(case=case,all_pixels_equal=True,shape=list(a.shape)))
    save(D/'standalone-verification.json',dict(status='PASS',native720_cases=len(checks),scalar_comparisons=len(checks)*720*1280*3,checks=checks))
    # Pixel-native close-ups are nearest-neighbour magnifications for inspection.
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);zoom=Image.new('RGB',(2304,672),'#101820');dr=ImageDraw.Draw(zoom)
    for i,(key,title) in enumerate([('one_before','单残差：修前'),('one_after','单残差：修后'),('two','双残差')]):
        im=Image.open(D/'review'/f'photo_{key}.png').crop((160,220,544,540)).resize((768,640),Image.Resampling.NEAREST)
        zoom.paste(im,(i*768,32));dr.text((i*768+8,2),title,font=font,fill='white')
    zoom.save(D/'review/雪山局部_三版原像素放大.png')
    stills=json.loads((D/'review/still-metrics.json').read_text('utf-8'));old=json.loads((D/'review/old-four-scene-temporal.json').read_text('utf-8'))
    text=['# 像素风：单残差块间接续修正','','只比较像素风。原单残差16款包、双残差16款及返修包均保留；575个R2资产和161个原R1资产结束时SHA全部未变。未改硬件、Flash或其他工作流。','', '本轮仍是S4R1_8_32_8_V1，11卷积13节点、3808权重+227偏置=4035参数，488908800MAC/720p、8KB S4R1PKG1。同原单残差结构/加载格式；与双残差16KB不直接互换。','', '实际只改变已有up1_low_dw的前24个低分辨DW权重。它们从中心127改为对称非负核[[8,16,8],[16,31,16],[8,16,8]]，每通道整数权重和127。让最后选色参考相邻格，减少彼此孤立的块；不加算子、不做输出CPU滤镜。全部偏置、Q17、其他权重及最近邻4倍/高分辨中心权重完全不变。','', '320个均匀RGB样本（256灰阶+64随机RGB）与修前逐值相同，保留均匀颜色响应；这不表示非均匀区域的每个颜色都不变，邻域选色本来会改变边界色。4×4块内仍严格同色，不能把本轮修正表述为增大像素块。','', '先直接构造这一组实际网络系数，再做500步约束QAT。验证最优保留第0步修正初值，训练并未带来额外改善；不把500步当作改善证据。训练仅原项目训练集，用户新图/视频未参与梯度拟合。','', '图片输入1280×960，等比缩到960×720并留边。视频是584×1280手机录屏：固定裁切雪山媒体区域(x0,y136,w584,h440)，排除手机状态/标题/评论区域，保留源中已叠加的昵称、字幕及播放图标；等比缩放到约956×720留边，尺寸四像素对齐，误差仅整数取整。各版使用完全相同的720p RGB。','', '全1073帧原生像素与输入720p SHA逐帧读取、90k时间戳保留，不用固定60FPS重排，不删原源重复帧、不插帧、不降低帧率。三模型均实际执行BIN，组执行对独立执行、修正检查点对BIN的全720结果相等。视频无推理光流，光流仅用于评价相邻帧。','', 'MP4采用已确认可正常显示的BT601 limited YUV444 CRF0/sRGB；源RGB到视频存在色彩换算舍入，不能称RGB字节精确。两份原生像素输出每个解码帧逐格核对同色；完整对照视频每个模型面板都是原生1280×720，不缩成小图。PNG是精确整数输出。','', '|测试图|版本|孤立格检测数|同区域异常跳变数|所有网格平均跳变|','|---|---|---:|---:|---:|']
    for s in stills:text.append(f"|{s['case']}|{s['variant']}|{s['isolated_cells']}|{s['unsupported_jumps']}|{s['all_grid_mean_jump']:.3f}|")
    text+=['','孤立格检测：四邻各方向最大通道色差均大于20；同区域异常跳变：原输入邻格色差<4而输出>24。它们是复查工具，不能等同于所有视觉割裂，更不能单凭数量减少就称成功。','', '|旧独立场景|单残差修前帧差|单残差修后帧差|双残差帧差|','|---|---:|---:|---:|']
    for s in old:
        vals=[statistics.mean(p[k]['warp_mae'] for p in s['pairs']) for k in ['one_before','one_after','two']];text.append('|'+s['scene']+'|'+ '|'.join(f'{v:.3f}' for v in vals)+'|')
    text+=['','以上每景5对原相邻帧（共20对），运动补偿帧差仅辅助检查，需结合结构、对比和视频观察；没有通过复制帧或降帧来降低数值。用户雪山视频另有连续邻帧时序采样，明细在validation.json。','', '软件核验通过不等于用户视觉验收或硬件放行；修正版单独保存，原版不覆盖。完整视频不塞ZIP重复占盘。','', '复演：python -B verify_package.py --full-reference；python -B infer_package.py --style-root styles/PIXEL_COHERENT --input test-inputs/photo.png --output check.png。前者标准库，后者使用项目已有torch/numpy/Pillow。包内8张720p输入和精确输出可独立复查。','', '视频顺序：左上原图、右上单残差修前、左下单残差修后、右下双残差。','']
    for v in r['videos']:text.append(f"[{Path(v['path']).name}]({Path(v['path']).as_posix()})")
    doc='\n'.join(text);(D/'像素修正说明与对照.md').write_text(doc,encoding='utf-8');(P/'交接说明.md').write_text(doc,encoding='utf-8')
    (P/'source').mkdir(exist_ok=True)
    for p in (D/'source').glob('*.py'):ast.parse(p.read_text('utf-8'));shutil.copy2(p,P/'source'/p.name)
    for p in ['source-manifest.json','preservation-verification.json','coefficient-change-proof.json','standalone-verification.json']:shutil.copy2(D/p,P/p)
    shutil.copy2(D/'review/complete.json',P/'validation.json');shutil.copy2(D/'review/photo_四版对照.png',P/'雪山图对照.png');shutil.copy2(D/'review/雪山局部_三版原像素放大.png',P/'雪山局部放大.png')
    save(P/'catalog.json',dict(model_id='S4R1_8_32_8_V1',accepted_or_retained_style_count=1,styles=[dict(id='PIXEL_COHERENT',directory='styles/PIXEL_COHERENT',name='像素动画块间接续修正候选',coefficient_sha256=sha(NEW/'coefficients-padded.bin'),visual_accepted=False)],original_R1_R2_retained=True,hardware_implemented=False))
    save(P/'manifest-sha256.json',{p.relative_to(P).as_posix():sha(p) for p in P.rglob('*') if p.is_file() and p.name!='manifest-sha256.json'})
    cmd=[sys.executable,'-B',str(P/'verify_package.py'),'--root',str(P),'--full-reference'];q=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',timeout=180);assert q.returncode==0,(q.stdout,q.stderr);save(D/'portable-verification.json',json.loads(q.stdout))
    zpath=D/'delivery/S4R1_像素块间接续修正_20261102.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,Path(P.name)/p.relative_to(P))
    with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
    save(D/'delivery-verification.json',dict(status='PASS_NUMERICAL_MEDIA_NOT_VISUAL_ACCEPTANCE',zip=str(zpath),zip_sha256=sha(zpath),zip_bytes=zpath.stat().st_size,original_protected_files=len(protected),all_original_sha256_unchanged=True,independent_full720_cases=len(checks),actual_video_source_frames=1073,parameters=4035,only_changed_layer='up1_low_dw weights',no_hardware_modified=True))
    print('PIXEL_REVISION_PACKAGED',zpath,flush=True)

if __name__=='__main__':main()
