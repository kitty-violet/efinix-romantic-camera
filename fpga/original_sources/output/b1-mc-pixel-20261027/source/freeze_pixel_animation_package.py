"""Freeze exactly user-selected R39 medium as 像素动画风格; no deployment edits."""
import ast
import csv
import importlib.util
import zipfile
import zlib
from core import *
from network_ab.training.media import decode_frames
from network_ab.training.fpga_video_resize import area_3to2

@torch.inference_mode()
def main():
    setup();disk_guard()
    src=OUT/'exports/pixel-A-sharp-medium-r39';videos=OUT/'pixel-r39-video-review'
    dest=WORK/'accepted/B1-像素动画风格-R39-20261027'
    delivery=OUT/'delivery';archive=delivery/'像素动画风格_B1_R39中强_20261027.zip'
    assert not dest.exists() and not archive.exists(),'Do not overwrite a frozen package.'
    expected='1c97b7c6aa578a64e259f15a4e3c6d2a932ffda9f02ae06ee3b14c48e2a73d90'
    raw=(src/'coefficients-padded.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected and f'{zlib.crc32(raw):08x}'=='2d9aad69'
    source_manifest=json.loads((src/'manifest.json').read_text('utf-8'))
    for name,digest in source_manifest['files'].items():assert sha(src/name)==digest,name
    model=load_model(src/'selected.pt',device='cpu');assert model.export()==raw
    verification=json.loads((src/'verification.json').read_text('utf-8'))
    assert verification['sha256_padded']==expected and verification['crc32_padded']=='2d9aad69'
    protected=[WORK/'accepted/B1-A-20261008/coefficients-padded.bin',WORK/'accepted/B1-水墨风-20261027/coefficients-padded.bin',
               WORK/'accepted/B1-清朝老片风格-20261027/coefficients-padded.bin',DEPLOY]
    protected_hashes={str(p):sha(p) for p in protected}
    params,residual=decode(raw);layout=json.loads((src/'contract/coefficient_layout.json').read_text('utf-8'));rows=[]
    for op in layout['operators']:
        name=op['name'];key=name.replace('.','__');p=params[name]
        nw=model.weight[key].numel();nb=model.bias[key].numel()
        assert nw==op['raw_weights']==sum(len(w) for w in p.weights)
        assert nb==op['output_channels']==len(p.bias)
        rows.append(dict(name=name,kind=op['kind'],shape=list(model.weight[key].shape),weights=nw,biases=nb,
            q17_multipliers=len(p.multiplier),nonzero_weights=sum(v!=0 for w in p.weights for v in w),nonzero_biases=sum(v!=0 for v in p.bias)))
    count=dict(definition='Stored convolution weights plus biases, including zeros; quantization/control fields separate.',
        weights=sum(r['weights'] for r in rows),biases=sum(r['biases'] for r in rows),network_parameters=sum(p.numel() for p in model.parameters()),
        q17_channel_multipliers=sum(p.numel() for p in model.buffers()),residual_q17_multipliers=2*len(residual),residual_branch_flags=len(residual),
        nonzero_weights=sum(r['nonzero_weights'] for r in rows),nonzero_biases=sum(r['nonzero_biases'] for r in rows),
        convolutions=len(rows),canonical_nodes=len(model.plan['nodes']),effective_image_bytes=17792,transport_bytes=20480,le64_words=2560,layers=rows)
    assert (count['weights'],count['biases'],count['network_parameters'],count['q17_channel_multipliers'])==(10256,563,10819,563)
    binding=json.loads((videos/'binding.json').read_text('utf-8'));summary=json.loads((videos/'summary.json').read_text('utf-8'))
    assert summary['complete'] and binding['models']['medium_r39']['sha256']==expected
    assert binding['detail_caps'][0]==3 and binding['bounded_luma_sharp_amount']==.375
    assert sha(videos/'source_used.py')==binding['script_sha256']
    dest.mkdir(parents=True)
    for name in ['coefficients-padded.bin','coefficients.bin','load_words.hex','residual_encoding.json','selected.pt']:
        shutil.copy2(src/name,dest/name)
    shutil.copytree(src/'contract',dest/'contract')
    for name in ['evidence','source_snapshot','previews','test_vectors']:(dest/name).mkdir()
    shutil.copy2(src/'verification.json',dest/'evidence/integer-verification.json')
    shutil.copy2(src/'README.md',dest/'evidence/pre-selection-README.md')
    save_json(dest/'parameter-count.json',count)
    with (dest/'parameter-count.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    clips=json.loads((ROOT/'output/b1-ink-wash-20261026/independent_test/manifest.json').read_text('utf-8'))['clips']
    model=model.cuda();engine=IntegerEngine(raw);video_audits={};temporal=[];sample=None
    for report in summary['reports']:
        scene=report['scene'];r=dict(scene=scene,coefficient_sha256=expected,source=report['source'],source_sha256=report['source_sha256'],
            frames=report['frames'],seconds=report['seconds'],pts=report['pts'],unique_raw_frames=report['unique_frames']['medium_r39'],
            temporal=dict(selected_R39=report['temporal']['medium_r39'],previous_R36=report['temporal']['r36']),
            content=[x['medium_r39'] for x in report['content']],all_frames_preserved=report['all_frames_preserved'])
        temporal.append(dict(scene=scene,frames=r['frames'],metrics=r['temporal']))
        if scene not in ['corridor','outdoor']:
            save_json(dest/'evidence'/(scene+'-report.json'),r);continue
        original=videos/f'{scene}_medium.mp4';assert sha(original)==report['videos']['medium']['sha256']
        shutil.copy2(original,dest/'previews'/f'{scene}.mp4');r['video']=report['videos']['medium']
        cap=cv2.VideoCapture(str(dest/'previews'/f'{scene}.mp4'));hashes=[];shapes=set();first=None
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if first is None:first=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest());shapes.add(tuple(bgr.shape))
        cap.release();assert len(hashes)==len(set(hashes))==91 and shapes=={(720,1280,3)}
        clip=next(c for c in clips if c['id']==scene);assert sha(clip['source'])==report['source_sha256']
        iterator=decode_frames(clip['source'],seconds=.1,hlg_exposure_stops=clip['exposure_stops'])
        try:idx,stamp,frame=next(iterator)
        finally:iterator.close()
        assert idx==0 and stamp==report['pts'][0]
        a=area_3to2(frame);x=tensor(a)[None];actual=rgb(engine(x*128)/128);qat=rgb(model(x))
        assert np.array_equal(actual,qat)
        Image.fromarray(actual).save(dest/'previews'/f'{scene}_frame0.png')
        Image.fromarray(first).save(dest/'previews'/f'{scene}_encoded_frame0.png')
        delta=float(np.abs(actual.astype(float)-first).mean())
        # H.264 preview is lossy; actual test vector below remains lossless.
        assert delta<8,(scene,delta)
        assert np.array_equal(first,read(videos/f'{scene}_medium_encoded_first.png'))
        video_audits[scene]=dict(frames=91,unique_frames=91,resolution=[1280,720],sha256=sha(original),
            encoded_first_mae_rgb=delta,original_encoded_first_exact=True,qat_integer_frame0_exact=True)
        if scene=='corridor':sample=(a,actual,stamp)
        save_json(dest/'evidence'/(scene+'-report.json'),r)
        print('R39_PACKAGE_VIDEO_CHECK',scene,flush=True)
    save_json(dest/'evidence/temporal-summary.json',dict(coefficient_sha256=expected,regression_vs_R36=True,scenes=temporal))
    save_json(dest/'evidence/selected-video-binding.json',dict(coefficient_sha256=expected,selected_candidate=src.name,
        original_binding_file=str(videos/'binding.json'),original_binding_sha256=sha(videos/'binding.json'),
        original_evaluator_sha256=binding['script_sha256'],source_media_preparation='fixed HLG to sRGB exposure -1; integer 1080 to 720 area3to2; no output filtering',
        curve='A',saturation=1.3,sharp_amount=.375,sharp_cap=3,native_cell=[4,4],inference_history=False,inference_postprocessing=False))
    a,y,stamp=sample
    grid=lambda cell:np.all(y.reshape(720//cell,cell,1280//cell,cell,3).max((1,3))==y.reshape(720//cell,cell,1280//cell,cell,3).min((1,3)))
    assert grid(4) and not grid(8)
    vector=dict(width=1280,height=720,channels=3,format='row-major interleaved RGB888 uint8, no header',scene='corridor',frame=0,pts=stamp,
        input_transform='uint8 minus128',output_transform='int8 plus128',coefficient_sha256=expected,qat_integer_byte_exact=True,
        strict4_constant=bool(grid(4)),strict8_constant=bool(grid(8)),files={})
    for name,v in [('input_rgb888.bin',a),('expected_rgb888.bin',y)]:
        data=v.tobytes();(dest/'test_vectors'/name).write_bytes(data)
        vector['files'][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    save_json(dest/'test_vectors/sample.json',vector)
    for p in (src/'source_snapshot').glob('*.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    for name in ['freeze_pixel_animation_package.py']:
        shutil.copy2(OUT/'source'/name,dest/'source_snapshot'/name)
    shutil.copy2(OUT/'source/verify_pixel_animation_package.py',dest/'verify_package.py')
    acceptance=dict(name='像素动画风格',style_id='pixel-animation',version='R39-medium-20261027',selected_candidate=src.name,
        sharpness='medium',native_cell=[4,4],excluded_R40=True,accepted_by='user',accepted_on='2026-10-27',
        user_statement='你把R39打包吧，命名为这个像素动画风格记住啊，是R39，不是你那个实际的什么最后合并实验',
        visual_selected_for_packaging=True,style_accepted=True,full_temporal_acceptance=False,temporal_regression_vs_R36=True,
        hardware_release=False,actual_board=False,production_deployment_modified=False,extra_inference_operators=False,
        coefficient_sha256=expected,crc32_padded='2d9aad69',effective_bytes=17792,transport_bytes=20480,le64_words=2560,
        convolutions=24,nodes=29,input_resolution=[1280,720],parameter_count=10819,
        checkpoint_reexport_byte_identical=True,integer_software_verified=True,source_experiment=str(src),
        contract_sha256=sha(dest/'contract/model_b_contract.json'))
    save_json(dest/'acceptance.json',acceptance)
    readme=f'''# 像素动画风格 · R39 中强 · 固定 B1 交接包

用户于 2026-10-27 明确选定 **R39 中强版**，并命名为 **“像素动画风格”**。本包冻结该版权重，原生 **4×4** 像素单元。包内只有这一套权重；不包含 R40 最后合并实验，也不是 R39 更强档。

## 主项目接入

1. 使用已有 Python 运行 `python -B verify_package.py`。只用标准库，检查完整清单、哈希、BIN 分区/CRC、HEX、参数量及视频与样本绑定，不安装依赖、不写设备。
2. `coefficients-padded.bin` 是装载入口：**20480 字节，2560 个 LE64 字**。`coefficients.bin` 是17792有效字节，后补2688个零得到装载文件；`load_words.hex` 每行16位十六进制，对应小端64位字。
3. 同时核对 `contract/` 和 `residual_encoding.json`。三个残差均为 **main_scaled=false、multiplier=1、post_multiplier=131071**；仅换卷积权重而不核对执行程序中的残差字段，不能保证同源。
4. 使用 `test_vectors/` 无损RGB888样本核对主项目整数执行结果，然后按接收项目的现行流程集成。

装载 CRC32：**`2d9aad69`**  
装载 SHA256：`{expected}`

本包不含硬件工程或位流，没有修改原A、水墨、清朝老片、V14或现行部署，也没有写Flash。软件数值通过不代表实板验收。

## 参数和数值接口

固定 B1：8/16/32通道、3残差块、24卷积、29规范节点，1280×720。输入RGB HWC uint8减128得到int8，输出int8加128恢复uint8；卷积权重INT8、偏置INT32、逐通道Q17，half-away-from-zero舍入及既定replicate边界。

| 项目 | 数量 |
|---|---:|
| 卷积权重（含零） | 10256 |
| 偏置（含零） | 563 |
| 网络参数合计 | **10819** |
| 另计逐通道Q17乘数 | 563 |
| 另计残差乘数 / 分支标志 | 6 / 3 |
| 实际整数非零权重 / 偏置 | {count['nonzero_weights']} / {count['nonzero_biases']} |

实际张量、导出解码及布局三方统计一致，见 `parameter-count.json` 和CSV；零系数不减少固定网络的装载或计算结构。

## 视觉配置

曲线A，S130色度增益，整数中性灰修正，浅色色度首档12，其余强色度档保留。亮度曲线已编码进网络，不在输出端再乘一次C120，以免改变用户选定的曲线。中强限幅亮度锐化amount=0.375、cap=3，保留原生4×4同色单元。均匀灰256级R=G=B；灰阶均值相对原曲线A偏差不超过1级。

全部效果来自这套实际B1系数。本轮采用系数构造与校准，没有新增梯度训练；没有额外推理滤镜、随机数、历史帧、推理光流、外部锐化或8×8缩放。

## 验证与已知边界

- 既有6组独立整数参考，共68800标量，全720p各节点QAT与导出解码整数误差为0；本次检查点重导出与冻结BIN逐字节一致。
- 室内不过曝走廊、室外街景各91帧约3.0302秒，均为R39中强整数输出。复制后再次完整解码91个唯一帧，保留原PTS，不降帧、不重复帧。
- 相对R36，运动补偿误差室内2.1011→3.0701、室外4.6227→6.3901；高频误差/信号室内0.3683→0.3951、室外0.5008→0.5260。**锐度提高伴随边界时间变化增大**，未宣称全场景无闪动。用户选择视觉版本用于打包，不改写上述测量。
- `evidence/integer-verification.json` 和 `pre-selection-README.md` 中“未定稿/style_accepted=false”是用户选定之前的历史证据，原样保存。本次选择以 `acceptance.json` 为准；实板及完整时序验收仍未完成。

## 观看与逐像素样本

[打开预览页面](previews/review.html) · [室内3秒](previews/corridor.mp4) · [室外3秒](previews/outdoor.mp4)

视频来自用户看过的R39中强文件，未重新调色或处理。MP4为有损编码；帧0 PNG及test_vectors为实际BIN重算的无损输出。原素材HLG经既定输入转换和1080→720面积缩放进入网络，这是素材准备条件，不能当作部署端风格后处理。

`test_vectors/input_rgb888.bin` 和 `expected_rgb888.bin` 各2764800字节，无文件头，逐行交错RGB888。输入按上述数值接口进入同源整数执行程序，输出恢复uint8后应逐字节匹配expected。`sample.json`给出帧号、PTS和SHA256；本次QAT与实际BIN完全一致。

`source_snapshot/`用于追溯，保留原工程路径，依赖原项目network_ab及早期资产，不是完整可迁移训练环境；部署运行不需要执行这些研究脚本。
'''
    (dest/'README.md').write_text(readme,encoding='utf-8')
    (dest/'发给主项目.md').write_text('请接收用户定稿“像素动画风格”：R39中强版，4×4原生像素，固定B1，10819参数。唯一系数CRC32=2d9aad69，装载20480字节。本包不含R40合并实验和R39更强档。先读README.md与acceptance.json，再执行python -B verify_package.py；同时核对三组残差字段和test_vectors无损RGB888样本。包内附室内外各3秒同源纯B1视频。已有时序退步如实保留，未实板验收；请按主项目现行流程接入，不自动覆盖现行部署。\n',encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>像素动画风格 · R39中强</title><style>body{max-width:1100px;margin:30px auto;padding:0 20px;background:#141922;color:#edf2f8;font:17px/1.7 "Microsoft YaHei",sans-serif}video{width:100%;background:black}a{color:#9dccff}h1{font-size:28px}</style><h1>像素动画风格 · R39中强</h1><p>用户选定版 · 原生4×4 · 纯B1整数输出 · CRC32 2d9aad69</p><p>室内、室外各91帧，约3.03秒。预览为有损编码，逐像素验证请使用包内无损样本。</p>'''
    for scene,title in [('corridor','室内 · 不过曝走廊'),('outdoor','室外 · 街景')]:
        html+=f'<h2>{title}</h2><video controls loop playsinline preload="metadata" src="{scene}.mp4" poster="{scene}_encoded_frame0.png"></video><p><a href="{scene}_frame0.png">查看实际BIN无损帧0</a></p>'
    html+='<p><a href="../README.md">交接说明</a> · <a href="../acceptance.json">用户选定记录</a></p></html>'
    (dest/'previews/review.html').write_text(html,encoding='utf-8')
    assert all(sha(Path(p))==h for p,h in protected_hashes.items())
    save_json(dest/'evidence/protected-artifacts.json',dict(unchanged=True,files=protected_hashes))
    save_json(dest/'evidence/package-audit.json',dict(checkpoint_reexport_byte_identical=True,video_decode_complete=True,videos=video_audits,
        source_manifest_all_hashes_pass=True,coefficient_sha256=expected,excluded_R40=True,source_preview_variant='medium',style_name='像素动画风格'))
    for p in dest.rglob('*.py'):ast.parse(p.read_text('utf-8-sig'))
    save_json(dest/'manifest-sha256.json',dict(schema='sha256-file-manifest-v1',files=[dict(path=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.rglob('*')) if p.is_file()]))
    spec=importlib.util.spec_from_file_location('verify_frozen_pixel',dest/'verify_package.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.verify(dest);delivery.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,arcname=dest.name+'/'+p.relative_to(dest).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={dest.name+'/'+p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
        for p in dest.rglob('*'):
            if p.is_file():assert hashlib.sha256(z.read(dest.name+'/'+p.relative_to(dest).as_posix())).hexdigest()==sha(p)
    result.update(package=str(dest),zip=str(archive),zip_bytes=archive.stat().st_size,zip_sha256=sha(archive),
        zip_all_members_byte_identical=True,checkpoint_reexport_byte_identical=True,protected_artifacts_unchanged=True,disk=disk_guard())
    save_json(delivery/'像素动画风格_R39_交付核验.json',result)
    save_json(OUT/'ACCEPTANCE.json',dict(name='像素动画风格',selected_candidate=src.name,sharpness='medium',accepted_style_package=str(dest),zip=str(archive),
        crc32='2d9aad69',coefficient_sha256=expected,excluded_R40=True,task_completed=True,style_selected_for_packaging=True,full_temporal_acceptance=False,board_accepted=False))
    selection=json.loads((OUT/'SELECTED_PIXEL_CURVE.json').read_text('utf-8'))
    selection.update(style_name='像素动画风格',accepted_style_package=str(dest),delivery_zip=str(archive),packaged_candidate=src.name,packaged_crc32='2d9aad69',
        user_selected_pixel_export=str(src),latest_preview_export=str(src),latest_preview_name=src.name,latest_preview_approved=True,finalized=True,deployment_changed=False)
    save_json(OUT/'SELECTED_PIXEL_CURVE.json',selection)
    print('PIXEL_ANIMATION_FROZEN',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
