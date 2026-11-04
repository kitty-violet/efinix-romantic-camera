"""Freeze sixteen new coefficient packs and their actual four-scene evidence."""
from pathlib import Path
import json,shutil,zipfile,subprocess,sys,time,ast,os
import library as l
from library import D,ROOT,OLD,RECON,save_json,sha

P=D/'delivery'/'S4R2_16款新结构风格包'

def link_or_copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert sha(src)==sha(dest);return
    try:os.link(src,dest)
    except OSError:shutil.copy2(src,dest)

def main():
    l.guard();entries=[e for e in l.read_inventory() if e['id']!='BW_LINE'];assert len(entries)==16
    sources=json.loads((D/'review/source_manifest.json').read_text('utf-8'))['scenes'];styles=[];checks=[];media={}
    P.mkdir(parents=True,exist_ok=True)
    for e in entries:
        key=e['id'];src=D/'models'/key;dest=P/'styles'/key;dest.mkdir(parents=True,exist_ok=True)
        training=json.loads((src/'training.json').read_text('utf-8'));assert sha(e['source'])==e['source_sha256']
        reports=[json.loads((D/'review/native'/key/(s['scene']+'.json')).read_text('utf-8')) for s in sources]
        current_sha=sha(src/'coefficients-padded.bin');assert current_sha==training['coefficient_sha256']
        for r,s in zip(reports,sources):
            assert r['coefficient_sha256']==current_sha and r['teacher_sha256']==e['source_sha256']
            assert r['input_sha256']==s['raw_input_frame_sha256'] and r['pts']==s['pts']
            assert r['native_video']['frames']==r['native_video']['unique_encoded_frames']==r['unique_new_raw_frames']==152
            assert r['no_repeat_inserted'] and r['no_frame_drop'] and r['new_pixels_from_delivered_bin']
            assert sha(Path(r['native_video']['path']))==r['native_video']['sha256']
        static=json.loads((D/'review/static'/key/'verification.json').read_text('utf-8'))
        assert static['coefficient_sha256']==current_sha and static['full720_checkpoint_vs_bin_exact']
        for name in ['coefficients.bin','coefficients-padded.bin','load_words.hex','graph.json','layout.json','integer-coefficients.npz','training.json','reference-vectors.json']:
            shutil.copy2(src/name,dest/name)
        ck=src/('selected.pt' if (src/'selected.pt').exists() else 'qat.pt');shutil.copy2(ck,dest/'selected.pt')
        shutil.copy2(D/'review/static'/key/'verification.json',dest/'static-verification.json')
        desc,decoded=l.decode(src);layer_counts=[]
        for name,values in decoded.items():layer_counts.append(dict(node=name,weights=int(values['weight'].size),biases=int(values['bias'].size),q17_multipliers=int(values['multiplier'].size),nonzero_weights=int((values['weight']!=0).sum()),nonzero_biases=int((values['bias']!=0).sum())))
        assert sum(x['weights'] for x in layer_counts)==6432 and sum(x['biases'] for x in layer_counts)==355
        save_json(dest/'parameter-count.json',dict(weights_including_zeros=6432,biases_including_zeros=355,model_parameters=6787,additional_q17_multipliers=355,additional_residual_multipliers=4,additional_residual_branch_flags=2,layers=layer_counts))
        link_or_copy(D/'review/static'/key/'四场景对比.jpg',P/'previews'/key/'四场景对比.jpg')
        for s in sources:
            n=s['frames']//2;scene=s['scene']
            source_png=OLD/'acceptance-four-scenes-5s-20261029/originals'/f'{scene}_{n:03d}.png'
            link_or_copy(source_png,P/'common/test-inputs'/f'{scene}.png')
            link_or_copy(D/'review/stills'/key/f'{scene}_{n:03d}_new.png',dest/'test-vectors'/f'{scene}_output.png')
        row=dict(**e,directory='styles/'+key,coefficient_sha256=current_sha,parameters=6787,weights=6432,biases=355,macs_per_720p=640051200,coefficient_bytes=16384,effective_bytes=json.loads((dest/'layout.json').read_text('utf-8'))['effective_bytes'],padded_crc32=json.loads((dest/'layout.json').read_text('utf-8'))['crc32'],new_style_user_accepted=False,new_hardware_implemented=False,numerical_checks_pass=True,detail_merge=training.get('detail_merge_complete',False),local_clipped_merge=training.get('local_clipped_merge'),four_scene_native_frames=608)
        styles.append(row);checks.append(dict(id=key,independent_integer_oracle=training['independent_oracle'],native_video_frames=sum(r['frames'] for r in reports),all_native_frames_unique=True,full720_checkpoint_vs_bin_exact=True,static_records=static['records'],temporal_records={r['scene']:r['temporal_records'] for r in reports},exact_raw_comparison_sampling={r['scene']:r['exact_raw_metric_frames'] for r in reports}))
        reel=D/'review/reels'/f'{key}_新结构_四场景各5秒.mp4';comparison=D/'review/reels'/f'{key}_原图旧版新版_四场景各5秒.mp4'
        assert reel.exists() and comparison.exists()
        media[key]=dict(native_reel=str(reel),native_reel_sha256=sha(reel),comparison_reel=str(comparison),comparison_reel_sha256=sha(comparison),clips=[r['native_video'] for r in reports])
    catalog=dict(model_id='S4R2_8_32_8_V1',date='2026-11-02',accepted_or_retained_style_count=16,styles=styles,original_source_packages_unchanged=True,original_fixed_B1_bin_compatible=False,hardware_implemented=False,board_test=False,source_disjoint_for_this_training=True,not_a_new_global_blind_test=True,black_white_teacher_reference_not_in_sixteen=True,demonstration_groups_only=[['A1','INK','OLD_FILM','PIXEL_R39','W-T256'],['R8-06','B03','R8-07','PI02','PI03'],['PI04','NI01','NI02','NI03','NI04'],['NI05']])
    save_json(P/'catalog.json',catalog);save_json(P/'numerical-and-video-verification.json',dict(status='PASS_SOFTWARE_NUMERICAL_AND_MEDIA_NOT_VISUAL_ACCEPTANCE',styles=checks,native_clips=64,native_frames=9728,hardware_test=False))
    save_json(P/'media-links.json',media);save_json(P/'common/source-manifest.json',dict(scenes=sources))
    shutil.copy2(D/'models/A1/graph.json',P/'common/model-contract.json')
    shutil.copy2(D/'constant-controls.json',P/'common/constant-controls.json')
    shutil.copy2(D/'style-and-temporal-summary.json',P/'common/style-and-temporal-summary.json')
    shutil.copy2(D/'风格与时序诊断.md',P/'风格与时序诊断.md')
    for name in ['verify_package.py','infer_package.py']:shutil.copy2(D/'source'/name,P/name)
    for src in (D/'source').glob('*.py'):
        (P/'source').mkdir(exist_ok=True);shutil.copy2(src,P/'source'/src.name);ast.parse(src.read_text('utf-8'))
    shutil.copy2(RECON/'source/reconstruct.py',P/'source/reconstruct.py')
    write_docs(catalog,media)
    manifest={str(p.relative_to(P)).replace('\\','/'):sha(p) for p in P.rglob('*') if p.is_file() and p.name!='manifest-sha256.json'}
    save_json(P/'manifest-sha256.json',manifest)
    proc=subprocess.run([sys.executable,'-B',str(P/'verify_package.py'),'--root',str(P),'--full-reference'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    assert proc.returncode==0,(proc.stdout,proc.stderr)
    portable=json.loads(proc.stdout);assert portable['status']=='PASS' and portable['style_count']==16
    save_json(D/'portable-package-verification.json',portable)
    # Standalone inference uses only the delivered BIN/graph/layout and the
    # environment's torch/numpy/Pillow, not the project B1 decoder.
    import importlib.util
    spec=importlib.util.spec_from_file_location('standalone_s4r2',P/'infer_package.py');mod=importlib.util.module_from_spec(spec);sys.path.insert(0,str(P));spec.loader.exec_module(mod)
    from PIL import Image
    import numpy as np
    x=np.array(Image.open(P/'common/test-inputs/indoor.png'));scalar=0
    for row in styles:
        y=mod.Engine(P/row['directory'])(x);expected=np.array(Image.open(P/row['directory']/'test-vectors/indoor_output.png'))
        assert np.array_equal(y,expected),row['id'];scalar+=expected.size
    save_json(D/'standalone-full720-verification.json',dict(status='PASS',style_count=16,scalar_comparisons=scalar,standalone_bin_vs_saved_pixels_exact=True))
    outzip=D/'delivery/新结构S4R2_16款风格包_20261102.zip'
    with zipfile.ZipFile(outzip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,str(Path(P.name)/p.relative_to(P)))
    with zipfile.ZipFile(outzip) as z:assert z.testzip() is None
    save_json(D/'delivery/交付核验.json',dict(status='PASS_16_STYLE_PACKS_AND_64_NATIVE_VIDEOS',zip=str(outzip),zip_bytes=outzip.stat().st_size,zip_sha256=sha(outzip),styles=16,independent_oracle_node_scalars=sum(x['independent_integer_oracle']['scalar_comparisons'] for x in checks),standalone_full720_scalars=scalar,native_clips=64,native_frames=9728,metadata_files=len(manifest),source_weights_unchanged=True,new_hardware_implemented=False,new_visual_acceptance=False))
    print('S4R2_16_STYLE_DELIVERY_COMPLETE',outzip.stat().st_size,sha(outzip),flush=True)

def write_docs(catalog,media):
    lines=['# 新结构16款风格：验收视频索引','','所有新结果来自交付BIN的整数规则执行。每款依次为室内、室外、人群、水面，各约5.061秒，合计约20.245秒；原生720p，每款608个唯一输出帧。','',
      '三列对照为原图／旧版／新结构。后12款的旧版画面复用同源、同PTS和系数SHA已绑定的既有整数视频（有一次旧H264解码）；源帧哈希全一致，关键帧及光流评价相邻帧重新跑原BIN逐值核对。新结果每帧均重新执行交付BIN，原生视频没有额外滤镜、重复或跳帧。','',
      '|编号|名称与当前别名|新结构原生20秒|原图／旧版／新版对照20秒|四景静态|','|---|---|---|---|---|']
    for row in catalog['styles']:
        key=row['id'];name=row.get('current_display_name',row['name']);m=media[key]
        still=D/'review/static'/key/'四场景对比.jpg'
        lines.append(f"|{key}|{name}|[播放]({Path(m['native_reel']).as_posix()})|[对照]({Path(m['comparison_reel']).as_posix()})|[图片]({still.as_posix()})|")
    lines+=['','黑白线条原资产是FP32教师／研究网络的输出，旧固定B1未实现；其新结构尝试属于额外实验，不在这16款已采用／备用风格迁移包内。','']
    text='\n'.join(lines);(D/'16款视频验收索引.md').write_text(text,encoding='utf-8');(P/'验收视频索引.md').write_text(text,encoding='utf-8')
    description='''# 交给主项目：S4R2新结构16款风格试验包

用户2026-11-02授权，将全部16款已采用／保留备用的原B1风格迁移到新结构；此前淘汰的R4-01、R4-09、R8-04、P4和其他组合不恢复。黑白线条旧资产只是教师／研究网络，旧B1未实现，额外尝试另列，禁止算进这16款成功迁移。

## 新图和参数

S4R2_8_32_8_V1：RGB → stem3×3 3→8 → DW3×3 stride4 → PW8→32 → 两个32通道残差块（各DW/PW、DW/PW）→ DW3×3 → PW32→8 → nearest×4 → DW3×3 → RGB1×1。15卷积、18节点。原生1280×720，6432权重+355偏置=6787参数；另有355个逐输出通道Q17乘数和两组残差字段。720p MAC为640051200，较原966758400少33.79%；这不是实板速度。

风格版残差对齐继承原包当前语义：两组main_scaled=false、multiplier=1、post_multiplier=131071，当前INT8域等价保留主支。前轮原图重建原型使用普通main+skip，两者数值设置不同；不得混用检查点。每款实际字段在graph.json和BIN中，通用执行按字段实现，不能只凭名称假定普通相加。

## 新二进制格式

这是新算子软件系数包，**不能直接替换现行固定B1的20KB装载文件或compute/dma程序**。S4R2PKG1新头64字节：magic8，LE32(version=1,width=1280,height=720,conv=15,res=2,RGB=3)，其后32字节为规范graph JSON的SHA256。有效9368字节，尾零到16384字节（2048个LE64字）。每层依layout.json保存OIHW INT8权重、LE32 INT32偏置与Q17乘数，字段8字节对齐；末尾两组3×LE32残差字段。

INT8输入RGB减128，输出INT8加128；3×3采用replicate边界，卷积偏置加后饱和INT32，Q17逐通道乘法，nearest-half-away-from-zero舍入，饱和INT8，再按图ReLU。不得以普通四舍五入、全局缩放或额外滤镜代替。source/及独立infer_package.py提供真实执行代码。每款parameter-count.json逐层核对6432权重和355偏置，零系数仍计入总参数；固定图不会仅因零权重自动减少计算。

## 训练和碎边归并

原16包保持只读，作为教师，以同源输出制靶。32张训练、3张验证按源视频隔离；四段回归视频未进入本次拟合，但以前展示过，不能称全项目新盲测。每款先实际FP32/QAT训练，再按验证选择；个别空间校正没有胜过初始化时，保留更优初始化，见各training.json，不伪称所有选中系数均来自最后梯度步。

漫画、异色等12款按用户新要求减少细碎边缘：先以训练专用边缘保留简化输入送原教师制靶；再在既有DW/PW/ReLU系数内实现局部弱边归并。若local_clipped_merge字段存在，最终为已训练前端加系数构造：mean+ReLU(delta−T)−ReLU(−delta−T)，强边差值超过T部分保留；均匀色块保持。R8类将阈值／配色放在归并后，配色后不混合不同色块。没有添加推理bilateral、颜色LUT、分割、光流、历史帧或输出滤镜。最后的系数构造是明确记录的构造步骤，不冒称新增梯度训练。

像素动画仅按R39中强原包迁移，最后高分辨DW限制中心项，所有输出4×4块内逐像素同色；没有混入R40。水墨保持RGB三个通道完全一致。新图的细节与时序仍需用户逐款接受，不自动覆盖旧定稿。

12款漫画／异色系的256级灰度和64组常量RGB与旧包逐值相同。其余4款是近似迁移：水墨灰阶相同；油画灰阶最大单通道差27、平均0.723，老片最大7、平均1.275，R39最大24、平均0.303，后者均匀灰最大通道差1。油画与像素的较大最大值主要在离散灰阶断点附近，不声称原曲线完全不变。common/constant-controls.json保存全部诊断数据；这些不等同复杂画面全像素相同。

## 实际核验

各款独立Python整数参考比较全部小张量节点、QAT/BIN验证块相等、四景整720p检查点/BIN相等；全16款独立便携BIN执行与保存720p样本逐像素相等。BIN/HEX/CRC/SHA/补零/布局与ZIP CRC均实际核验。test-vectors对应common/test-inputs的四张原生720p图。

64段原生视频各152帧、无跳帧无插入重复帧、原PTS、每段约5.061秒；每款另有20秒原生汇总和三列对照。视频在本地独立review目录，ZIP含模型、检查点、测试图、指标和核验代码，不重复装入大视频。实际路径见验收视频索引.md及media-links.json。光流只用于评价；定点评价不证明任意视频无闪动。MP4播放帧率不是硬件AI FPS。

回归输入沿用既有手机素材固定HLG→sRGB减1档及1080→720整数面积缩放，只是这组素材的输入条件，不指令主项目修改相机ISP、曝光或部署LUT。

## 复核与接入

用已有Python执行：
```powershell
python -B verify_package.py --full-reference
python -B infer_package.py --style-root styles/A1 --input common/test-inputs/indoor.png --output check.png --device cuda
```
第一条只需标准库；第二条需要已存在的torch/numpy/Pillow环境，不自动安装。新核对程序不读写板卡。

主项目先读取catalog.json、common/model-contract.json、每款layout.json和上述执行语义，再决定适配新stride4、nearest4、两残差调度及新的系数读取格式。原程序只有旧stride2/nearest2路径，不能只换权重便宣称支持。此包未改RTL、Flash、发布目录或现行部署，没有综合／布局布线／时序／实板验证；不得直接用于烧写。

约23FPS是先前旧150MHz账本保留未归因开销的条件估计，不是新硬件实测。不能和其他工作流的约6.628ms优化直接相加，收益可能重叠。继续保持主项目自己的最新实现、完整功能和监督上板流程；本交接不指令回退或替换其当前实现。

其他对话完整读取本说明即可恢复此包边界。只更新接收对话自己的project-memory，不改本迁移流记录，不使用多智能体。新风格仍待用户视频验收；catalog中的旧源采用状态不能冒称新包已接受。
'''
    (P/'交给主项目.md').write_text(description,encoding='utf-8')
    (D/'交给主项目_新结构16款风格.md').write_text(description+'\n\n包目录：'+str(P)+'\n',encoding='utf-8')

if __name__=='__main__':main()
