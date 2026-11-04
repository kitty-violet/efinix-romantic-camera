"""Verify and freeze all 16 single-block experiments, retaining every R2 file."""
from pathlib import Path
import json,sys,shutil,hashlib,zipfile,subprocess,ast,importlib.util,statistics
import numpy as np
from PIL import Image
import oneblock as o
from oneblock import D,BASE,save_json,sha

P=D/'delivery/S4R1_16款单残差实验包'

def mean(a):return statistics.mean(a)

def main():
    o.setup(20261102);o.guard();entries=json.loads((D/'inventory.json').read_text('utf-8'))['styles'];complete=json.loads((D/'review/complete.json').read_text('utf-8'));assert complete['styles']==16 and complete['frames']==9728
    protected=json.loads((D/'protected-r2-artifacts.json').read_text('utf-8'))['files']
    for p in protected:assert Path(p['path']).stat().st_size==p['bytes'] and sha(p['path'])==p['sha256'],p['path']
    save_json(D/'r2-preservation-verification.json',dict(status='PASS',protected_files=len(protected),all_hashes_unchanged=True,original_packages_not_overwritten=True))
    P.mkdir(parents=True,exist_ok=True);styles=[];diagnostics=[];native_checks=[]
    sources=json.loads((D/'review/source_manifest.json').read_text('utf-8'))['scenes']
    for e in entries:
        k=e['id'];src=D/'models'/k;dest=P/'styles'/k;dest.mkdir(parents=True,exist_ok=True)
        t=json.loads((src/'training.json').read_text('utf-8'));r=json.loads((D/'review/results'/f'{k}.json').read_text('utf-8'));assert t['gradient_steps']==700
        assert t['coefficient_sha256']==r['coefficient_sha256']==sha(src/'coefficients-padded.bin')
        assert t['teacher_sha256']==r['teacher_sha256']==e['source_sha256']==sha(e['source'])
        video=r['video'];assert video['frames']==video['unique_encoded_frames']==video['unique_raw_frames']==608
        assert video['pts_error_seconds']<.00001 and sha(video['path'])==video['sha256']
        if k=='PIXEL_R39':assert video['pixel_unequal_cells']==0
        for name in ['coefficients.bin','coefficients-padded.bin','load_words.hex','graph.json','layout.json','integer-coefficients.npz','reference-vectors.json','selected.pt','training.json']:shutil.copy2(src/name,dest/name)
        _,ps=o.decode(src);wc=sum(p['weight'].size for p in ps.values());bc=sum(p['bias'].size for p in ps.values());assert(wc,bc)==(3808,227)
        save_json(dest/'parameter-count.json',dict(weights_including_zeros=wc,biases_including_zeros=bc,model_parameters=4035,additional_q17_multipliers=227,convolutions=11,residual_blocks=1,nodes=13,macs_per_720p=488908800,layers=[dict(id=n,weights=p['weight'].size,biases=p['bias'].size) for n,p in ps.items()]))
        (P/'common/test-inputs').mkdir(parents=True,exist_ok=True);(dest/'test-vectors').mkdir(exist_ok=True)
        for s in sources:
            scene=s['scene'];shutil.copy2(o.r2.OLD/'acceptance-four-scenes-5s-20261029/originals'/f'{scene}_076.png',P/'common/test-inputs'/f'{scene}.png');shutil.copy2(D/'review/stills'/k/f'{scene}_one.png',dest/'test-vectors'/f'{scene}_output.png')
        (P/'previews').mkdir(exist_ok=True);shutil.copy2(D/'review/stills'/k/'四场景_原图双残差单残差.jpg',P/'previews'/f'{k}.jpg')
        styles.append(dict(**e,directory='styles/'+k,coefficient_sha256=t['coefficient_sha256'],parameters=4035,weights=wc,biases=bc,macs_per_720p=488908800,gradient_steps=700,selected_step=t['selected_step'],visual_accepted=False,new_hardware_implemented=False))
        pairs=[v for scene in r['scenes'] for v in scene['temporal_records']];metrics=[v for scene in r['scenes'] for v in scene['metrics']]
        diagnostics.append(dict(id=k,name=e.get('current_display_name',e['name']),initial_score=t['initial_score'],selected_score=t['selected_score'],selected_step=t['selected_step'],natural_validation_mae=t['selected_natural_mae'],training_nonzero_gradient_steps=t['nonzero_gradient_steps'],four_scene_13_raw_samples_mae=mean([m['mean_absolute_difference'] for m in metrics]),warp_mae_two=mean([p['two']['warp_mae'] for p in pairs]),warp_mae_one=mean([p['one']['warp_mae'] for p in pairs]),features={s['scene']:s.get('special_features') for s in r['scenes']},native_video=video['path'],visual_accepted=False))
        native_checks.append(dict(id=k,source=e['source_sha256'],new=t['coefficient_sha256'],reference=t['independent_oracle'],video=video,all_four_full720_checkpoint_vs_bin=True))
    save_json(P/'catalog.json',dict(model_id='S4R1_8_32_8_V1',accepted_or_retained_style_count=16,styles=styles,hardware_implemented=False,source_two_residual_packages_retained=True))
    save_json(P/'validation.json',dict(software_status='PASS_NUMERICAL_AND_VIDEO_NOT_VISUAL_ACCEPTANCE',native_frames=9728,scene_segments=64,video_files=16,newly_encoded_videos=4,reused_after_all_608_fresh_R1_raw_hashes_match=complete['reused_only_after_all_608_fresh_R1_raw_sha_matched'],styles=native_checks))
    save_json(P/'style-diagnostics.json',dict(records=diagnostics,temporal_pairs_per_style=20,mae_exact_teacher_frames_per_scene=13,style_significance_requires_user_review=True))
    save_json(P/'media-links.json',{r['id']:r['native_video'] for r in diagnostics})
    shutil.copy2(D/'protected-r2-artifacts.json',P/'protected-r2-artifacts.json');shutil.copy2(D/'r2-preservation-verification.json',P/'r2-preservation-verification.json')
    shutil.copy2(D/'training-data.json',P/'common/training-data.json');shutil.copy2(D/'review/source_manifest.json',P/'common/source-manifest.json')
    shutil.copy2(D/'16款单残差视频索引.md',P/'视频验收索引.md')
    for n in ['verify_package.py','infer_package.py']:shutil.copy2(D/'source'/n,P/n)
    (P/'source').mkdir(exist_ok=True)
    for p in (D/'source').glob('*.py'):ast.parse(p.read_text('utf-8'));shutil.copy2(p,P/'source'/p.name)
    lines=['# 全16款：一残差对两残差诊断','','各款均执行700步QAT；12款等价初始化若已满足教师，实际梯度可为0，验证最优会保留第0步，不伪称产生了新的学习收益。所有数值指标仅辅助评估。','','|编号|名称|验证MAE|四景采样MAE|两块→一块补偿帧差|选中步|','|---|---|---:|---:|---:|---:|']
    for r in diagnostics:lines.append(f"|{r['id']}|{r['name']}|{r['natural_validation_mae']:.3f}|{r['four_scene_13_raw_samples_mae']:.3f}|{r['warp_mae_two']:.3f} → {r['warp_mae_one']:.3f}|{r['selected_step']}|")
    lines+=['','四景MAE每景13个实际教师整数帧；补偿帧差每景5对、共20对，仅评价用光流，无推理光流。减少细节本身也可能降低此指标，不能单独作为成功。老片按底色和分亮度噪条统计选择，而非逐点噪条位置；像素锁定硬阈值、配色和4×4中心解码，仍只支持原4×4微格，不重启已否定的大方格风格。','', '本轮老片单残差版的噪条密度明显偏高，时序差异也更大，暂不建议替换双残差版。以下覆盖率是中间噪条门控检测，不等同于肉眼可见噪条比例；可见正负噪条比例另见style-diagnostics.json。','', '|场景|两残差门控覆盖率|单残差门控覆盖率|两残差横条数|单残差横条数|','|---|---:|---:|---:|---:|']
    film=next(r for r in diagnostics if r['id']=='OLD_FILM')
    for scene,feature in film['features'].items():
        a,b=feature['two'],feature['one'];lines.append(f"|{scene}|{a['bar_coverage']:.3%}|{b['bar_coverage']:.3%}|{a['thin_bar_count']}|{b['thin_bar_count']}|")
    diag='\n'.join(lines);(D/'单残差风格与时序诊断.md').write_text(diag,encoding='utf-8');(P/'风格与时序诊断.md').write_text(diag,encoding='utf-8')
    doc='''# 交给主项目：16款单残差独立实验

本轮用户明确要求全部16款都用一个残差块训练一次看结果，同时保留所有双残差权重包。575个R2权重/检查点/NPZ/HEX/ZIP在开始和结束均核对SHA，全部不变。本包不覆盖R2，不修改原采用B1包、硬件、Flash、部署基点或其他工作流。

新图S4R1_8_32_8_V1仍为RGB720p、stem8、stride4→32、一个32通道残差块、PW→8、nearest4、RGB。11卷积13节点，3808权重+227偏置=4035参数，488908800MAC/720p。相对双块6787参数/640051200MAC，参数少40.55%，MAC少23.61%。不把MAC比例当实板速度；原150MHz账本条件估计约23到27FPS只是历史情景，新图未实现硬件/调度/布局布线/时序或实板验证。

新S4R1PKG1头64字节，LE32(version1,width1280,height720,conv11,res1,RGB3)，其后为规范graph JSON SHA256。按layout保存OIHW INT8权重、LE32偏置和Q17，8字节对齐，末尾一组残差字段，补零到8192字节、1024LE64字。有效长度见layout。与旧R2 16KB/15卷积不兼容，不能只换BIN就用于旧执行器。

数值为RGB uint8减128、replicate边界、INT32饱和后Q17 half-away-from-zero舍入、INT8饱和/按图ReLU，输出加128。残差继承main_scaled=false、multiplier=1、post_multiplier=131071，不能擅自按普通main+skip代替。

教师为当前实际R2系数：初版14款加老片/像素返修2款，均只读。全部每款一次700步QAT，包含初始系数择优。9款第二块为恒等；B03将通道位移移回已有DW；R8-06/07用整数矩阵重排合并弱边合成与阈值，将配色放解码PW。若初始化已与教师逐值相同，训练梯度可为0，选第0步，报告明确记录，不伪称这些款从梯度得到了改善。

油画/水墨实际QAT；水墨保持RGB一致。像素冻结最后硬阈值与配色，空间权重保持整数DC、高分辨DW中心恒等，4×4格严格同色，但不表示能够产生8×8/16×16空间相位。先前被否决的大方格路线不重启。老片用有限Laplacian窗口省掉两级min，保留横向延展，按底色与分亮度的正负噪条统计监督。实际单残差视频噪条偏密，时序差异更大，暂不建议替换双残差版；它是已完成一次训练的实验结果，未通过视觉验收。四景覆盖率和补偿帧差见风格与时序诊断.md，原双残差老片包保留。

16款均独立整数参考、验证集检查点/BIN、四景整720p检查点/BIN核对，包内独立执行另做同源输入复演。每款一段608帧约20.245秒，依次室内/室外/人群/水面各152帧约5.061秒；不重复保存同一电影的四个分段，避免占盘。所有输入帧SHA/PTS与原同组视频绑定，新帧实际执行导出BIN；无推理光流/历史帧/滤镜/额外生成器。

有变化的4款新MP4用已获用户确认的BT601/SMPTE170M limited YUV444 CRF0、sRGB传递，无色度子采样；不再用曾显示粉绿的RGB/GBR H264。视频有点色彩转换舍入，不能称RGB字节精确；逐帧探针和12个整帧色差核验，像素逐帧所有格核验。PNG/test-vectors为实际整数输出精确样本。光流仅评价每景5对相邻帧，不用抹细节/降帧/重复插入掩盖闪动。

12款整数等价的异色/漫画/B03：本轮R1 BIN的每个608帧均实际重新执行，所有原始RGB SHA逐帧与R2旧记录相同后，直接复用对应同源R2 MP4作视觉呈现，不另存重复电影。旧视频保留原有H264/YUV420编码及其舍入/压缩误差，不冒称新编码或RGB精确。验证字段reused_exact_equivalent_r2_video/all_608_raw_sha_match_R2明确记录；这里省掉的是重复视频编码和存储，模型执行及逐帧等价检查未省。

首次全16款无损新编码触发自身3GiB输出保护，未完成MP4无moov均不可播放，已只清理这些本轮无效缓存3273130752字节；训练模型和所有R2资产未动。随后按上述4款新编码+12款逐帧等价复用完成正式验收，不把被中断前缀当完成。

视频只放独立review目录，入口见视频验收索引.md和media-links.json，不塞进ZIP重复占盘。原R2对照视频完整保留。红色R8-06叫热成像，R8-07叫漫画；只按用户命名，不作温度推断。

复核：python -B verify_package.py --full-reference（标准库），python -B infer_package.py --style-root styles/A1 --input common/test-inputs/indoor.png --output check.png（使用已有torch/numpy/Pillow）。source仅原工程追溯，独立infer/verify可单独执行。新风格效果待用户看视频逐款评估，不以软件PASS当视觉或实板放行。
'''
    (D/'交给主项目_16款单残差实验.md').write_text(doc,encoding='utf-8');(P/'交给主项目.md').write_text(doc,encoding='utf-8')
    manifest={str(p.relative_to(P)).replace('\\','/'):sha(p) for p in P.rglob('*') if p.is_file() and p.name!='manifest-sha256.json'};save_json(P/'manifest-sha256.json',manifest)
    proc=subprocess.run([sys.executable,'-B',str(P/'verify_package.py'),'--root',str(P),'--full-reference'],capture_output=True,text=True,encoding='utf-8',timeout=240);assert proc.returncode==0,(proc.stdout,proc.stderr);portable=json.loads(proc.stdout);assert portable['style_count']==16;save_json(D/'portable-verification.json',portable)
    sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('r1_standalone',P/'infer_package.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    count=0
    for e in entries:
        k=e['id'];engine=mod.Engine(P/'styles'/k)
        for scene in ['indoor','outdoor','crowd','water']:
            a=np.array(Image.open(P/'common/test-inputs'/f'{scene}.png'));expected=np.array(Image.open(P/'styles'/k/'test-vectors'/f'{scene}_output.png'));assert np.array_equal(engine(a),expected),(k,scene);count+=a.size
    save_json(D/'standalone-full720-verification.json',dict(status='PASS',styles=16,cases=64,scalar_comparisons=count,all_pixels_equal=True))
    zpath=D/'delivery/S4R1_16款单残差实验包_20261102.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(P.rglob('*')):
            if p.is_file():z.write(p,str(Path(P.name)/p.relative_to(P)))
    with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
    save_json(D/'delivery-verification.json',dict(status='PASS_NUMERICAL_AND_VIDEO_NOT_VISUAL_ACCEPTANCE',zip=str(zpath),zip_bytes=zpath.stat().st_size,zip_sha256=sha(zpath),styles=16,parameters_each=4035,gradient_steps_each=700,native_reels_available=16,newly_encoded_reels=4,reused_only_after_all_fresh_R1_raw_hashes_match=12,five_second_scene_segments=64,native_frames_actually_executed_from_R1_BIN=9728,standalone_full720_cases=64,standalone_scalar_comparisons=count,protected_R2_files=len(protected),all_protected_R2_hashes_unchanged=True,hardware_modified=False))
    print('SINGLE_BLOCK_16_STYLE_DELIVERY_COMPLETE',zpath,flush=True)

if __name__=='__main__':main()
