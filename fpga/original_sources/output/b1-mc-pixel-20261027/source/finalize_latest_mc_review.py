"""Check delivered videos, local data policy and unchanged protected weights."""
from core import *
import ast

def main():
    setup(271164);d=OUT/'mc-latest-r25-review';summary=json.loads((d/'summary.json').read_text('utf-8'));assert summary['complete']
    qa=[];out=d/'encoded-video-qa';out.mkdir(exist_ok=True)
    for c in summary['clips']:
        for kind in ['current_r25','comparison']:
            path=Path(c['videos'][kind]['path']);cap=cv2.VideoCapture(str(path));hashes=[];sizes=set();index=0
            while True:
                ok,bgr=cap.read()
                if not ok:break
                sizes.add((bgr.shape[1],bgr.shape[0]));hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest())
                if kind=='current_r25' and index in [0,30,60]:Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(out/f'{c["scene"]}_decoded_{index:04d}.png')
                index+=1
            cap.release();expected=(1280,720) if kind=='current_r25' else (1280,788)
            assert index==c['frames'] and sizes=={expected} and len(set(hashes))==index
            qa.append(dict(scene=c['scene'],kind=kind,frames=index,decoded_unique_frames=len(set(hashes)),size=list(expected),sha256=sha(path)))
    old_files=[d/f'station_{stem}' for stem in ['comparison_0000.jpg','comparison_0030.jpg','native_0000.png','native_0030.png']]
    for p in old_files:
        if p.exists():assert p.is_file() and p.resolve().parent==d.resolve();p.unlink()
    checks=[]
    for p,expected in [(WORK/'accepted/B1-A-20261008/coefficients-padded.bin','ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523'),
        (WORK/'accepted/B1-水墨风-20261027/coefficients-padded.bin','d81910c309e14e4ecd965e39554cd13209f061c0e7e0b19c71e28c470d81b445'),
        (DEPLOY,'b62919cf6973530bf3d419e4c509c47321e8b42207384c728d1b9c2b613ada0f')]:
        actual=sha(p);assert actual==expected;checks.append(dict(path=str(p),sha256=actual,unchanged=True))
    count=0
    for p in (OUT/'source').glob('*.py'):ast.parse(p.read_text('utf-8-sig'),filename=str(p));count+=1
    stats=[dict(scene=c['scene'],frames=c['frames'],seconds=c['seconds'],temporal=c['temporal']) for c in summary['clips']]
    save_json(d/'verification.json',dict(encoded_videos=qa,source_AST_pass=count,protected_coefficients=checks,
        cache_empty=not list((OUT/'cache').iterdir()),disk=disk_guard(),subjective_full_motion_review_performed=False))
    readme='''# 当前 MC R25：实际视频与问题定位

本轮没有重新训练MC；这些视频来自既有R25的已核验纯B1整数系数（CRC4c66dbd6）。室内改为住宅走廊，旧车站大厅退出主对比。所有模型使用相同原始帧、固定HLG解码及3:2缩放，无外置风格滤镜、推理光流或历史帧。

- corridor_current_r25.mp4：新室内，原生1280×720，3.03秒。
- outdoor_current_r25.mp4：室外，原生1280×720，3.03秒。
- corridor_comparison.mp4 / outdoor_comparison.mp4：左上原图、右上早期R9、左下前版R20、右下当前R25。面板只缩小展示，数值推理由原生720p完成。

视频完整解码检查了尺寸、帧数、逐帧唯一性和原始PTS。光流仅用于离线评分。source/R9/R20/R25的时序数据在对应report和summary中，不能把低误差等同于MC风格成功；本轮仍未达到大立方体场景目标。

## 根因与修正目标

1. 旧体积目标以五档亮度分别做固定12/16像素挤出，产生的主要是薄片浮雕。同一物体表面的亮度变化会被当作不同体块。
2. R18到R25主要训练已有通道的“允许挤出”支持场，不能据此认为学会了统一立方体、共享平面或随物体尺寸确定纵深。
3. R25封闭了支持场的绝对RGB捷径，但主F/H/D形状路径仍依赖亮度。给同一方块正面增加一块平面浅色，实际B1仍会在内部生成额外体积边。
4. mc-complete-cube-target-audit明确改为一个已知物体遮罩、同一纵深向量、完整三个面的教师示意。它不是运行时分割器，也不是已经学会的学生输出。
5. 超出感受野的全物体搬移不能仅靠训练补偿；audit.json保留了大128像素正面/64像素纵深的相同77×77局部输入、不同目标输出的具体反例。此反例不否定小尺度完整方块的可行性。

像素路线恢复exports/pixel-user-curve-A为后续起点。R23已被用户否决；后续只清理旧A色层的杂色/毛边，不再扩张亮度区域改变块面分布。

数据质量策略在data-quality/selection.json；仅影响本工作流未来训练取样，不修改原素材、历史权重或其他工作流。数据筛查是近白区域风险代理，未把普通白墙等同于传感器过曝。
'''
    (d/'README.md').write_text(readme,encoding='utf-8')
    save_json(d/'metrics-overview.json',stats)
    files=[p for p in d.rglob('*') if p.is_file() and p.name!='manifest-sha256.json']
    save_json(d/'manifest-sha256.json',{str(p.relative_to(d)):sha(p) for p in files})
    print('LATEST_MC_REVIEW_VERIFIED',json.dumps(dict(videos=qa,metrics=stats,disk=disk_guard())),flush=True)

if __name__=='__main__':main()
