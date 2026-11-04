"""Bind the exact user-approved still to exported coefficients and full review."""
from core import *
import html,zlib

@torch.inference_mode()
def main():
    setup();d=OUT/'exports/flat-r3-user-selected';r=OUT/'approved-pixel-review'
    v=json.loads((d/'verification.json').read_text('utf-8'))
    summary=json.loads((r/'summary.json').read_text('utf-8'))
    assert summary['all_six_complete'] and len(summary['clips'])==6
    raw=(d/'coefficients-padded.bin').read_bytes()
    assert raw==(OUT/'r3-shared-palette/flat-large/initial.bin').read_bytes()
    assert len(raw)==20480 and not any(raw[17792:])
    engine=IntegerEngine(raw);n=load_model(d/'selected.pt')
    counts=dict(weights=sum(x.numel() for x in n.weight.values()),biases=sum(x.numel() for x in n.bias.values()))
    counts['total']=sum(counts.values());assert counts==dict(weights=10256,biases=563,total=10819)
    previews={}
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');o=rgb(engine(tensor(a)[None]*128)/128)
        original=read(OUT/f'r3-shared-palette/flat-large/{scene}_student.png')
        assert np.array_equal(o,original)
        p=d/f'{scene}_approved.png';Image.fromarray(o).save(p)
        previews[scene]=dict(path=str(p),sha256=sha(p),raw_rgb_sha256=hashlib.sha256(o.tobytes()).hexdigest(),exact_original_still_match=True)
    previous=json.loads((OUT/'r3-r4-review/summary.json').read_text('utf-8'))
    previous={c['id']:c for c in previous['clips']};results=[]
    for c in summary['clips']:
        assert c['no_frame_drop_or_duplication'] and not c['post_filter'] and not c['inference_flow']
        assert all(x['frames']==c['frames'] and x['max_pts_error_seconds']<.001 for x in c['video'].values())
        assert c['raw_unique_frames']['flat_r3']==c['frames']
        old=previous[c['id']]['temporal']['flat_previous']['warp_mae'];now=c['temporal']['flat_r3']['warp_mae']
        results.append(dict(id=c['id'],frames=c['frames'],seconds=c['duration_seconds'],warp_mae=now,previous_pixel_warp_mae=old,
            improved=now<old,coarse_luma_correlations=[x['flat_r3']['coarse_luma_correlation'] for x in c['content']],
            pure_video=c['video']['flat_r3']['path'],comparison_video=c['video']['comparison']['path']))
    protected={
      WORK/'accepted/B1-A-20261008/coefficients-padded.bin':'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523',
      WORK/'accepted/B1-水墨风-20261027/coefficients-padded.bin':'d81910c309e14e4ecd965e39554cd13209f061c0e7e0b19c71e28c470d81b445',
      DEPLOY:'b62919cf6973530bf3d419e4c509c47321e8b42207384c728d1b9c2b613ada0f'}
    assert all(sha(p)==s for p,s in protected.items())
    sources=d/'source-snapshot';sources.mkdir(exist_ok=True)
    for name in ['core.py','r3_palette_blocks.py','train_pixel.py','pixel_targets.py','verify_export.py','evaluate_r3_r4.py','evaluation_utils.py','finalize_approved_pixel.py']:
        shutil.copy2(OUT/'source'/name,sources/name)
    binding=json.loads((r/'binding.json').read_text('utf-8'))
    assert binding['coefficients']['flat_r3']['sha256']==sha(d/'coefficients-padded.bin')
    status=dict(name='大色块像素风 R3',user_visual_acceptance=True,
        acceptance_scope='2026-10-27用户先确认所指版本为flat-large/outdoor_student.png，随后回复挺ok的；只认可纯像素原版外观',
        stage='FROZEN_OFFLINE_REFERENCE',coefficient_sha256=sha(d/'coefficients-padded.bin'),crc32_padded=f'{zlib.crc32(raw):08x}',
        counts=counts,source='constructed ordinary B1 coefficients; later constrained-QAT candidate kept separately',
        actual_B1_coefficients=True,strict_6x6_grid=False,native_interior_grid=4,semantic_3D_MC=False,
        no_extra_inference_processing=True,numerical_reference_verified=True,small_integer_comparisons=v['small_oracle_scalar_comparisons'],
        continuous_frame_pts_protocol_pass=True,all_six_warp_improve_over_previous_pixel=all(x['improved'] for x in results),
        high_frequency_caveat='像素边界运动存在网格步进；warp下降不单独等于无可见闪动，不能以色块简化掩盖时序风险',
        board_validated=False,protected_sha256_unchanged={str(p):s for p,s in protected.items()},previews=previews,clips=results,disk=disk_guard())
    save_json(d/'approved-reference.json',status)
    lines=['# 大色块像素风 R3：用户认可原版','',
      '保存的是用户看到并认可的 `flat-large/outdoor_student.png` 同源系数。两张认可版图片已用导出的 BIN 重算，并与原图逐像素比对一致。',
      '','## 接入资料','',
      '- `coefficients-padded.bin`：20480字节，CRC32 `'+status['crc32_padded']+'`。有效系数17792字节，末尾2688字节为零。',
      '- `coefficients.bin`：有效部分；`load_words.hex`：同一镜像的2560个64位小端装载字。',
      '- `residual_encoding.json`：必须与系数配套交给现有装载/程序生成工具；不应只换BIN并沿用别版残差字段。',
      '- `contract/`：固定B1合同；`selected.pt`：本版精确检查点。',
      '- `verification.json`：独立整数核验原始记录；`approved-reference.json`：外观认可范围、视频绑定与指标。',
      '- `source-snapshot/`：保存本次核验源码快照，依赖现有E:\\fpga工程与隔离环境，不是脱离项目的独立安装包。',
      '',f"参数：{counts['total']} = {counts['weights']}权重 + {counts['biases']}偏置。24个卷积，29个节点，固定1280×720。",'',
      '本版用现有卷积、ReLU、饱和、残差和最近邻上采样表达共享明暗分区、色块合并与少量明暗调整；没有额外CPU像素滤镜。它属于大色块纯像素路线，不是全局MC立体重建，也不声称是严格固定6×6网格。',
      '','## 视频验证','',
      '同组原视频采用相同的固定输入转换；室外和室内各约3秒，另外四类场景各约1秒。原始连续帧和PTS全部保留，未插帧、重复帧或降低帧率。光流仅用于离线测量。MP4为纯B1输出的预览编码，指标在编码前的整数RGB上计算。','',
      '| 片段 | 帧数 | 本版warp MAE | 上轮像素warp MAE |','|---|---:|---:|---:|']
    lines += [f"| {c['id']} | {c['frames']} | {c['warp_mae']:.3f} | {c['previous_pixel_warp_mae']:.3f} |" for c in results]
    lines+=['','较低warp误差需结合大体结构、边界与实际播放判断；像素边界仍会随运动发生网格步进。六组记录和视频位于 `../../approved-pixel-review/`。',
      '','本目录仅冻结离线参考；未更换现行部署基点，未修改硬件、A、水墨、V14或Flash，未做板级验收。',
      '','SHA256：`'+status['coefficient_sha256']+'`。','']
    (d/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    blocks=[]
    for c in results:
        pure=Path(c['pure_video']).relative_to(OUT).as_posix();comp=Path(c['comparison_video']).relative_to(OUT).as_posix()
        blocks.append(f'<section><h2>{html.escape(c["id"])} · {c["seconds"]:.2f}s</h2><video controls preload="metadata" src="{html.escape(pure)}"></video><p><a href="{html.escape(comp)}">同帧对照：原图 / 旧V14 / 认可原版 / R3训练对照</a></p></section>')
    (OUT/'approved-pixel.html').write_text('<!doctype html><meta charset="utf-8"><title>大色块像素风R3 · 认可原版</title><style>body{background:#131923;color:#eef3f8;font:16px system-ui;max-width:1120px;margin:28px auto;padding:16px}video{width:100%;max-width:960px}a{color:#9dcfff}section{margin:28px 0}</style><h1>大色块像素风 R3 · 用户认可原版</h1><p>实际B1整数输出；原连续帧与PTS。此页的像素风与MC探索分别保存。</p>'+''.join(blocks),encoding='utf-8')
    files={str(p.relative_to(d)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in d.rglob('*') if p.is_file() and p.name!='manifest-sha256.json'}
    save_json(d/'manifest-sha256.json',files)
    print('APPROVED_REFERENCE_FROZEN',json.dumps(dict(counts=counts,crc=status['crc32_padded'],six_warp_improved=status['all_six_warp_improve_over_previous_pixel'],results=results,disk=status['disk'])),flush=True)

if __name__=='__main__':main()
