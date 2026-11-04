"""Bind completed media, actual reconstruction bytes and unchanged contract."""
import json
import re
from pathlib import Path
from .train_b1_reconstruction import OUT,ROOT,CONTRACT_SHA
from .runtime import file_hash,write_json


def main():
    delivery=json.loads((OUT/'delivery.json').read_text(encoding='utf-8'))
    media=json.loads((OUT/'videos/results.json').read_text(encoding='utf-8'))
    assert media['sha256']==delivery['package_sha256']==file_hash(delivery['weights'])
    assert file_hash(ROOT/'network_ab/model_b_contract.json')==CONTRACT_SHA
    assert {r['id'] for r in media['results']}=={'stairs','plaza'}
    rows=[]
    for row in media['results']:
        p=OUT/'videos'/(row['id']+'_integer1080.mp4')
        v=row['video'];expected=360 if row['id']=='stairs' else 450
        assert v['frames']==expected and v['seconds']==expected/30 and v['fps']==30 and v['size']==[1920,1080]
        assert file_hash(p)==v['sha256'] and p.stat().st_size==v['bytes']
        assert len(row['encoding'])==4
        rows.append(dict(scene=row['id'],path=str(p),frames=v['frames'],seconds=v['seconds'],
            bytes=v['bytes'],sha256=v['sha256'],encoding_rgb_mae=[r['mae'] for r in row['encoding']],
            measured_motion_output_mean=sum(x['output']['mean'] for x in row['motion'])/len(row['motion']),
            measured_motion_source_mean=sum(x['source']['mean'] for x in row['motion'])/len(row['motion'])))
    page=OUT/'watch.html';text=page.read_text(encoding='utf-8')
    text=text.replace('视频是实际20KiB系数包的软件整数复演，未上板。','视频是实际20KiB系数包的软件整数离线复演、30fps播放，未上板。')
    for rel in re.findall(r'src="([^"]+)"',text):
        assert (page.parent/rel).resolve().is_file(),rel
    page.write_text(text,encoding='utf-8')
    preview=OUT/'videos/preview.html'
    p=preview.read_text(encoding='utf-8').replace('B1颜色分区学习试验','B1原图重建能力测试').replace('B1真实整数包试验','B1实际整数包原图重建')
    preview.write_text(p,encoding='utf-8')
    code=['train_b1_reconstruction.py','refine_b1_reconstruction.py','publish_b1_reconstruction.py',
          'finalize_b1_reconstruction.py','render_b1_region_trial.py','verify_b1_reconstruction_delivery.py',
          'train_b1_learned_colour.py','probe_b1_cartoon_int8.py','b1_trained_image.py','fpga_video_resize.py']
    write_json(OUT/'final_verification.json',dict(status='RECONSTRUCTION_EXPERIMENT_DELIVERY_VERIFIED_NOT_QUALITY_PASS',
        package_sha256=delivery['package_sha256'],contract_sha256=CONTRACT_SHA,
        no_hardware_changes=True,actual_board=False,video_rows=rows,watch_sources_exist=True,
        videos_fully_decoded_by_renderer=True,visual_quality_accepted=False,
        temporal_metric_not_proof_of_no_flicker=True,
        code_hashes={name:file_hash(ROOT/'network_ab/training'/name) for name in code}))
    print(json.dumps(rows,ensure_ascii=False))
    print('RECON_FINAL_VERIFIED',flush=True)


if __name__=='__main__':main()
