"""Freeze Y labels from un-antialiased native region colours before cropping.

Only auxiliary training labels change. Source frames, RGB style targets,
region geometry, split and flow remain byte-identical to the parent manifest.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from PIL import Image
from .train_b1_luma_chroma_interfaces import luminance_codes
from .qat_b1_deploy_preview import Round
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_joint_interface_targets import joint_interface_fields
from .probe_b1_monotone_pair_decoder import make_board
from .audit_b1_rim_tolerance import rim_terms
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);data=copy.deepcopy(m.load_data(a.data));records=[];rnd=Round()
    for split in ('training','validation','reserved_test'):
        for row in data[split]:
            hard=m.t.r.read(row['hard_region_target']);lab=m.load_region(row['region']).astype(np.int64)
            ids,first=np.unique(lab,return_index=True);assert ids[0]>=0
            colour=np.zeros((int(ids[-1])+1,3),np.uint8);colour[ids]=hard.reshape(-1,3)[first]
            assert np.array_equal(colour[lab],hard),'Hard region target must really be flat per label'
            c=colour.astype(np.float64);y=(c[:,0]+2*c[:,1]+c[:,2])/4
            level=np.floor((y-16)/16+.5).clip(0,14).astype(np.uint16);codes=81*level[lab]+40
            path=out/(row['sha256'][:20]+'.npz');np.savez_compressed(path,codes=codes)
            row.update(native_luma_codes=str(path),native_luma_codes_sha256=file_hash(path),
                       native_luma_colour_source_sha256=file_hash(row['hard_region_target']))
            # Actual old code, including PyTorch rounding/region reduction.
            t=torch.from_numpy(m.t.r.read(row['target']).transpose(2,0,1).copy())[None].float()/128-1
            old=luminance_codes(t,torch.from_numpy(lab)[None],rnd)[0].numpy()//81;delta=old.astype(np.int16)-level[lab].astype(np.int16)
            records.append(dict(id=row['id'],split=split,changed=float((delta!=0).mean()),old_brighter=float((delta>0).mean()),old_darker=float((delta<0).mean()),max_levels=int(np.abs(delta).max())))
        print('NATIVE_LUMA_LABELS',split,len(data[split]),flush=True)
    data.update(native_luma_definition='nearest15-level colour from verified flat hard_region_target BEFORE antialias/crop/flow',
                parent_luma_manifest_sha256=file_hash(a.data),native_luma_preparation_sha256=file_hash(__file__))
    write_json(out/'data.json',data);write_json(out/'review.json',dict(records=records,source_sha256=file_hash(__file__),
        unchanged_source_rgb_geometry_splits=True,labels_training_only=True,visual_acceptance=False))
    _,net=load(a.checkpoint);boards=[{},{},{},{}];diagnostics=[]
    for row in data['validation']:
        tensor=lambda path:torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
        x=tensor(row['path']);t=tensor(row['target']);lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None]
        nodes=net(x,all_nodes=True);old=luminance_codes(t,lab,rnd);new=torch.from_numpy(np.load(row['native_luma_codes'])['codes'].astype(np.int64))[None]
        boards[0][row['id']]=Image.fromarray(m.t.r.read(row['target']));boards[3][row['id']]=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128)
        for j,(label,code) in enumerate((('soft_mean_labels',old),('native_hard_labels',new)),1):
            z=nodes['up1_low_pw'].clone();z[:,:14]=net.round(joint_interface_fields(code)[:,:14]);p=fragment(net,z,'up1_low_pw')['rgb_head']/128
            boards[j][row['id']]=m.t.r.c.rgb_image(p[0]);rim,_=rim_terms(p,t,lab,2)
            diagnostics.append(dict(id=row['id'],mode=label,mae=float((p-t).abs().mean()*128),tight_rim=float(rim)))
    make_board(boards,['原风格目标','旧亮度标签注入：诊断','原生纯色标签注入：诊断','原实际预测'],out/'known-label-comparison.png')
    write_json(out/'known-label-review.json',dict(records=diagnostics,ground_truth_injection_not_deployable=True,visual_acceptance=False))


if __name__=='__main__':main()
