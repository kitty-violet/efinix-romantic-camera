"""One rounding per colour coordinate after complete bit-group summation.

Avoid rounding partial sums independently: equal colours with different bit
assignments must not acquire a phantom seam. Existing B1 layers/bytes only.
"""
import argparse
import copy
import hashlib
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .audit_b1_native_colour_regions import spatial_permutation_audit
from .build_b1_ordered_palette import GROUPS
from .runtime import write_json,file_hash,contract_guard


@torch.no_grad()
def install(net,kind):
    expected=net.weight['rgb_head'].new_tensor([[2,2,-2],[2,0,2],[2,-2,-2]])
    assert torch.equal(net.weight['rgb_head'][:,:3,0,0],expected)
    for i,(a,z) in enumerate(GROUPS):
        w=net.weight['up2_low_pw'][i,a:z]
        assert torch.equal(w,w[:1].expand_as(w))
    net.weight['up2_high_dw'].zero_();net.bias['up2_high_dw'].zero_()
    if kind=='gauss':kernel=expected.new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    elif kind=='box':kernel=expected.new_ones(3,3)
    else:raise ValueError(kind)
    for i,increment in enumerate((16,9,6)):
        scale=(131072+int(kernel.sum())*increment-1)//(int(kernel.sum())*increment)
        net.up2_high_dw_m[:,i]=scale
        net.weight['up2_high_dw'][i,0]=kernel*(scale/131072)
        net.weight['rgb_head'][:,i,0,0]=expected[:,i]*increment
    net.rgb_head_m.fill_(65536)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    torch.set_num_threads(2);contract_guard();dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert not state.get('packed_native_count_decoder')
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    base=copy.deepcopy(net.state_dict());rows=m.load_data()['validation'];all_images={};records=[]
    patterns=((torch.arange(32768)[:,None]>>torch.arange(15)[None])&1).float()*127
    z=torch.zeros(32768,16,1,1);z[:,:15,0,0]=patterns
    original=fragment(net,z,'up2_low_dw')['rgb_head'].clone()
    for kind in ('current','gauss','box'):
        net.load_state_dict(base)
        if kind!='current':install(net,kind)
        error=float((fragment(net,z,'up2_low_dw')['rgb_head']-original).abs().max());assert error==0
        spatial=spatial_permutation_audit(net);assert max(r['max_rgb_error'] for r in spatial)==0
        sub=dest/kind;sub.mkdir();blob=net.export();(sub/'coefficients-padded.bin').write_bytes(blob)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(blob),device='cpu');tested=[];images={}
        for row in rows:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
            nodes=net(x,all_nodes=True);ref,_=actual(x*128,True)
            node_error=max(float((ref[k]-v).abs().max()) for k,v in nodes.items() if k!='input');assert node_error==0
            if kind!='current':
                count=ref['up2_high_dw'][:,:3]
                assert torch.all(count==count.round()) and count.min()>=0
                assert count[:,0].max()<=7 and count[:,1:].max()<=4
            im=m.t.r.c.rgb_image(ref['rgb_head'][0]/128);images[row['id']]=im;im.save(sub/(row['id']+'.png'))
            tested.append(dict(id=row['id'],all_node_error=node_error))
        all_images[kind]=images
        if kind!='current':
            result=dict(state);result.update(model=copy.deepcopy(net.state_dict()),parent=a.checkpoint,
                group_native_count_decoder=kind,manual_decoder_change=True,training_proxy_not_yet_verified=True,
                visual_acceptance=False,minimum_area_inference_guarantee=False)
            torch.save(result,sub/'manual_candidate.pt')
        records.append(dict(kind=kind,flat_32768_pattern_error=error,spatial_same_colour=spatial,
            records=tested,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
        print('GROUP_NATIVE_COUNTS',kind,error,flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pictures=[Image.open(by_id[key]['target'])]+[all_images[k][key] for k in ('current','gauss','box')]
        for j,(im,title) in enumerate(zip(pictures,('区域目标','原整数B1','单次720计数·高斯核','单次720计数·方核'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.checkpoint,parent_sha256=file_hash(a.checkpoint),records=records,
        learned_classifier_changed=False,added_operators=0,postfilter=False,actual_board=False,visual_acceptance=False,
        preserves_flat_palette=True,same_colour_seam_probe_passed=True,minimum_area_inference_guarantee=False))


if __name__=='__main__':main()
