"""Matched boundary-width probe for the learned RGB region decoder.

The old count rounding hid intermediate field values. Removing that colour
lattice exposed the existing span5 transitions. Probe only that decision span
and the existing final spatial kernel; all learned parameters stay identical.
"""
import copy
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_native_boundary_phase import install
from .runtime import configure_workspace,contract_guard,write_json,file_hash


@torch.no_grad()
def main():
    configure_workspace();contract_guard();torch.set_num_threads(2)
    source=m.OUT/'joint-field-colour-qat/candidate_000500.pt';dest=m.OUT/'joint-field-boundary-probe';dest.mkdir(exist_ok=False)
    state=torch.load(source,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    initial=copy.deepcopy(net.state_dict());data=m.load_data(m.OUT/'hierarchical-targets-v2/data.json');images={};records=[]
    variants=[('current',5,False),('narrow',1,False),('narrow_center',1,True)]
    for name,span,center in variants:
        net.load_state_dict(initial)
        if name!='current':install(net,span,'halfgauss')
        if center:
            net.weight['up2_high_dw'][:3,0]=net.weight['up2_high_dw'].new_tensor([[8,8,8],[8,64,8],[8,8,8]])/128
        folder=dest/name;folder.mkdir();raw=net.export();assert len(raw)==20480
        (folder/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');ims={};checks=[]
        for row in data['validation']:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
            nodes=net(x,all_nodes=True);ref,_=actual(x*128,True)
            error=max(float((v-ref[k]).abs().max()) for k,v in nodes.items() if k!='input');assert error==0
            im=m.t.r.c.rgb_image(ref['rgb_head'][0]/128);im.save(folder/(row['id']+'.png'));ims[row['id']]=im
            field=ref['up2_low_dw'][:,:15]
            checks.append(dict(id=row['id'],all_node_error=error,
                intermediate_region_field_fraction=float(((field>0)&(field<127)).float().mean())))
        result=dict(state,model=copy.deepcopy(net.state_dict()),manual_boundary_change=dict(span=span,center_weight64=center),
            manual_decoder_parent=str(source),visual_acceptance=False)
        torch.save(result,folder/'candidate.pt');records.append(dict(variant=name,bytes=len(raw),records=checks))
        images[name]=ims;print('JOINT_FIELD_BOUNDARY',name,checks,flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    by={r['id']:r for r in data['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by[key]['target'])]+[images[n][key] for n,_,_ in variants]
        for j,(im,title) in enumerate(zip(pics,('修订目标','整数B1·当前过渡','整数B1·窄判定','整数B1·窄判定/轻AA'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');write_json(dest/'checks.json',dict(records=records,parent_sha256=file_hash(source),
        inference_added_operators=0,learned_parameters_identical_except_declared_fixed_boundary_coefficients=True,
        visual_acceptance=False,temporal_acceptance=False))


if __name__=='__main__':main()
