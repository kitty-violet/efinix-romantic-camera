"""Keep hard region fields but learn their shared RGB colour vectors.

Controlled response to two measured failures: fixed8x5x5 colour lattice loses
contrast/hue, whereas a freely continuous RGB decoder returns soft photographs.
The existing threshold node remains; its15 region fields now feed a learned
RGB matrix. No separately counted Y/Co/Cg palette, no new inference operation.
This does NOT guarantee connected-component size or perfect shared boundaries.
"""
import argparse
import copy
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import JointRegionColourTraining,evaluate,preview,regression
from .train_b1_aligned_decoder_regions import DecoderAlignedB1,regression as gradient_regression
from .train_b1_quantum_regions import RowScaledAdam
from .build_b1_binary_palette import calibrate_rows
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from . import train_b1_late_region_supervision as driver


@torch.no_grad()
def install(net):
    old=net.weight['up2_low_pw'].clone()
    convert=old.new_tensor([[1,1,-1],[1,0,1],[1,-1,-1]])
    w=net.weight['up2_low_pw'];w.zero_();w[:3,:,0,0]=convert@old[:3,:,0,0]
    net.bias['up2_low_pw'].zero_();net.bias['up2_low_pw'][:3]=(net.bias['rgb_head']+128)/2
    calibrate_rows(net,['up2_low_pw'])
    w=net.weight['up2_high_dw'];w.zero_();w[:3,0]=w.new_tensor([[14,14,14],[14,16,14],[14,14,14]])/128
    net.bias['up2_high_dw'].zero_();net.up2_high_dw_m.fill_(1024)
    net.weight['rgb_head'].zero_()
    for c in range(3):net.weight['rgb_head'][c,c,0,0]=2
    net.bias['rgb_head'].fill_(-128);net.rgb_head_m.fill_(65536)


class JointFieldColourTraining(JointRegionColourTraining):
    def __init__(self,args,data):
        super().__init__(args,data);assert args.qat
        state=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert state.get('joint_field_colour_decoder')
        self.net.__class__=DecoderAlignedB1
        fixed_names={'up1_high_dw','up2_low_dw','up2_high_dw','rgb_head'}
        self.fixed={};groups=[]
        for key,w in self.net.weight.items():
            b=self.net.bias[key];train=key not in fixed_names
            w.requires_grad_(train);b.requires_grad_(train)
            if train:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(s/(1/256))[:,None,None,None]),dict(params=[b],lr=.002)])
            else:self.fixed[key]=(w.detach().clone(),b.detach().clone())
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]

    def update(self,native=None):
        result=super().update(native)
        with torch.no_grad():
            for key,w in self.net.weight.items():
                if w.requires_grad:
                    limit=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
                    w.copy_(w.maximum(-limit).minimum(limit))
            for key,(w,b) in self.fixed.items():
                assert torch.equal(w,self.net.weight[key]) and torch.equal(b,self.net.bias[key])
        return result

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(joint_field_colour_decoder=True,hard_region_fields=15,
            fixed_global_colour_lattice=False,learned_shared_rgb_matrix=True,
            inference_added_operators=0,minimum_area_inference_guarantee=False)
        torch.save(state,path)


def build():
    configure_workspace();contract_guard();torch.set_num_threads(2)
    source=m.OUT/'native-boundary-field-probe/span5/candidate.pt';dest=m.OUT/'joint-field-colour-initial';dest.mkdir(exist_ok=False)
    state=torch.load(source,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
    install(net);raw=net.export();assert len(raw)==20480
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    records=[]
    for row in m.load_data()['validation']:
        rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)/128-1
        with torch.no_grad():nodes=net(x,all_nodes=True);replay,_=actual(x*128,True)
        error=max(float((v-replay[k]).abs().max()) for k,v in nodes.items() if k!='input');assert error==0
        records.append(dict(id=row['id'],all_node_error=error))
        m.t.r.c.rgb_image(replay['rgb_head'][0]/128).save(dest/(row['id']+'.png'))
    (dest/'coefficients-padded.bin').write_bytes(raw)
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
        joint_field_colour_decoder=True,parent=str(source),fixed_global_colour_lattice=False,
        minimum_area_inference_guarantee=False,visual_acceptance=False),dest/'initial.pt')
    gradient_regression(dest/'initial.pt')
    write_json(dest/'checks.json',dict(bytes=len(raw),all_node_records=records,parent_sha256=file_hash(source),
        code_sha256=file_hash(__file__),inference_added_operators=0,minimum_area_inference_guarantee=False,visual_acceptance=False))
    print('JOINT_FIELD_BUILD_PASS',flush=True)


if __name__=='__main__':
    import sys
    if '--build' in sys.argv:build()
    else:
        regression();driver.main(training_factory=JointFieldColourTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
