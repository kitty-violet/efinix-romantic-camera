"""Parameter-only spatial scale probe, before the existing integer colour head.

The existing DW kernels are changed, no filter is added after the network.
This probe is an initialization candidate, not a claimed trained improvement.
"""
import copy
import torch
from . import train_b1_style_recovery as t
from .minimum_area_blocks import OUT
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100633)
    root=OUT/'spatial-initialization';root.mkdir(exist_ok=False)
    parent=t.OUT/'clear-regions/best.pt';state=torch.load(parent,map_location='cpu',weights_only=True)
    d=t.data();records=[]
    profiles=[('low-half',['up1_low_dw'],.5),('low-full',['up1_low_dw'],1.),
              ('decoder-half',['up1_low_dw','up1_high_dw','up2_low_dw'],.5),
              ('decoder-full',['up1_low_dw','up1_high_dw','up2_low_dw'],1.)]
    for name,keys,alpha in profiles:
        net=t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
        for key in keys:
            w=net.weight[key];assert w.shape[1:]==(1,3,3)
            kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
            dc=w.sum((-1,-2),keepdim=True)
            w.copy_((1-alpha)*w+alpha*dc*kernel)
            mm=(w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/112).ceil().clamp(1,131071)
            getattr(net,key+'_m').copy_(mm.double()[None,:,None,None])
        dest=root/name;dest.mkdir()
        raw=net.export();(dest/'initial.bin').write_bytes(raw)
        actual=t.r.IntegerB1(t.r.runtime_from_image(raw));errors=[]
        for row in d['validation']:
            x=t.r.tensor(t.r.read(row['path']))[None];a=net(x,all_nodes=True);b,_=actual(x*128,True)
            errors.append(max(float((v-b[k]).abs().max()) for k,v in a.items()))
        assert max(errors)==0
        torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA,
                        parameter_initialization_only=True,parent_sha256=file_hash(parent)),dest/'initial.pt')
        t.preview(net,d['validation'],dest/'preview')
        records.append(dict(name=name,keys=keys,alpha=alpha,all_node_error=max(errors),
                            validation=t.evaluate(net,d['validation'])))
        print('SPATIAL_INIT',name,records[-1]['validation']['mean'],flush=True)
    write_json(root/'review.json',dict(records=records,additional_training_steps=0,added_inference_operators=0))


if __name__=='__main__':main()
