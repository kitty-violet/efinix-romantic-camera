"""Remove isolated ordinal decisions in existing B1 DW layers, no postfilter."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);assert state.get('manual_sector_selection')
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True;net.package['residual_encoding']=state['residual_encoding']
    k='res3_conv2__dw';net.weight[k].zero_();net.bias[k].zero_()
    net.weight[k][:15,0,1,1]=127*(131071/131072);net.bias[k][:15]=-63*127*(131071/131072)
    getattr(net,k+'_m').fill_(131071)
    k='up1_low_dw';net.weight[k].zero_();net.bias[k].zero_();net.weight[k][:15,0]=1;net.bias[k][:15]=-4*127
    getattr(net,k+'_m').fill_(2048)
    rows=m.load_data()['validation'];records=[]
    for row in rows:
        nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
        for name in ('res3_conv2.dw','up1_low_dw','up2_low_dw'):
            z=nodes[name][:,:15];assert torch.all((z==0)|(z==127)),name
            for start,end in ((0,7),(7,11),(11,15)):assert torch.all(z[:,start:end-1]>=z[:,start+1:end])
        records.append(dict(id=row['id'],binary_and_monotonic=True))
    export(net,rows,dest/'package');m.t.preview(net,rows,dest/'preview')
    state.update(model=net.state_dict(),parent=a.warm,manual_ordinal_majority=True)
    torch.save(state,dest/'initial.pt');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,parent_sha256=file_hash(a.warm),changed_layers=['res3_conv2.dw','up1_low_dw'],
        added_operators=0,postfilter=False,minimum_area_guaranteed=False,visual_acceptance=False))
    print('SECTOR_CLEANUP_COMPLETE',flush=True)


if __name__=='__main__':main()
