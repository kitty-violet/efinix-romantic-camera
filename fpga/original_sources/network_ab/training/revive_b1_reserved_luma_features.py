"""Revive two multiply-zero feature paths without changing initial RGB.

Duplicate one TRAIN-selected live path into reserved30/31 and attach equal
opposite existing readout coefficients. The initial contributions cancel
exactly in integer arithmetic, but their incoming gradients are nonzero.
Only these reserved intermediates change; no graph/node/parameter is added.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1,backend_palette
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install(net,source):
    assert 0<=source<30
    for key in ('res3_conv1__pw','res3_conv2__dw','res3_conv2__pw'):
        assert net.weight[key][30:].abs().max()==0 and net.bias[key][30:].abs().max()==0
    assert net.weight['res3_conv2__pw'][:,30:].abs().max()==0
    assert net.weight['up1_low_pw'][:,30:].abs().max()==0
    assert net.package['residual_encoding']['res3_add']==dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for key in ('res3_conv1__pw','res3_conv2__dw','res3_conv2__pw','up1_low_dw'):
        for dst in (30,31):
            net.weight[key][dst].copy_(net.weight[key][source]);net.bias[key][dst].copy_(net.bias[key][source])
            getattr(net,key+'_m')[:,dst].copy_(getattr(net,key+'_m')[:,source])
            if key=='res3_conv2__pw':
                # Route the duplicated DW, not the original path, so the new
                # upstream features actually receive a derivative.
                net.weight[key][dst,dst].copy_(net.weight[key][dst,source]);net.weight[key][dst,source].zero_()
    net.weight['up1_low_pw'][:14,30]=1/16;net.weight['up1_low_pw'][:14,31]=-1/16


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);state,base=load(a.checkpoint);data=m.load_data(a.data);out=Path(a.out);out.mkdir(exist_ok=False)
    tensor=lambda path:torch.from_numpy(m.t.r.read(path).transpose(2,0,1).copy())[None].float()/128-1
    ids=[];variance=torch.zeros(30)
    with torch.no_grad():
        for video in sorted({r['video'] for r in data['training']}):
            rows=[r for r in data['training'] if r['video']==video];row=rows[len(rows)//2]
            nodes=base(tensor(row['path']),all_nodes=True);variance+=nodes['up1_low_dw'][:,:30].std((0,2,3));ids.append(row['id'])
        src=int(variance.argmax());net=copy.deepcopy(base);install(net,src)
        assert torch.equal(backend_palette(base),backend_palette(net));raw=net.export();assert len(raw)==20480
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');records=[]
        reserved={'res3_conv1.pw','res3_conv2.dw','res3_conv2.pw','res3_add','up1_low_dw'}
        for row in data['validation']:
            x=tensor(row['path']);aa=base(x,all_nodes=True);bb=net(x,all_nodes=True);cc,_=actual(net.round(x*128),True)
            assert all(torch.equal(v,cc[k]) for k,v in bb.items())
            for k in aa:
                if k in reserved:assert torch.equal(aa[k][:,:30],bb[k][:,:30])
                else:assert torch.equal(aa[k],bb[k]),k
            assert torch.equal(bb['up1_low_dw'][:,30],bb['up1_low_dw'][:,31])
            assert torch.equal(bb['up1_low_dw'][:,30],bb['up1_low_dw'][:,src])
            records.append(dict(id=row['id'],rgb_error=0,nonreserved_nodes_error=0,actual_byte_all_nodes_error=0))
        saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
        saved.update(model=copy.deepcopy(net.state_dict()),revived_reserved_luma_features=[30,31],duplicated_source_channel=src,
            source_channel_chosen_on_train_ids=ids,revival_readout_weights=[.0625,-.0625],initial_rgb_exactly_preserved=True,visual_acceptance=False)
        torch.save(saved,out/'initial.pt')
    # Test their previously missing Jacobian through all four duplicated paths.
    net.__class__=ReadoutB1;net.zero_grad(set_to_none=True)
    row=next(r for r in data['training'] if r['id']==ids[0]);x=tensor(row['path'])[...,:320,:320]
    nodes=net(x,all_nodes=True);loss=nodes['up1_low_pw'][:,:14].square().mean();loss.backward();gradients={}
    for key in ('res3_conv1__pw','res3_conv2__dw','res3_conv2__pw','up1_low_dw'):
        g=net.weight[key].grad[30:];assert torch.isfinite(g).all() and (g.flatten(1).norm(dim=1)>0).all(),key
        gradients[key]=g.flatten(1).norm(dim=1).tolist()
    write_json(out/'review.json',dict(records=records,train_selection_ids=ids,duplicated_source_channel=src,gradient_norms=gradients,
        reserved_intermediates_intentionally_changed=sorted(reserved),rgb_and_nonreserved_nodes_error=0,backend_constant1215_error=0,bytes=20480,
        extra_inference_operations=0,visual_acceptance=False,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__)))
    print('RESERVED_PATH_REVIVED_PASS source',src,'RGB0/nonreserved0/all_actual_nodes0/four_paths_gradients',gradients,flush=True)


if __name__=='__main__':main()
