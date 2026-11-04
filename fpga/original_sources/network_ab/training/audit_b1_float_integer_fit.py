"""Separate optimization fit from integer replay on the declared TRAIN set."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_count_regions import count_targets,count_loss
from .fit_b1_sixteen_regions import palette_key,region_class_map
from .runtime import file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.set_num_threads(2);state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert state.get('seven_scene_fit_diagnostic')
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin')
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding']
    palette=torch.tensor(state['palette']);pk=palette_key(palette);by_id={r['id']:r for r in m.load_data()['training']}
    manifest=state['args']['class_manifest'];assert file_hash(manifest)==state['temporal_class_manifest_sha256']
    overrides=json.loads(Path(manifest).read_text(encoding='utf-8'))['overrides'];records=[]
    for key in state['fit_ids']:
        row=by_id[key];assert row['split']=='train'
        rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1).copy(),dtype=torch.float32)[None]/128-1
        codes=region_class_map(row['target'],row['region'],pk)
        if key in overrides:
            entry=overrides[key];assert file_hash(entry['path'])==entry['sha256']
            with np.load(entry['path']) as z:codes=z['classes'].copy()
        truth=count_targets(torch.tensor(codes.astype('int64'))[None])
        pure=torch.stack([t.amax(1)>.999 for t in truth]).all(0)
        wanted=truth[0].argmax(1)*25+truth[1].argmax(1)*5+truth[2].argmax(1)
        record=dict(id=key);predictions=[]
        for quantized in (False,True):
            net.quantized=quantized;nodes=net(x,all_nodes=True)
            logits=(nodes['up1_high_dw'][:,:15]-63.5)/4;bits=logits>=0
            if quantized:assert torch.equal(bits,nodes['up2_low_dw'][:,:15]>0)
            code=bits[:,:7].sum(1)*25+bits[:,7:11].sum(1)*5+bits[:,11:].sum(1)
            predictions.append(code)
            record['integer' if quantized else 'fp32']=dict(count_loss=float(count_loss(logits,truth)),
                accuracy_pure=float((code[pure]==wanted[pure]).float().mean()))
        record['fp32_integer_class_disagreement']=float((predictions[0][pure]!=predictions[1][pure]).float().mean())
        records.append(record);print('FLOAT_INTEGER_FIT',record,flush=True)
    write_json(a.out,dict(checkpoint=a.checkpoint,sha256=file_hash(a.checkpoint),records=records,
        selection='declared TRAIN fit only',fp32_is_threshold_diagnostic_not_rgb_delivery=True,
        means={kind:sum(r[kind]['accuracy_pure'] for r in records)/len(records) for kind in ('fp32','integer')}))


if __name__=='__main__':main()
